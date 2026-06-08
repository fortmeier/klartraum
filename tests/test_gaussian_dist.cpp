/**
 * TESTS:
 * - cullsAndCompactsVisibleSplatsInBackToFrontOrder: dispatches GaussianDist
 *   (gsplat_dist.comp) over a small set of splats placed along and off the
 *   camera's view axis, and confirms drawArgs.instanceCount equals the number
 *   of splats inside the (dilated) frustum, that indices[0..instanceCount)
 *   contains exactly those splats' ids (compaction correctness), and that
 *   their encoded depth keys preserve the splats' back-to-front view-space
 *   depth order (the invariant the radix sort in stage B relies on)
 **/
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>
#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"
#include "klartraum/interface_camera_orbit.hpp"

using namespace klartraum;

TEST(GaussianDist, cullsAndCompactsVisibleSplatsInBackToFrontOrder) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setDistance(5.0f);
    orbit.update(cameraUBO->ubo);

    // The orbit camera always looks at the origin, so placing splats along the
    // camera-to-origin ray at increasing fractions gives strictly increasing
    // view-space depth (= strictly increasing back-to-front order).
    glm::vec3 camPos = glm::vec3(cameraUBO->ubo.cameraWorldPos);
    glm::vec3 toOrigin = -camPos;

    const uint32_t numSplats = 5;
    std::vector<glm::vec3> positionData = {
        camPos + toOrigin * 0.3f,                              // 0: near, visible
        camPos + toOrigin * 0.6f,                              // 1: mid, visible
        camPos + toOrigin * 1.0f,                              // 2: far (at origin), visible
        camPos - toOrigin * 2.0f,                              // 3: behind the camera, culled
        camPos + toOrigin * 0.5f + glm::vec3(1000.0f, 0, 0),   // 4: far off-axis, culled
    };
    std::set<uint32_t> expectedVisible = {0, 1, 2};

    auto positions = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto keys = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto indices = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(vc, 1,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    drawArgs->setRecordToZeroRange(offsetof(VkDrawIndirectCommand, instanceCount), sizeof(uint32_t));

    auto dist = std::make_shared<GaussianDist>(vc, "shaders/gsplat/gsplat_dist.comp.spv");
    dist->setInput(positions, 0);
    dist->setInput(cameraUBO, 1);
    dist->setInput(keys,      2);
    dist->setInput(indices,   3);
    dist->setInput(drawArgs,  4);
    dist->setGroupCountX((numSplats + 255) / 256);
    dist->setPushConstants({{numSplats, 0.0f}});

    auto computegraph = ComputeGraph(vc, 1);
    computegraph.compileFrom(dist);

    positions->getBuffer(0).memcopyFrom(positionData);
    cameraUBO->update(0);

    computegraph.submitAndWait(vc.getGraphicsQueue(), 0);

    std::vector<VkDrawIndirectCommand> argsVec(1);
    drawArgs->getBuffer(0).memcopyTo(argsVec);
    VkDrawIndirectCommand argsResult = argsVec[0];

    ASSERT_EQ(argsResult.instanceCount, expectedVisible.size());

    std::vector<uint32_t> indicesResult(numSplats);
    std::vector<uint32_t> keysResult(numSplats);
    indices->getBuffer(0).memcopyTo(indicesResult);
    keys->getBuffer(0).memcopyTo(keysResult);

    std::set<uint32_t> visibleIds;
    for (uint32_t slot = 0; slot < argsResult.instanceCount; slot++) {
        visibleIds.insert(indicesResult[slot]);
    }
    EXPECT_EQ(visibleIds, expectedVisible);

    // Splat ids 0, 1, 2 were placed in strictly increasing back-to-front order
    // (increasing distance from the camera along the view axis); the encoded
    // keys must preserve that order so the radix sort reproduces it.
    auto slotOf = [&](uint32_t id) -> uint32_t {
        for (uint32_t slot = 0; slot < argsResult.instanceCount; slot++) {
            if (indicesResult[slot] == id) return slot;
        }
        ADD_FAILURE() << "id " << id << " not found among compacted slots";
        return 0;
    };
    uint32_t key0 = keysResult[slotOf(0)];
    uint32_t key1 = keysResult[slotOf(1)];
    uint32_t key2 = keysResult[slotOf(2)];
    EXPECT_LT(key0, key1) << "nearer splat must sort before farther splat (back-to-front)";
    EXPECT_LT(key1, key2) << "nearer splat must sort before farther splat (back-to-front)";
}
