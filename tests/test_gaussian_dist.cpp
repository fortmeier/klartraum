/**
 * TESTS:
 * - cullsAndCompactsVisibleSplatsInBackToFrontOrder: dispatches GaussianDist
 *   (gsplat_dist.comp) over a small set of splats placed along and off the
 *   camera's view axis, and confirms drawArgs.instanceCount equals the number
 *   of splats inside the (dilated) frustum, that indices[0..instanceCount)
 *   contains exactly those splats' ids (compaction correctness), and that
 *   their encoded depth keys decrease with increasing distance from the
 *   camera — farthest splat gets the smallest key, so the ascending sort in
 *   stage B draws back-to-front, the invariant premultiplied "over" blending
 *   relies on
 * - sortsCompactedSplatsBackToFrontViaRadixSort: chains GaussianDist into the
 *   existing (reused, unmodified) RadixSort over keys/indices ping-pong
 *   buffers seeded each frame with the 0xFFFFFFFF sentinel (setRecordToFill),
 *   dispatched over the full splat count N, and confirms the sorted indices
 *   buffer holds exactly the visible splats in correct back-to-front order in
 *   its [0, instanceCount) prefix — i.e. stage A + the sentinel-fill sizing
 *   scheme + the existing radix sort compose into a correct stage A/B pipeline
 *   without any sort-shader changes
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
    // dist binding 5: (r,g,b,alpha). The opacity cull is disabled here
    // (alphaThreshold 0), so alpha=1 keeps every splat — this test exercises
    // frustum culling only.
    auto colorsAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(vc, 1,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    drawArgs->setRecordToZeroRange(offsetof(VkDrawIndirectCommand, instanceCount), sizeof(uint32_t));

    auto dist = std::make_shared<GaussianDist>(vc, "shaders/gsplat/gsplat_dist.comp.spv");
    dist->setInput(positions,   0);
    dist->setInput(cameraUBO,   1);
    dist->setInput(keys,        2);
    dist->setInput(indices,     3);
    dist->setInput(drawArgs,    4);
    dist->setInput(colorsAlpha, 5);
    dist->setGroupCountX((numSplats + 255) / 256);
    dist->setPushConstants({{numSplats, 0.0f}});

    auto computegraph = ComputeGraph(vc, 1);
    computegraph.compileFrom(dist);

    positions->getBuffer(0).memcopyFrom(positionData);
    colorsAlpha->getBuffer(0).memcopyFrom(std::vector<glm::vec4>(numSplats, glm::vec4(1.0f)));
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

    // Splat ids 0, 1, 2 were placed at strictly increasing distance from the
    // camera (fractions 0.3, 0.6, 1.0 along the view axis); for back-to-front
    // draw order the farthest splat must get the smallest key (drawn first,
    // nearer splats blend over it), so keys must be in strictly *decreasing*
    // order with increasing distance.
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
    EXPECT_GT(key0, key1) << "farther splat must get the smaller key (sorts first, back-to-front)";
    EXPECT_GT(key1, key2) << "farther splat must get the smaller key (sorts first, back-to-front)";
}

TEST(GaussianDist, sortsCompactedSplatsBackToFrontViaRadixSort) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setDistance(5.0f);
    orbit.update(cameraUBO->ubo);

    glm::vec3 camPos = glm::vec3(cameraUBO->ubo.cameraWorldPos);
    glm::vec3 toOrigin = -camPos;

    // Visible splats placed at distinct, scrambled-order depths so the sort
    // must actually reorder them; two splats are deliberately culled.
    const uint32_t numSplats = 6;
    std::vector<glm::vec3> positionData = {
        camPos + toOrigin * 0.8f,                              // 0: far,      visible
        camPos + toOrigin * 0.2f,                              // 1: near,     visible
        camPos - toOrigin * 2.0f,                              // 2: behind camera, culled
        camPos + toOrigin * 0.5f,                              // 3: mid,      visible
        camPos + toOrigin * 0.5f + glm::vec3(1000.0f, 0, 0),   // 4: off-axis, culled
        camPos + toOrigin * 0.35f,                             // 5: near-mid, visible
    };
    // Farthest-first (back-to-front) order of the visible splats by id.
    std::vector<uint32_t> expectedBackToFront = {0, 3, 5, 1};

    auto positions = std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // keys/indices ping-pong pair A (read by dist, read+written by the sort)
    // and B (sort scratch only). The sort runs 8 passes (even), so the final
    // sorted result lands back in A — the buffers dist already wrote into.
    auto keysA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto indicesA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto keysB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto indicesB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Sentinel sizing scheme (RASTER_BACKEND_STATUS.md): reset the whole key
    // buffer to 0xFFFFFFFF (> any encodeDepthKey output) each frame, then sort
    // the fixed full count N — culled/garbage slots sort to the tail beyond
    // instanceCount and are never read by the indirect draw.
    keysA->setRecordToFill(0xFFFFFFFFu);

    auto drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(vc, 1,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    drawArgs->setRecordToZeroRange(offsetof(VkDrawIndirectCommand, instanceCount), sizeof(uint32_t));

    // dist binding 5: (r,g,b,alpha). Opacity cull disabled (alphaThreshold 0),
    // alpha=1 keeps every visible splat — this test exercises the sort chain.
    auto colorsAlpha = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vc, numSplats,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    auto dist = std::make_shared<GaussianDist>(vc, "shaders/gsplat/gsplat_dist.comp.spv");
    dist->setInput(positions,   0);
    dist->setInput(cameraUBO,   1);
    dist->setInput(keysA,       2);
    dist->setInput(indicesA,    3);
    dist->setInput(drawArgs,    4);
    dist->setInput(colorsAlpha, 5);
    dist->setGroupCountX((numSplats + 255) / 256);
    dist->setPushConstants({{numSplats, 0.0f}});

    // Reused, unmodified radix sort (existing key/index parallel-buffer sort —
    // see RASTER_BACKEND_STATUS.md note that it already operates on plain
    // (uint key, uint index) pairs, not struct payloads as the guide assumed).
    const uint32_t numBins = 16;     // 4-bit radix digit
    const uint32_t numSortWGs = 1;   // numSplats is tiny; one workgroup suffices
    auto scratchHist = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins * numSortWGs);
    scratchHist->setRecordToZero(true);
    auto scratchCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numBins);
    scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, numSortWGs + 1);
    scratchOffsets->setRecordToZero(true);
    auto totalCount = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vc, 1);
    totalCount->setRecordToZero(true);

    auto sortOp = std::make_shared<RadixSort>(vc, std::vector<std::string>{
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    });
    sortOp->setInput(dist,   0, 2);  // keysA    (dist's output index 2)
    sortOp->setInput(dist,   1, 3);  // indicesA (dist's output index 3)
    sortOp->setInput(keysB,    2);
    sortOp->setInput(indicesB, 3);
    sortOp->addScratchBufferElement(scratchCounts,  true);
    sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(totalCount,     false);
    sortOp->addScratchBufferElement(scratchHist,    true);
    sortOp->setGroupCountX(numSortWGs);
    {
        std::vector<SortPushConstants> pcs;
        for (uint32_t i = 0; i < 32 / 4; ++i)
            pcs.push_back({i, numSplats, numBins});
        sortOp->setPushConstants(pcs);
    }

    auto computegraph = ComputeGraph(vc, 1);
    computegraph.compileFrom(sortOp);

    positions->getBuffer(0).memcopyFrom(positionData);
    colorsAlpha->getBuffer(0).memcopyFrom(std::vector<glm::vec4>(numSplats, glm::vec4(1.0f)));
    cameraUBO->update(0);

    computegraph.submitAndWait(vc.getGraphicsQueue(), 0);

    std::vector<VkDrawIndirectCommand> argsVec(1);
    drawArgs->getBuffer(0).memcopyTo(argsVec);
    uint32_t visibleCount = argsVec[0].instanceCount;
    ASSERT_EQ(visibleCount, expectedBackToFront.size());

    std::vector<uint32_t> sortedIndices(numSplats);
    indicesA->getBuffer(0).memcopyTo(sortedIndices);
    std::vector<uint32_t> actualOrder(sortedIndices.begin(), sortedIndices.begin() + visibleCount);
    EXPECT_EQ(actualOrder, expectedBackToFront);
}
