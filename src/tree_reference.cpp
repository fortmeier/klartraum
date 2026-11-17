// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_reference.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/tree/tree_placement.hpp"

namespace klartraum {

namespace {

// Reads an image the passes leave in TRANSFER_SRC_OPTIMAL back as BGRA bytes.
std::vector<uint8_t> readTarget(VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return pixels;
}

// The key of a tree's foot, for telling the base renderer's near trees apart.
std::pair<int, int> footKey(glm::vec3 foot) {
    return {int(std::lround(foot.x * 1e4f)), int(std::lround(foot.z * 1e4f))};
}

} // namespace

TreeReferenceRenderer::TreeReferenceRenderer(VulkanContext& vulkanContext, const BonsaiScene& scene, VkExtent2D extent,
                                             TreeReferenceConfig config)
    : vulkanContext(vulkanContext),
      scene(scene),
      extent(extent),
      config(config) {
    target = std::make_shared<OffscreenTarget>(vulkanContext, extent, 1);
    BonsaiRenderConfig baseConfig;
    baseConfig.mode = BonsaiRenderMode::Classical;
    baseConfig.extent = extent;
    baseConfig.supersample = 1;
    baseConfig.farTrees = false;
    baseConfig.storeDepth = true;
    baseConfig.reducedTrees = 0;
    baseConfig.numberPaths = 1;
    base = std::make_unique<BonsaiRenderer>(vulkanContext, scene, target, baseConfig);
    baseGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    baseGraph->compileFrom(base->getOutput());
    base->initialize(1);

    // Room for `batch` trees of either type at the tallest height of the placement.
    const TreeFrame frame = defaultTreeFrame();
    const TreePlacementParams& placement = scene.placement;
    TreeCapacity capacity{};
    for (uint32_t type = 0; type < scene.types.size(); ++type) {
        TreeInstance tallest = treeInstanceFor(TreeSite{}, scene.types, scene.config.seed, config.variants);
        tallest.ids.x = type;
        tallest.posScale =
            glm::vec4(0.0f, 0.0f, 0.0f,
                      placement.size.x * (1.0f + placement.size.y) / placement.grid.w / scene.types[type].shapeBase.z);
        const std::vector<TreeInstance> trees(config.batch, tallest);
        const TreeCapacity c = TreeCapacity::fromReference(scene.types, trees, frame);
        for (size_t level = 0; level < capacity.stems.size(); ++level) {
            capacity.stems[level] = std::max(capacity.stems[level], c.stems[level]);
        }
        capacity.segments = std::max(capacity.segments, c.segments);
        capacity.leaves = std::max(capacity.leaves, c.leaves);
    }
    TreeInstance hidden{};
    hidden.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    generator = std::make_shared<TreeGenerator>(vulkanContext, scene.types,
                                                std::vector<TreeInstance>(config.batch, hidden), capacity, frame);
    settings = std::make_shared<HostValues<TreeRenderSettings>>(
        vulkanContext, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
    std::shared_ptr<CameraUboType> camera = base->getCamera();
    stems = std::make_unique<TreeStemRenderer>(vulkanContext, *generator, camera, settings, extent);
    leaves = std::make_unique<TreeLeafRenderer>(vulkanContext, *generator, scene.types, camera, settings, extent);
    auto toGeneral = std::make_shared<ImageToGeneral>();
    toGeneral->setInput(std::make_shared<OffscreenTargetView>(target), 0);
    pass = std::make_shared<RenderPass>(vulkanContext.getSwapChainImageFormat(), extent);
    pass->setInput(toGeneral, 0);
    pass->setInput(camera, 1);
    pass->setLoadExisting(true);
    pass->setDepthFrom(base->getPass());
    pass->addComputeDependency(stems->getClassify());
    pass->addComputeDependency(leaves->getClassify());
    pass->addComputeDependency(settings);
    pass->addDrawComponent(stems->getDrawComponent());
    pass->addDrawComponent(leaves->getDrawComponent());
    batchGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    batchGraph->compileFrom(pass);
}

TreeReferenceRenderer::Frame TreeReferenceRenderer::render(glm::vec3 eye, glm::vec3 lookAt, float fovDegrees,
                                                           float nearPlane, float farPlane, float season,
                                                           float timeOfDay) {
    CameraMVP& ubo = base->getCamera()->ubo;
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(eye, lookAt, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj =
        glm::perspective(glm::radians(fovDegrees), float(extent.width) / float(extent.height), nearPlane, farPlane);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
    base->getCamera()->update(0);
    base->setSeason(season);
    base->setTime(0.0f, timeOfDay, season, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));

    // The base view: terrain, sky and the near trees, without the forest canopy, until its tiles are there.
    auto withoutCanopy = [&]() {
        TreeRenderSettings values = base->getSettings().values().at(0);
        values.canopyBand = glm::vec4(1e-4f, 2e-4f, 0.0f, 0.0f);
        base->getSettings().set(0, values);
    };
    for (int i = 0; i < 40; ++i) {
        base->setView(eye);
        withoutCanopy();
        baseGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
        if (base->isSettled()) {
            break;
        }
    }
    base->setView(eye);
    withoutCanopy();
    baseGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
    Frame frame;
    const std::vector<uint8_t> baseImage = readTarget(vulkanContext, target->getImage(0), extent);

    // The far trees: in the view, within range, tall enough on screen, and not among the base's near trees.
    std::set<std::pair<int, int>> near;
    for (const TreeInstance& inst : base->getGenerator().getInstances().values()) {
        if (inst.posScale.w > 0.0f) {
            near.insert(footKey(glm::vec3(inst.posScale)));
        }
    }
    const float m = 1.0f / scene.config.scale.metersPerUnit();
    const float range = config.rangeMeters * m;
    const float cell = scene.placement.grid.x;
    const glm::mat4 clip = ubo.proj * ubo.view;
    const float focal = std::abs(ubo.proj[1][1]) * 0.5f * float(extent.height);
    const std::vector<TreeSite> sites =
        treeSitesIn(scene.terrain, scene.placement, int(std::floor((eye.x - range) / cell)),
                    int(std::floor((eye.z - range) / cell)), int(std::ceil((eye.x + range) / cell)),
                    int(std::ceil((eye.z + range) / cell)));
    std::vector<TreeInstance> far;
    for (const TreeSite& site : sites) {
        const glm::vec3 foot(site.posHeight);
        const float height = site.posHeight.w;
        const glm::vec3 middle = foot + glm::vec3(0.0f, 0.5f * height, 0.0f);
        const float distance = glm::length(middle - eye);
        if (height <= 0.0f || distance > range || height * focal / std::max(distance, 1e-6f) < config.minPixels ||
            near.count(footKey(foot))) {
            continue;
        }
        // In the view: the crown's sphere is not wholly outside any plane of the frustum.
        const float radius = 0.7f * height;
        bool inside = true;
        for (int axis = 0; axis < 3 && inside; ++axis) {
            for (float sign : {-1.0f, 1.0f}) {
                const glm::vec4 plane = glm::vec4(clip[0][3], clip[1][3], clip[2][3], clip[3][3]) +
                                        sign * glm::vec4(clip[0][axis], clip[1][axis], clip[2][axis], clip[3][axis]);
                const float d = (glm::dot(glm::vec3(plane), middle) + plane.w) / glm::length(glm::vec3(plane));
                if (d < -radius && !(axis == 2 && sign < 0.0f)) {
                    inside = false;
                    break;
                }
            }
        }
        if (inside) {
            far.push_back(treeInstanceFor(site, scene.types, scene.config.seed, config.variants));
        }
    }

    // The batches, lit as the base's trees without its light grid (which follows the near trees), leaves in full.
    TreeRenderSettings values = base->getSettings().values().at(0);
    values.occlusion.x = 0.0f;
    values.lightGridSize.w = 0u;
    values.leafLod.z = 0.0f;
    values.leafLod.w = 1.0f;
    settings->set(0, values);
    TreeFrame treeFrame = base->getGenerator().getFrame().values().at(0);
    generator->getFrame().set(0, treeFrame);
    TreeInstance hidden{};
    hidden.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    for (size_t first = 0; first < far.size(); first += config.batch) {
        std::vector<TreeInstance> batch(far.begin() + first, far.begin() + std::min(far.size(), first + config.batch));
        batch.resize(config.batch, hidden);
        generator->getInstances().set(batch);
        batchGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
        ++frame.batches;
    }
    frame.trees = uint32_t(far.size());
    frame.image = readTarget(vulkanContext, target->getImage(0), extent);
    frame.mask.assign(size_t(extent.width) * extent.height, 0);
    for (size_t i = 0; i < frame.mask.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            if (frame.image[4 * i + c] != baseImage[4 * i + c]) {
                frame.mask[i] = 255;
            }
        }
    }
    return frame;
}

} // namespace klartraum
