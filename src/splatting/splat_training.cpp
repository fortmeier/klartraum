// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/splatting/splat_training.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/raster/differentiable_shading.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum::splatting {

namespace {

constexpr uint32_t kTile = 16;      // TILE in splat_common.slang
constexpr uint32_t kDepthBits = 20; // DEPTH_BITS in splat_common.slang

struct ProjectPushConstants {
    uint32_t count, width, height, tilesX, tilesY, capacity;
    float depthScale, nearDepth;
};

struct CapacityPushConstants {
    uint32_t capacity;
};

struct RenderPushConstants {
    uint32_t width, height, tilesX;
};

struct ProjectBackwardPushConstants {
    uint32_t count, width, height;
};

} // namespace

SplatParameters createSplatParameters(VulkanContext& vulkanContext, uint32_t count) {
    if (count == 0)
        throw std::invalid_argument("createSplatParameters: at least one Gaussian is needed");
    const auto tensor = [&](std::vector<uint32_t> shape) { return vulkanContext.create<TensorElement<float>>(shape); };
    return {count, tensor({count, 4}), tensor({count, 4}), tensor({count, 4}), tensor({count, 4}), tensor({count})};
}

SplatForward::SplatForward(VulkanContext& vulkanContext, const SplatParameters& parameters,
                           const std::shared_ptr<TensorElement<float>>& cameras,
                           const ComputeGraphElementPtr& viewSource, int viewSlot, const SplatRasterOptions& options)
    : options(options) {
    tilesX = (options.width + kTile - 1) / kTile;
    tilesY = (options.height + kTile - 1) / kTile;
    if (tilesX * tilesY > (1u << (32 - kDepthBits)))
        throw std::invalid_argument("SplatForward: too many tiles for the sort key");
    const uint32_t pixels = options.width * options.height, tiles = tilesX * tilesY;
    projected = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{3 * parameters.count, 4});
    colors = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{pixels, 4});
    keys = std::make_shared<UintBuffer>(vulkanContext, options.capacity);
    values = std::make_shared<UintBuffer>(vulkanContext, options.capacity);
    keysScratch = std::make_shared<UintBuffer>(vulkanContext, options.capacity);
    valuesScratch = std::make_shared<UintBuffer>(vulkanContext, options.capacity);
    counters = std::make_shared<UintBuffer>(vulkanContext, 1);
    sortCount = std::make_shared<UintBuffer>(vulkanContext, 1);
    tileRanges = std::make_shared<UintBuffer>(vulkanContext, 2 * tiles);
    lastContributor = std::make_shared<UintBuffer>(vulkanContext, pixels);

    clear = raster::clearBuffers(vulkanContext);
    clear->setInput(counters, 0);
    clear->setInput(tileRanges, 1);

    auto projection =
        vulkanContext.create<GeneralComputation<ProjectPushConstants>>("shaders/splatting/splat_project.comp.spv");
    projection->setPushConstants({{parameters.count, options.width, options.height, tilesX, tilesY, options.capacity,
                                   options.depthScale, options.nearDepth}});
    projection->setGroupCountX((parameters.count + 127) / 128);
    projection->setInput(parameters.means, 0);
    projection->setInput(parameters.logScales, 1);
    projection->setInput(parameters.rotations, 2);
    projection->setInput(parameters.colors, 3);
    projection->setInput(parameters.opacities, 4);
    projection->setInput(cameras, 5);
    projection->setInput(viewSource, 6, viewSlot);
    projection->setInput(projected, 7);
    projection->setInput(keys, 8);
    projection->setInput(values, 9);
    projection->setInput(clear, 10, 0);
    project = projection;

    auto clamp =
        vulkanContext.create<GeneralComputation<CapacityPushConstants>>("shaders/splatting/splat_clamp_count.comp.spv");
    clamp->setPushConstants({{options.capacity}});
    clamp->setGroupCountX(1);
    clamp->setInput(project, 0, 10);
    clamp->setInput(sortCount, 1);
    clampCount = clamp;

    // The radix sort of the Gaussian splatting renderer: eight passes of four bits over the first sortCount keys,
    // ping-ponging between the buffers, so the sorted keys end where they started.
    const uint32_t sortGroups = std::max(1u, std::min(256u, options.capacity / 128 + 1));
    auto radixSort = std::make_shared<RadixSort>(
        vulkanContext, std::vector<std::string>{"shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
                                                "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
                                                "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"});
    radixSort->setInput(project, 0, 8);
    radixSort->setInput(project, 1, 9);
    radixSort->setInput(keysScratch, 2);
    radixSort->setInput(valuesScratch, 3);
    radixSort->addScratchBufferElement(std::make_shared<UintBuffer>(vulkanContext, 16), true);
    radixSort->addScratchBufferElement(std::make_shared<UintBuffer>(vulkanContext, sortGroups + 1), true);
    radixSort->addScratchBufferElement(sortCount, false);
    radixSort->addScratchBufferElement(std::make_shared<UintBuffer>(vulkanContext, 16 * sortGroups), true);
    radixSort->setGroupCountX(sortGroups);
    std::vector<SortPushConstants> passes;
    for (uint32_t pass = 0; pass < 8; ++pass)
        passes.push_back({pass, options.capacity, 16, 1});
    radixSort->setPushConstants(passes);
    radixSort->addDependency(clampCount);
    sort = radixSort;

    auto tileRange = vulkanContext.create<GeneralComputation<>>("shaders/splatting/splat_tile_ranges.comp.spv");
    tileRange->setGroupCountX((options.capacity + 255) / 256);
    tileRange->setInput(sort, 0, 0);
    tileRange->setInput(clampCount, 1, 1);
    tileRange->setInput(clear, 2, 1);
    ranges = tileRange;

    auto blend =
        vulkanContext.create<GeneralComputation<RenderPushConstants>>("shaders/splatting/splat_render.comp.spv");
    blend->setPushConstants({{options.width, options.height, tilesX}});
    blend->setGroupCount(tilesX, tilesY, 1);
    blend->setInput(project, 0, 7);
    blend->setInput(sort, 1, 1);
    blend->setInput(ranges, 2, 2);
    blend->setInput(colors, 3);
    blend->setInput(lastContributor, 4);
    render = blend;
}

SplatBackward::SplatBackward(VulkanContext& vulkanContext, const SplatParameters& parameters,
                             const std::shared_ptr<TensorElement<float>>& cameras,
                             const ComputeGraphElementPtr& viewSource, int viewSlot, const SplatForward& forward,
                             const ComputeGraphElementPtr& imageGradientSource, int imageGradientSlot) {
    const uint32_t count = parameters.count;
    splatGradients = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{count, 12});
    gradients = createSplatParameters(vulkanContext, count);
    statistics = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{count, 2});

    clear = raster::clearBuffers(vulkanContext);
    clear->setInput(splatGradients, 0);

    auto blendBackward = vulkanContext.create<GeneralComputation<RenderPushConstants>>(
        "shaders/splatting/splat_render_backward.comp.spv");
    blendBackward->setPushConstants({{forward.options.width, forward.options.height, forward.tilesX}});
    blendBackward->setGroupCount(forward.tilesX, forward.tilesY, 1);
    blendBackward->setInput(forward.project, 0, 7);
    blendBackward->setInput(forward.sort, 1, 1);
    blendBackward->setInput(forward.ranges, 2, 2);
    blendBackward->setInput(forward.render, 3, 3);
    blendBackward->setInput(forward.render, 4, 4);
    blendBackward->setInput(imageGradientSource, 5, imageGradientSlot);
    blendBackward->setInput(clear, 6, 0);
    renderBackward = blendBackward;

    auto projection = vulkanContext.create<GeneralComputation<ProjectBackwardPushConstants>>(
        "shaders/splatting/splat_project_backward.comp.spv");
    projection->setPushConstants({{count, forward.options.width, forward.options.height}});
    projection->setGroupCountX((count + 127) / 128);
    projection->setInput(parameters.means, 0);
    projection->setInput(parameters.logScales, 1);
    projection->setInput(parameters.rotations, 2);
    projection->setInput(parameters.opacities, 3);
    projection->setInput(cameras, 4);
    projection->setInput(viewSource, 5, viewSlot);
    projection->setInput(forward.project, 6, 7);
    projection->setInput(renderBackward, 7, 6);
    projection->setInput(gradients.means, 8);
    projection->setInput(gradients.logScales, 9);
    projection->setInput(gradients.rotations, 10);
    projection->setInput(gradients.colors, 11);
    projection->setInput(gradients.opacities, 12);
    projection->setInput(statistics, 13);
    projectBackward = projection;
}

} // namespace klartraum::splatting
