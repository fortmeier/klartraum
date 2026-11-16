// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/splatting/splat_training.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

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

std::vector<float> read(const std::shared_ptr<TensorElement<float>>& tensor) {
    std::vector<float> values(tensor->getDataElementCount());
    tensor->getDataBuffer(0).memcopyTo(values);
    return values;
}

void upload(const SplatParameters& parameters, const SplatSet& splats) {
    parameters.means->getDataBuffer(0).memcopyFrom(splats.means);
    parameters.logScales->getDataBuffer(0).memcopyFrom(splats.logScales);
    parameters.rotations->getDataBuffer(0).memcopyFrom(splats.rotations);
    parameters.colors->getDataBuffer(0).memcopyFrom(splats.colors);
    parameters.opacities->getDataBuffer(0).memcopyFrom(splats.opacities);
}

std::vector<float> packCameras(const std::vector<glm::mat4>& cameras) {
    std::vector<float> values;
    for (const auto& m : cameras)
        values.insert(values.end(), glm::value_ptr(m), glm::value_ptr(m) + 16);
    return values;
}

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

SplatSet splatsAt(const std::vector<glm::vec3>& points, float scale, glm::vec3 color, float opacity) {
    SplatSet splats;
    const float logScale = std::log(scale), logit = std::log(opacity / (1.0f - opacity));
    for (const auto& p : points) {
        splats.means.insert(splats.means.end(), {p.x, p.y, p.z, 0.0f});
        splats.logScales.insert(splats.logScales.end(), {logScale, logScale, logScale, 0.0f});
        splats.rotations.insert(splats.rotations.end(), {0.0f, 0.0f, 0.0f, 1.0f});
        splats.colors.insert(splats.colors.end(), {color.r, color.g, color.b, 0.0f});
        splats.opacities.push_back(logit);
    }
    return splats;
}

std::vector<float> renderSplats(VulkanContext& vulkanContext, const SplatSet& splats,
                                const std::vector<glm::mat4>& cameras, const SplatRasterOptions& options) {
    const SplatParameters parameters = createSplatParameters(vulkanContext, splats.size());
    auto cameraBuffer = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(cameras.size()), 16});
    auto viewIndex = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 1);
    SplatForward forward(vulkanContext, parameters, cameraBuffer, viewIndex, -1, options);
    ComputeGraph graph(vulkanContext, 1);
    graph.compileFrom(forward.render);
    upload(parameters, splats);
    cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));
    std::vector<float> images;
    for (uint32_t view = 0; view < uint32_t(cameras.size()); ++view) {
        viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
        graph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);
        const std::vector<float> image = read(forward.colors);
        images.insert(images.end(), image.begin(), image.end());
    }
    return images;
}

SplatTraining::SplatTraining(VulkanContext& vulkanContext, const SplatSet& start, const std::vector<glm::mat4>& cameras,
                             const std::vector<float>& targets, const SplatTrainingOptions& options)
    : vulkanContext(vulkanContext),
      options(options),
      cameras(cameras),
      targetValues(targets),
      seed(options.seed) {
    const size_t pixels = size_t(options.raster.width) * options.raster.height;
    if (cameras.empty() || targets.size() != 4 * pixels * cameras.size())
        throw std::invalid_argument("SplatTraining: four target floats per pixel and camera are needed");
    // The learning rates decay over the planned steps: the positions to a hundredth, the rest to a tenth.
    for (auto* adam : {&this->options.means, &this->options.logScales, &this->options.rotations, &this->options.colors,
                       &this->options.opacities}) {
        adam->decaySteps = options.steps;
        adam->finalFactor = adam == &this->options.means ? 0.01f : 0.1f;
    }
    std::vector<std::vector<float>> zeroMoments;
    for (const auto* values : {&start.means, &start.logScales, &start.rotations, &start.colors, &start.opacities})
        for (int k = 0; k < 2; ++k)
            zeroMoments.emplace_back(values->size(), 0.0f);
    build(start, zeroMoments, std::vector<float>(2 * size_t(start.size()), 0.0f));
}

void SplatTraining::build(const SplatSet& splats, const std::vector<std::vector<float>>& momentValues,
                          const std::vector<float>& statisticValues) {
    // The view the next step advances from: the last one drawn, or the last camera before the first step.
    uint32_t view = uint32_t(cameras.size()) - 1;
    if (viewIndex) {
        std::vector<uint32_t> value(1);
        viewIndex->getBuffer(0).memcopyTo(value);
        view = value[0];
    }
    graph.reset();
    const uint32_t count = splats.size(), views = uint32_t(cameras.size());
    const uint32_t pixels = options.raster.width * options.raster.height;
    parameters = createSplatParameters(vulkanContext, count);
    cameraBuffer = vulkanContext.create<Tensor>(std::vector<uint32_t>{views, 16});
    targets = vulkanContext.create<Tensor>(std::vector<uint32_t>{views * pixels, 4});
    imageGradients = vulkanContext.create<Tensor>(std::vector<uint32_t>{pixels, 4});
    lossValue = vulkanContext.create<Tensor>(std::vector<uint32_t>{1});
    viewIndex = std::make_shared<UintBuffer>(vulkanContext, 1);
    stepBuffer = std::make_shared<UintBuffer>(vulkanContext, 1);

    auto nextView = raster::cycleView(vulkanContext, views);
    nextView->setInput(viewIndex, 0);
    forward = std::make_unique<SplatForward>(vulkanContext, parameters, cameraBuffer, nextView, 0, options.raster);
    auto clearLoss = raster::clearBuffers(vulkanContext);
    clearLoss->setInput(lossValue, 0);
    auto loss = raster::imageLoss(vulkanContext, options.raster.width, options.raster.height, options.maskWeight);
    loss->setInput(forward->render, 0, 3);
    loss->setInput(targets, 1);
    loss->setInput(nextView, 2, 0);
    loss->setInput(imageGradients, 3);
    loss->setInput(clearLoss, 4, 0);
    backward = std::make_unique<SplatBackward>(vulkanContext, parameters, cameraBuffer, nextView, 0, *forward, loss, 3);

    // Adam per parameter group, after the step counter and one after another (all reads are upstream).
    auto counter = training::stepCounter(vulkanContext);
    counter->setInput(stepBuffer, 0);
    ComputeGraphElementPtr last = counter;
    const std::shared_ptr<Tensor> groups[] = {parameters.means, parameters.logScales, parameters.rotations,
                                              parameters.colors, parameters.opacities};
    const training::AdamParameters* adams[] = {&options.means, &options.logScales, &options.rotations, &options.colors,
                                               &options.opacities};
    moments.clear();
    for (int group = 0; group < 5; ++group) {
        const uint32_t elements = uint32_t(groups[group]->getDataElementCount());
        for (int k = 0; k < 2; ++k)
            moments.push_back(vulkanContext.create<Tensor>(std::vector<uint32_t>{elements}));
        auto adam = training::adamStep(vulkanContext, elements, *adams[group]);
        adam->setInput(groups[group], 0);
        adam->setInput(backward->projectBackward, 1, 8 + group);
        adam->setInput(moments[2 * group], 2);
        adam->setInput(moments[2 * group + 1], 3);
        adam->setInput(stepBuffer, 4);
        adam->addDependency(last);
        last = adam;
    }
    graph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    graph->compileFrom(last);

    upload(parameters, splats);
    for (size_t k = 0; k < moments.size(); ++k)
        moments[k]->getDataBuffer(0).memcopyFrom(momentValues[k]);
    backward->statistics->getDataBuffer(0).memcopyFrom(statisticValues);
    cameraBuffer->getDataBuffer(0).memcopyFrom(packCameras(cameras));
    targets->getDataBuffer(0).memcopyFrom(targetValues);
    viewIndex->getBuffer(0).memcopyFrom(std::vector<uint32_t>{view});
    stepBuffer->getBuffer(0).memcopyFrom(std::vector<uint32_t>{steps});
}

void SplatTraining::step() {
    if (steps >= options.densifyFrom && steps < options.densifyUntil && steps > 0 &&
        steps % options.densifyInterval == 0)
        densify();
    graph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
    ++steps;
}

float SplatTraining::loss() const { return read(lossValue)[0]; }

uint32_t SplatTraining::instanceCount() const {
    std::vector<uint32_t> value(1);
    forward->counters->getBuffer(0).memcopyTo(value);
    return value[0];
}

SplatSet SplatTraining::splats() const {
    return {read(parameters.means), read(parameters.logScales), read(parameters.rotations), read(parameters.colors),
            read(parameters.opacities)};
}

void SplatTraining::densify() {
    const SplatSet current = splats();
    const std::vector<float> statistics = read(backward->statistics);
    std::vector<std::vector<float>> momentValues;
    for (const auto& moment : moments)
        momentValues.push_back(read(moment));
    const uint32_t count = current.size();
    const float pixels = float(options.raster.width) * float(options.raster.height);
    // Group sizes per Gaussian: means, scales, rotations, colours (four floats), opacities (one).
    const int widths[] = {4, 4, 4, 4, 1};
    std::vector<float> SplatSet::* const members[] = {&SplatSet::means, &SplatSet::logScales, &SplatSet::rotations,
                                                      &SplatSet::colors, &SplatSet::opacities};
    // Gaussian `i` of `source` alone.
    const auto single = [&](const SplatSet& source, uint32_t i) {
        SplatSet one;
        for (int group = 0; group < 5; ++group) {
            const auto& values = source.*members[group];
            (one.*members[group]).assign(values.begin() + widths[group] * i, values.begin() + widths[group] * (i + 1));
        }
        return one;
    };
    SplatSet result;
    std::vector<std::vector<float>> resultMoments(10);
    // Appends Gaussian `i` of `source` (moments copied when it is an original, zero for new ones).
    const auto append = [&](const SplatSet& source, uint32_t i, bool original) {
        for (int group = 0; group < 5; ++group) {
            const auto& values = source.*members[group];
            auto& target = result.*members[group];
            target.insert(target.end(), values.begin() + widths[group] * i, values.begin() + widths[group] * (i + 1));
            for (int k = 0; k < 2; ++k) {
                auto& moment = resultMoments[2 * group + k];
                if (original)
                    moment.insert(moment.end(), momentValues[2 * group + k].begin() + widths[group] * i,
                                  momentValues[2 * group + k].begin() + widths[group] * (i + 1));
                else
                    moment.insert(moment.end(), widths[group], 0.0f);
            }
        }
    };
    const auto normal = [&]() { // Box-Muller from a linear congruential generator
        const auto uniform = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return (float(seed >> 8) + 0.5f) / float(1u << 24);
        };
        return std::sqrt(-2.0f * std::log(uniform())) * std::cos(6.2831853f * uniform());
    };
    uint32_t added = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const float seen = statistics[2 * i + 1];
        const float gradient = seen > 0.0f ? statistics[2 * i] / seen * pixels : 0.0f;
        const glm::vec3 logScale(current.logScales[4 * i], current.logScales[4 * i + 1], current.logScales[4 * i + 2]);
        const float largest = std::exp(std::max(logScale.x, std::max(logScale.y, logScale.z)));
        const bool grow = gradient > options.densifyGradient && count + added < options.maxCount;
        if (grow && largest <= options.cloneScale) { // clone: a copy that the gradient moves apart
            append(single(current, i), 0, false);
            ++added;
        } else if (grow) { // split: two samples of the Gaussian, 1.6 times smaller; the original goes
            const glm::quat q = glm::normalize(glm::quat(current.rotations[4 * i + 3], current.rotations[4 * i],
                                                         current.rotations[4 * i + 1], current.rotations[4 * i + 2]));
            const glm::vec3 mean(current.means[4 * i], current.means[4 * i + 1], current.means[4 * i + 2]);
            for (int k = 0; k < 2; ++k) {
                const glm::vec3 offset = q * (glm::exp(logScale) * glm::vec3(normal(), normal(), normal()));
                SplatSet one = single(current, i);
                for (int c = 0; c < 3; ++c) {
                    one.means[c] += offset[c];
                    one.logScales[c] -= std::log(1.6f);
                }
                append(one, 0, false);
            }
            ++added;
            continue;
        }
        append(current, i, true);
    }
    // Prune faint and oversized Gaussians.
    SplatSet kept;
    std::vector<std::vector<float>> keptMoments(10);
    for (uint32_t i = 0; i < result.size(); ++i) {
        const float opacity = 1.0f / (1.0f + std::exp(-result.opacities[i]));
        const float largest = std::exp(
            std::max(result.logScales[4 * i], std::max(result.logScales[4 * i + 1], result.logScales[4 * i + 2])));
        if (opacity < options.pruneOpacity || largest > options.pruneScale)
            continue;
        for (int group = 0; group < 5; ++group) {
            const auto& values = result.*members[group];
            auto& target = kept.*members[group];
            target.insert(target.end(), values.begin() + widths[group] * i, values.begin() + widths[group] * (i + 1));
            for (int k = 0; k < 2; ++k)
                keptMoments[2 * group + k].insert(keptMoments[2 * group + k].end(),
                                                  resultMoments[2 * group + k].begin() + widths[group] * i,
                                                  resultMoments[2 * group + k].begin() + widths[group] * (i + 1));
        }
    }
    if (kept.size() == 0)
        throw std::runtime_error("SplatTraining: densification removed every Gaussian");
    build(kept, keptMoments, std::vector<float>(2 * size_t(kept.size()), 0.0f));
}

} // namespace klartraum::splatting
