// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/neural/photo_alignment.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>

#include "klartraum/onnx/onnx_network.hpp"

namespace klartraum::neural {

namespace {

// Bilinear sample of a planar [height][width] plane with half-pixel centres, at pixel (x, y) of a toWidth x toHeight
// image.
float sample(const float* plane, uint32_t width, uint32_t height, uint32_t x, uint32_t y, uint32_t toWidth,
             uint32_t toHeight) {
    const float sx = std::clamp((x + 0.5f) * width / toWidth - 0.5f, 0.0f, float(width - 1));
    const float sy = std::clamp((y + 0.5f) * height / toHeight - 0.5f, 0.0f, float(height - 1));
    const uint32_t x0 = uint32_t(sx), y0 = uint32_t(sy);
    const uint32_t x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const float fx = sx - float(x0), fy = sy - float(y0);
    const auto at = [&](uint32_t px, uint32_t py) { return plane[size_t(py) * width + px]; };
    return (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
}

std::vector<double> ranks(const std::vector<float>& values) {
    std::vector<size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return values[a] < values[b]; });
    std::vector<double> result(values.size());
    for (size_t k = 0; k < order.size(); ++k)
        result[order[k]] = double(k);
    return result;
}

} // namespace

float spearman(const std::vector<float>& a, const std::vector<float>& b) {
    const auto ra = ranks(a), rb = ranks(b);
    const double n = double(ra.size());
    const double ma = std::accumulate(ra.begin(), ra.end(), 0.0) / n,
                 mb = std::accumulate(rb.begin(), rb.end(), 0.0) / n;
    double covariance = 0.0, va = 0.0, vb = 0.0;
    for (size_t i = 0; i < ra.size(); ++i) {
        covariance += (ra[i] - ma) * (rb[i] - mb);
        va += (ra[i] - ma) * (ra[i] - ma);
        vb += (rb[i] - mb) * (rb[i] - mb);
    }
    return float(covariance / std::sqrt(va * vb));
}

PhotoAlignment::PhotoAlignment(VulkanContext& vulkanContext, const std::string& model, uint32_t networkHeight,
                               uint32_t networkWidth)
    : vulkanContext(vulkanContext),
      networkHeight(networkHeight),
      networkWidth(networkWidth) {
    if (!std::filesystem::exists(model))
        throw std::runtime_error(model + " is missing: run scripts/landscape/export_depth.py");
    network = vulkanContext.create<OnnxNetwork>(model);
    network->retainTensor("depth");
    input = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, 3, networkHeight, networkWidth});
    network->setInputTensor("image", input, -1);
    graph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    graph->compileFrom(network);
    output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("depth"));
}

PhotoAlignment::~PhotoAlignment() = default;

float PhotoAlignment::score(const std::vector<float>& photo, const std::vector<float>& renderedDepth, uint32_t width,
                            uint32_t height) {
    const size_t pixels = size_t(width) * height;
    // The photograph at the network's size (squeezed as the Python check does).
    std::vector<float> resized(size_t(3) * networkWidth * networkHeight);
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t y = 0; y < networkHeight; ++y)
            for (uint32_t x = 0; x < networkWidth; ++x)
                resized[(size_t(c) * networkHeight + y) * networkWidth + x] =
                    sample(photo.data() + c * pixels, width, height, x, y, networkWidth, networkHeight);
    input->getDataBuffer(0).memcopyFrom(resized);
    graph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
    std::vector<float> metres(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(metres);
    // Every 7th covered pixel: the rendered inverse depth and the estimated one (from at least 0.1 m).
    std::vector<float> rendered, estimated;
    size_t covered = 0;
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            const size_t p = size_t(y) * width + x;
            if (renderedDepth[p] * 255.0f < 0.5f)
                continue;
            if (covered++ % 7 != 0)
                continue;
            rendered.push_back(renderedDepth[p]);
            estimated.push_back(
                1.0f / std::max(sample(metres.data(), networkWidth, networkHeight, x, y, width, height), 0.1f));
        }
    if (covered < 1000)
        return -1.0f;
    return spearman(rendered, estimated);
}

} // namespace klartraum::neural
