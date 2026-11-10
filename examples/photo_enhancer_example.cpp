// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The bonsai landscape's forest fly-in rendered photo-realistically in Klartraum: every frame is rasterized at 1280 x
// 720 into G-buffers (colour, classes, 16-bit depth, surface coordinates u, v and object id) and downsampled to the
// 832 x 464 samples of training, a learnable surface texture (neural::hashTexture(), a multi-resolution hash grid over
// (u, v, id)) gives every pixel its features, and the photo enhancer's generator (an ONNX graph) turns G-buffers and
// features into the frame -- the network scripts/neural/train_photo_enhancer.py --texture trains from Qwen-Image
// photographs.
//
//   photo_enhancer_example --model build/TestingOutput/gan_runs/v6/photo_enhancer --out-dir build/TestingOutput/pe
//
// Frames go to <out-dir>/frames/frame_NNNN.png (enhanced) and render_NNNN.png (the downsampled rendering).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/image_io.hpp"
#include "klartraum/neural/hash_texture.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/tree/bonsai_flight.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kRenderWidth = 1280, kRenderHeight = 720;                   // the training flights' frames
constexpr uint32_t kWidth = 832, kHeight = 464, kScaledHeight = 468, kTop = 2; // the training's sample size
constexpr uint32_t kClasses = 7, kInputChannels = 3 + 1 + kClasses;

// Reads an OffscreenTarget image back as tightly packed BGRA bytes; the renderers leave it in TRANSFER_SRC_OPTIMAL.
std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkExtent2D extent) {
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

void setCamera(CameraMVP& ubo, glm::vec3 eye, glm::vec3 target, float aspect, float nearPlane) {
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(40.0f), aspect, nearPlane, 50.0f);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
}

float value16(const std::vector<uint8_t>& bgra, size_t p) {
    return float(bgra[4 * p + 2] * 256u + bgra[4 * p + 1]) / 65535.0f;
}

// One pass of PIL's resampling (Image.resize with BICUBIC or BILINEAR): output sample i of `outSize` blends the inputs
// around (i + 0.5) * scale with the filter stretched by the scale, the weights normalized.
struct Taps {
    std::vector<uint32_t> first;
    std::vector<std::vector<float>> weights;
};

Taps resamplingTaps(uint32_t inSize, uint32_t outSize, bool bicubic) {
    const auto filter = [bicubic](float x) {
        x = std::abs(x);
        if (!bicubic)
            return x < 1.0f ? 1.0f - x : 0.0f;
        constexpr float a = -0.5f;
        if (x < 1.0f)
            return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        if (x < 2.0f)
            return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
        return 0.0f;
    };
    const float scale = float(inSize) / float(outSize), filterScale = std::max(scale, 1.0f);
    const float support = (bicubic ? 2.0f : 1.0f) * filterScale;
    Taps taps;
    for (uint32_t i = 0; i < outSize; ++i) {
        const float centre = (float(i) + 0.5f) * scale;
        const int lo = std::max(int(centre - support + 0.5f), 0);
        const int hi = std::min(int(centre + support + 0.5f), int(inSize));
        std::vector<float> w;
        float sum = 0.0f;
        for (int x = lo; x < hi; ++x) {
            w.push_back(filter((float(x) - centre + 0.5f) / filterScale));
            sum += w.back();
        }
        for (float& v : w)
            v /= sum;
        taps.first.push_back(uint32_t(lo));
        taps.weights.push_back(std::move(w));
    }
    return taps;
}

// A [kRenderHeight][kRenderWidth] plane resampled to the sample size and cropped to its kHeight rows; with `bytes`,
// rounded to whole 8-bit levels after each pass as PIL does for 8-bit images.
std::vector<float> toSampleSize(const std::vector<float>& plane, bool bicubic, bool bytes) {
    static const Taps horizontal[2] = {resamplingTaps(kRenderWidth, kWidth, false),
                                       resamplingTaps(kRenderWidth, kWidth, true)};
    static const Taps vertical[2] = {resamplingTaps(kRenderHeight, kScaledHeight, false),
                                     resamplingTaps(kRenderHeight, kScaledHeight, true)};
    const auto level = [bytes](float v) { return bytes ? std::clamp(std::round(v), 0.0f, 255.0f) : v; };
    const Taps& h = horizontal[bicubic];
    const Taps& v = vertical[bicubic];
    std::vector<float> rows(size_t(kRenderHeight) * kWidth), out(size_t(kHeight) * kWidth);
    for (uint32_t y = 0; y < kRenderHeight; ++y)
        for (uint32_t x = 0; x < kWidth; ++x) {
            float sum = 0.0f;
            for (size_t k = 0; k < h.weights[x].size(); ++k)
                sum += h.weights[x][k] * plane[size_t(y) * kRenderWidth + h.first[x] + k];
            rows[size_t(y) * kWidth + x] = level(sum);
        }
    for (uint32_t y = 0; y < kHeight; ++y)
        for (uint32_t x = 0; x < kWidth; ++x) {
            const uint32_t row = y + kTop;
            float sum = 0.0f;
            for (size_t k = 0; k < v.weights[row].size(); ++k)
                sum += v.weights[row][k] * rows[(v.first[row] + k) * kWidth + x];
            out[size_t(y) * kWidth + x] = level(sum);
        }
    return out;
}

// The render pixel PIL's NEAREST resampling takes for sample pixel p.
size_t nearest(size_t p) {
    const uint32_t x = uint32_t(p % kWidth), y = uint32_t(p / kWidth) + kTop;
    const uint32_t sx = uint32_t((float(x) + 0.5f) * float(kRenderWidth) / float(kWidth));
    const uint32_t sy = uint32_t((float(y) + 0.5f) * float(kRenderHeight) / float(kScaledHeight));
    return size_t(sy) * kRenderWidth + sx;
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path model = "build/TestingOutput/gan_runs/v6/photo_enhancer";
    std::filesystem::path outDir = "build/TestingOutput/photo_enhancer";
    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 1.0f;
    float azimuth = -2.09f, seconds = 24.0f, fps = 30.0f, season = 2.0f;
    uint32_t every = 1, first = 0, last = UINT32_MAX;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--model" && hasValue)
            model = argv[++i];
        else if (arg == "--out-dir" && hasValue)
            outDir = argv[++i];
        else if (arg == "--seed" && hasValue)
            sceneConfig.seed = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--azimuth" && hasValue)
            azimuth = std::stof(argv[++i]);
        else if (arg == "--season" && hasValue)
            season = std::stof(argv[++i]);
        else if (arg == "--fly-in" && hasValue)
            seconds = std::stof(argv[++i]);
        else if (arg == "--every" && hasValue)
            every = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--from" && hasValue)
            first = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--to" && hasValue)
            last = uint32_t(std::stoul(argv[++i]));
        else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    try {
        HeadlessFrontend frontend;
        VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
        const BonsaiScene scene = makeBonsaiScene(sceneConfig);
        const glm::vec3 start = scene.focus + 4.2f * glm::vec3(std::sin(azimuth) * std::cos(0.32f), std::sin(0.32f),
                                                               std::cos(azimuth) * std::cos(0.32f));
        ForestApproachConfig flightConfig;
        flightConfig.seconds = seconds;
        flightConfig.season = season;
        ForestFlight flight(scene, start, scene.focus, findForestApproach(scene, flightConfig), flightConfig);
        if (!flight.isValid())
            throw std::runtime_error("no place found to fly into a forest");

        const VkExtent2D extent{kRenderWidth, kRenderHeight};
        auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        BonsaiRenderer bonsai(vc, scene, target, config);
        ComputeGraph renderGraph(vc, 1);
        renderGraph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        bonsai.setSeason(season);

        // The network: texture features into the generator's input after its G-buffer channels.
        neural::HashTextureShape shape = neural::HashTextureShape::fromFile(model.string() + "_texture.txt");
        shape.outputOffset = kInputChannels;
        const uint32_t channels = kInputChannels + shape.levels * shape.features;
        const size_t pixels = size_t(kWidth) * kHeight;
        auto coordinates = vc.create<TensorElement<float>>(std::vector<uint32_t>{3, kHeight, kWidth});
        auto classTensor = vc.create<TensorElement<float>>(std::vector<uint32_t>{kHeight, kWidth});
        auto table = vc.create<TensorElement<float>>(std::vector<uint32_t>{
            uint32_t(std::filesystem::file_size(model.string() + "_texture_f32.bin") / sizeof(float))});
        auto input = vc.create<TensorElement<float>>(std::vector<uint32_t>{1, channels, kHeight, kWidth});
        auto texture = neural::hashTexture(vc, shape);
        texture->setInput(coordinates, 0);
        texture->setInput(classTensor, 1);
        texture->setInput(table, 2);
        texture->setInput(input, 3);
        ComputeGraph textureGraph(vc, 1);
        textureGraph.compileFrom(texture);
        auto network = vc.create<OnnxNetwork>(model.string() + ".onnx");
        network->retainTensor("image");
        network->setInputTensor("input", input, -1);
        ComputeGraph networkGraph(vc, 1);
        networkGraph.compileFrom(network);
        {
            std::vector<float> values(table->getDataElementCount());
            std::ifstream(model.string() + "_texture_f32.bin", std::ios::binary)
                .read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
            table->getDataBuffer(0).memcopyFrom(values);
        }
        auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("image"));

        std::filesystem::create_directories(outDir / "frames");
        const uint32_t frames = uint32_t(std::lround(seconds * fps));
        float depthFar = 50.0f; // the depth window, fitted per frame as tree_bonsai_example --gbuffers does
        const auto render = [&](BonsaiOutput output, float near = 0.0f, float far = 1.0f) {
            bonsai.setOutput(output, near, far);
            renderGraph.submitAndWait(vc.getGraphicsQueue(), 0);
            return readImage(vc, target->getImage(0), extent);
        };
        for (uint32_t frame = 0; frame < std::min(frames, last); ++frame) {
            const FlyInPose pose = flight.at(float(frame) / float(std::max(frames - 1, 1u)));
            setCamera(bonsai.getCamera()->ubo, pose.eye, pose.lookAt, float(kRenderWidth) / float(kRenderHeight),
                      pose.nearPlane);
            bonsai.getCamera()->update(0);
            bonsai.setView(pose.eye);
            bonsai.setTime(float(frame) / fps, 0.5f, season,
                           glm::vec4(glm::normalize(glm::vec3(1.0f, 0.0f, 0.35f)), 0.5f));
            // Tiles of the placement come a few per frame; a second frame lets them keep up with the flight.
            renderGraph.submitAndWait(vc.getGraphicsQueue(), 0);
            bonsai.setView(pose.eye);
            if (frame < first || (frame - first) % every != 0) {
                renderGraph.submitAndWait(vc.getGraphicsQueue(), 0);
                continue;
            }
            const auto started = std::chrono::steady_clock::now();
            const auto colour = render(BonsaiOutput::Colour);
            const auto classImage = render(BonsaiOutput::Classes);
            const size_t renderPixels = size_t(kRenderWidth) * kRenderHeight;
            // Classes: the nearest class colour; the background (black) is the only uncovered class.
            std::vector<uint8_t> classes(renderPixels);
            for (size_t p = 0; p < renderPixels; ++p) {
                const glm::vec3 c = glm::vec3(classImage[4 * p + 2], classImage[4 * p + 1], classImage[4 * p]) / 255.0f;
                uint32_t best = 0;
                for (uint32_t i = 1; i < kClasses; ++i)
                    if (glm::length(tree::treeClassColor(i) - c) < glm::length(tree::treeClassColor(best) - c))
                        best = i;
                classes[p] = uint8_t(best);
            }
            // The depth window as tree_bonsai_example --gbuffers fits it: the 98th percentile of the covered
            // pixels' distances near 80 % of the grey range, starting from the previous frame's window.
            for (int attempt = 0; attempt < 12; ++attempt) {
                const auto depth = render(BonsaiOutput::Depth, 0.0f, depthFar);
                std::vector<uint8_t> covered;
                for (size_t p = 0; p < renderPixels; ++p)
                    if (classes[p] != 0)
                        covered.push_back(depth[4 * p]);
                if (covered.empty())
                    break;
                const size_t rank = covered.size() * 98 / 100;
                std::nth_element(covered.begin(), covered.begin() + rank, covered.end());
                const float percentile = float(covered[rank]);
                float next = percentile >= 255.0f  ? depthFar * 2.0f
                             : percentile < 150.0f ? depthFar * std::max(percentile, 4.0f) / 205.0f
                                                   : depthFar;
                next = std::min(next, 50.0f);
                if (next == depthFar)
                    break;
                depthFar = next;
            }
            const auto depth16 = render(BonsaiOutput::Depth16, 0.0f, depthFar);
            const auto us = render(BonsaiOutput::CoordinatesU);
            const auto vs = render(BonsaiOutput::CoordinatesV);

            // The inputs as scripts/neural/prepare_photo_pairs.py and add_surface_coordinates.py make them from the
            // flight's frames: RGB resampled bicubically, the inverse depth stretched between its 2nd and 98th
            // percentile over the covered pixels and resampled bilinearly, classes and surface coordinates sampled
            // nearest (they must not blend between surfaces); classes one-hot.
            std::vector<float> plane(renderPixels), inverse(renderPixels, 0.0f), coveredInverse;
            std::vector<float> values(size_t(kInputChannels) * pixels, 0.0f), coords(3 * pixels), classValues(pixels);
            for (int c = 0; c < 3; ++c) {
                for (size_t p = 0; p < renderPixels; ++p)
                    plane[p] = colour[4 * p + 2 - c];
                const auto scaled = toSampleSize(plane, true, true);
                for (size_t p = 0; p < pixels; ++p)
                    values[c * pixels + p] = scaled[p] / 255.0f;
            }
            for (size_t p = 0; p < renderPixels; ++p)
                if (classes[p] != 0) {
                    inverse[p] = 1.0f / std::max(value16(depth16, p) * depthFar, 1e-9f);
                    coveredInverse.push_back(inverse[p]);
                }
            std::fill(plane.begin(), plane.end(), 0.0f);
            if (!coveredInverse.empty()) {
                const auto percentile = [&](double q) {
                    const double position = q * double(coveredInverse.size() - 1);
                    const size_t below = size_t(position);
                    std::nth_element(coveredInverse.begin(), coveredInverse.begin() + below, coveredInverse.end());
                    const float low = coveredInverse[below];
                    if (below + 1 >= coveredInverse.size())
                        return low;
                    const float high = *std::min_element(coveredInverse.begin() + below + 1, coveredInverse.end());
                    return low + float(position - double(below)) * (high - low);
                };
                const float a = percentile(0.02), b = percentile(0.98);
                for (size_t p = 0; p < renderPixels; ++p)
                    if (classes[p] != 0)
                        plane[p] = std::clamp((inverse[p] - a) / std::max(b - a, 1e-12f), 0.0f, 1.0f);
            }
            const auto depth = toSampleSize(plane, false, false);
            std::copy(depth.begin(), depth.end(), values.begin() + 3 * pixels);
            for (size_t p = 0; p < pixels; ++p) {
                const size_t q = nearest(p);
                values[(4 + classes[q]) * pixels + p] = 1.0f;
                classValues[p] = float(classes[q]);
                coords[p] = value16(us, q);
                coords[pixels + p] = value16(vs, q);
                coords[2 * pixels + p] = float(us[4 * q] * 256u + vs[4 * q]) / 65535.0f;
            }
            input->getDataBuffer(0).memcopyFrom(values.data(), values.size());
            coordinates->getDataBuffer(0).memcopyFrom(coords);
            classTensor->getDataBuffer(0).memcopyFrom(classValues);
            textureGraph.submitAndWait(vc.getGraphicsQueue(), 0);
            networkGraph.submitAndWait(vc.getGraphicsQueue(), 0);
            std::vector<float> image(output->getDataElementCount());
            output->getDataBuffer(0).memcopyTo(image);

            std::vector<uint8_t> rgb(3 * pixels), rendering(3 * pixels);
            for (size_t p = 0; p < pixels; ++p)
                for (int c = 0; c < 3; ++c) {
                    rgb[3 * p + c] = uint8_t(std::lround(std::clamp(image[c * pixels + p], 0.0f, 1.0f) * 255.0f));
                    rendering[3 * p + c] = uint8_t(std::lround(values[c * pixels + p] * 255.0f));
                }
            char name[64];
            std::snprintf(name, sizeof(name), "frame_%04u.png", frame);
            writeImage(outDir / "frames" / name, rgb.data(), kWidth, kHeight, 3);
            std::snprintf(name, sizeof(name), "render_%04u.png", frame);
            writeImage(outDir / "frames" / name, rendering.data(), kWidth, kHeight, 3);
            std::cout << "frame " << frame << " in "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s"
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "photo_enhancer_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
