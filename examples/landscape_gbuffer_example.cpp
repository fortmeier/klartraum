// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Estimates G-buffers of photographs or generated images with two networks run in Klartraum: the metric depth of
// Depth Anything V2 Metric Outdoor Small and the ADE20K classes of UperNet ConvNeXt-Base (ONNX graphs exported by
// scripts/landscape/export_depth.py and export_semseg.py). Every input image is resized to each network's size and
// its outputs are scaled back to the image's size. Per image it writes, into --out-dir:
//
//   <name>_depth.f32  the depth in metres, [height][width] float32
//   <name>_depth.pgm  the depth in centimetres, 16 bit
//   <name>_class.pgm  the ADE20K class index (0..149) of every pixel
//   <name>_class.ppm  the classes in colour, for looking at
//
//   landscape_gbuffer_example --out-dir build/TestingOutput/landscape_gbuffers image1.ppm image2.ppm ...
//
// Inputs are binary PPM (P6) of any size.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"

using namespace klartraum;

namespace {

struct Image {
    uint32_t width = 0, height = 0;
    std::vector<float> rgb; // [height][width][3] in [0, 1]
};

Image readPpm(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    Image image;
    uint32_t maximum = 0;
    file >> magic >> image.width >> image.height >> maximum;
    file.get();
    if (magic != "P6" || maximum != 255 || image.width == 0 || image.height == 0)
        throw std::runtime_error(path.string() + " is not a binary 8-bit PPM");
    std::vector<uint8_t> bytes(size_t(image.width) * image.height * 3);
    file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    image.rgb.resize(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i)
        image.rgb[i] = bytes[i] / 255.0f;
    return image;
}

// Bilinear sample of a [height][width][channels] (interleaved) or planar [channels][height][width] array with
// half-pixel centres, at the position of pixel (x, y) of an image of `toWidth` x `toHeight`.
float sample(const std::vector<float>& values, uint32_t width, uint32_t height, uint32_t channels, uint32_t channel,
             bool planar, uint32_t x, uint32_t y, uint32_t toWidth, uint32_t toHeight) {
    const float sx = std::clamp((x + 0.5f) * width / toWidth - 0.5f, 0.0f, float(width - 1));
    const float sy = std::clamp((y + 0.5f) * height / toHeight - 0.5f, 0.0f, float(height - 1));
    const uint32_t x0 = uint32_t(sx), y0 = uint32_t(sy), x1 = std::min(x0 + 1, width - 1),
                   y1 = std::min(y0 + 1, height - 1);
    const float fx = sx - x0, fy = sy - y0;
    auto at = [&](uint32_t px, uint32_t py) {
        return planar ? values[(size_t(channel) * height + py) * width + px]
                      : values[(size_t(py) * width + px) * channels + channel];
    };
    const float top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
    const float bottom = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
    return top * (1 - fy) + bottom * fy;
}

// The image resized to a network's input [1][3][height][width].
std::vector<float> networkInput(const Image& image, uint32_t width, uint32_t height) {
    std::vector<float> planar(size_t(3) * width * height);
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                planar[(size_t(c) * height + y) * width + x] =
                    sample(image.rgb, image.width, image.height, 3, c, false, x, y, width, height);
    return planar;
}

// A network loaded once and run on one input after the other.
class Network {
public:
    Network(VulkanContext& vc, const std::string& path, const std::string& inputName, const std::string& outputName,
            std::vector<uint32_t> inputShape)
        : vc(vc),
          shape(inputShape) {
        if (!std::filesystem::exists(path))
            throw std::runtime_error(path + " is missing: run the export scripts in scripts/landscape/");
        network = vc.create<OnnxNetwork>(path);
        network->retainTensor(outputName);
        input = vc.create<TensorElement<float>>(inputShape);
        network->setInputTensor(inputName, input, -1);
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(network);
        output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement(outputName));
    }

    uint32_t height() const { return shape[2]; }
    uint32_t width() const { return shape[3]; }

    std::vector<float> run(const std::vector<float>& values) {
        input->getDataBuffer(0).memcopyFrom(values);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        std::vector<float> result(output->getDataElementCount());
        output->getDataBuffer(0).memcopyTo(result);
        return result;
    }

private:
    VulkanContext& vc;
    std::vector<uint32_t> shape;
    std::shared_ptr<OnnxNetwork> network;
    std::shared_ptr<TensorElement<float>> input, output;
    std::unique_ptr<ComputeGraph> graph;
};

// A distinct colour per class (golden-angle hues), for the class previews.
void classColour(uint32_t c, uint8_t* rgb) {
    const float hue = std::fmod(c * 137.508f, 360.0f) / 60.0f, value = c % 2 ? 0.95f : 0.7f;
    const float f = hue - std::floor(hue);
    const float p = 0.25f * value, q = value * (1 - 0.75f * f), t = value * (0.25f + 0.75f * f);
    float r = value, g = t, b = p;
    switch (int(hue) % 6) {
    case 1:
        r = q, g = value, b = p;
        break;
    case 2:
        r = p, g = value, b = t;
        break;
    case 3:
        r = p, g = q, b = value;
        break;
    case 4:
        r = t, g = p, b = value;
        break;
    case 5:
        r = value, g = p, b = q;
        break;
    default:
        break;
    }
    rgb[0] = uint8_t(r * 255), rgb[1] = uint8_t(g * 255), rgb[2] = uint8_t(b * 255);
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path outDir = "build/TestingOutput/landscape_gbuffers";
    std::string depthModel = "build/models/depth/onnx/depth_anything_v2_metric_outdoor_small_518x924.onnx";
    std::string segmentationModel = "build/models/semseg/onnx/upernet_convnext_base_ade20k_512x896.onnx";
    std::vector<std::filesystem::path> images;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out-dir" && i + 1 < argc)
            outDir = argv[++i];
        else if (arg == "--depth-model" && i + 1 < argc)
            depthModel = argv[++i];
        else if (arg == "--segmentation-model" && i + 1 < argc)
            segmentationModel = argv[++i];
        else
            images.push_back(arg);
    }
    if (images.empty()) {
        std::cerr << "usage: landscape_gbuffer_example [--out-dir DIR] image.ppm..." << std::endl;
        return 1;
    }
    try {
        std::filesystem::create_directories(outDir);
        HeadlessFrontend frontend;
        auto& vc = frontend.getKlartraumEngine().getVulkanContext();
        Network depth(vc, depthModel, "image", "depth", {1, 3, 518, 924});
        Network segmentation(vc, segmentationModel, "image", "logits", {1, 3, 512, 896});
        constexpr uint32_t kClasses = 150;
        for (const auto& path : images) {
            const auto start = std::chrono::steady_clock::now();
            const Image image = readPpm(path);
            const uint32_t w = image.width, h = image.height;
            const std::string name = path.stem().string();

            const auto metres = depth.run(networkInput(image, depth.width(), depth.height()));
            std::vector<float> depthOut(size_t(w) * h);
            std::vector<uint16_t> centimetres(depthOut.size());
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x) {
                    const float d = sample(metres, depth.width(), depth.height(), 1, 0, true, x, y, w, h);
                    depthOut[size_t(y) * w + x] = d;
                    // 16-bit PGM is big-endian.
                    const auto cm = uint16_t(std::clamp(std::lround(d * 100.0f), 0l, 65535l));
                    centimetres[size_t(y) * w + x] = uint16_t((cm >> 8) | (cm << 8));
                }
            std::ofstream(outDir / (name + "_depth.f32"), std::ios::binary)
                .write(reinterpret_cast<const char*>(depthOut.data()), std::streamsize(depthOut.size() * 4));
            std::ofstream pgm(outDir / (name + "_depth.pgm"), std::ios::binary);
            pgm << "P5\n" << w << " " << h << "\n65535\n";
            pgm.write(reinterpret_cast<const char*>(centimetres.data()), std::streamsize(centimetres.size() * 2));

            // Classes: the argmax of the logits upsampled bilinearly (as the reference pipeline) to the image.
            const auto logits = segmentation.run(networkInput(image, segmentation.width(), segmentation.height()));
            const uint32_t lw = segmentation.width() / 4, lh = segmentation.height() / 4;
            std::vector<uint8_t> classes(size_t(w) * h), colours(size_t(w) * h * 3);
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x) {
                    uint32_t best = 0;
                    float bestValue = -1e30f;
                    for (uint32_t c = 0; c < kClasses; ++c) {
                        const float v = sample(logits, lw, lh, kClasses, c, true, x, y, w, h);
                        if (v > bestValue)
                            bestValue = v, best = c;
                    }
                    classes[size_t(y) * w + x] = uint8_t(best);
                    classColour(best, &colours[(size_t(y) * w + x) * 3]);
                }
            std::ofstream classPgm(outDir / (name + "_class.pgm"), std::ios::binary);
            classPgm << "P5\n" << w << " " << h << "\n255\n";
            classPgm.write(reinterpret_cast<const char*>(classes.data()), std::streamsize(classes.size()));
            std::ofstream classPpm(outDir / (name + "_class.ppm"), std::ios::binary);
            classPpm << "P6\n" << w << " " << h << "\n255\n";
            classPpm.write(reinterpret_cast<const char*>(colours.data()), std::streamsize(colours.size()));
            std::cout << name << ": depth " << *std::min_element(depthOut.begin(), depthOut.end()) << " .. "
                      << *std::max_element(depthOut.begin(), depthOut.end()) << " m in "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s"
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "landscape_gbuffer_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
