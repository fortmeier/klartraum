// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The geometric flow between two nearby views of the crate scene, and how consistent a trained latent crate model
// (crate_training_example --latent) renders them: view a and view b a few degrees further along the orbit, both
// rendered by the model with the same latent; the flow of b onto a from the G-buffers (crate and floor, so that every
// pixel has geometry), a warped onto b, their difference and the pixels a sees (the mask).
//
// Writes <out>/flow_pair.png (a | b | flow | a warped onto b | 4 x |warped a - b| | mask) and prints the masked mean
// squared differences with and without the warp, and as a check of the flow itself, how many of b's valid pixels
// find the same surface point in a (face id and surface coordinates of a's G-buffer at the flow's target).
//
//   crate_flow_example --parameters build/TestingOutput/crate_views_i2i/runs/v4/parameters_2000 \
//       --out build/TestingOutput/crate_views_i2i/flow

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "crate_scene.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gbuffer_mesh_renderer.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/image_io.hpp"
#include "klartraum/neural/photo_enhancer.hpp"
#include "klartraum/neural/photo_enhancer_training.hpp"
#include "klartraum/offscreen_target.hpp"

using namespace klartraum;

namespace {

// A renderer of the scene into its own offscreen target: the floor-less crate the model was trained on, or the crate
// and the floor for the flow.
struct SceneRenderer {
    std::shared_ptr<OffscreenTarget> target;
    std::unique_ptr<GBufferMeshRenderer> renderer;
    std::unique_ptr<ComputeGraph> graph;
    uint32_t size;

    SceneRenderer(VulkanContext& vc, bool floor, uint32_t size)
        : size(size) {
        const VkExtent2D extent{size, size};
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        renderer = std::make_unique<GBufferMeshRenderer>(vc, crate::vertices(floor), target, extent);
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(renderer->getOutput());
    }

    // One G-buffer of the view the camera shows, as BGRA bytes.
    std::vector<uint8_t> render(VulkanContext& vc, GBufferOutput output) {
        renderer->setOutput(output, 0.0f, crate::kDepthFar);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return crate::readImage(vc, target->getImage(0), VkExtent2D{size, size});
    }
};

// The orbit camera at `azimuth` degrees: 22 degrees above the crate, 3 away, looking at its middle.
CameraMVP orbitCamera(float azimuth) {
    const glm::vec3 look(0.0f, 0.5f, 0.0f);
    const float a = glm::radians(azimuth), elevation = glm::radians(22.0f);
    const glm::vec3 eye = look + 3.0f * glm::vec3(std::sin(a) * std::cos(elevation), std::sin(elevation),
                                                  std::cos(a) * std::cos(elevation));
    CameraMVP camera{};
    crate::setCamera(camera, eye, look);
    return camera;
}

// A bilinear sample of channel c of a [3][size][size] image at pixel coordinates (x, y) (pixel centres at i + 0.5),
// clamped to the image.
float sample(const std::vector<float>& image, uint32_t size, uint32_t c, float x, float y) {
    const float fx = std::clamp(x - 0.5f, 0.0f, float(size - 1)), fy = std::clamp(y - 0.5f, 0.0f, float(size - 1));
    const uint32_t x0 = uint32_t(fx), y0 = uint32_t(fy), x1 = std::min(x0 + 1, size - 1),
                   y1 = std::min(y0 + 1, size - 1);
    const float tx = fx - float(x0), ty = fy - float(y0);
    const size_t plane = size_t(c) * size * size;
    const auto at = [&](uint32_t px, uint32_t py) { return image[plane + size_t(py) * size + px]; };
    return (1.0f - ty) * ((1.0f - tx) * at(x0, y0) + tx * at(x1, y0)) +
           ty * ((1.0f - tx) * at(x0, y1) + tx * at(x1, y1));
}

uint8_t byte(float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

// A colour of the flow vector (dx, dy) on the colour wheel, its brightness the length relative to `scale`.
std::array<float, 3> flowColour(float dx, float dy, float scale) {
    const float hue = (std::atan2(dy, dx) / 3.14159265f + 1.0f) * 3.0f; // 0 .. 6
    const float value = std::min(1.0f, std::sqrt(dx * dx + dy * dy) / std::max(scale, 1e-6f));
    const float f = hue - std::floor(hue);
    std::array<float, 6> r{1, 1 - f, 0, 0, f, 1}, g{f, 1, 1, 1 - f, 0, 0}, b{0, 0, f, 1, 1, 1 - f};
    const int i = int(hue) % 6;
    return {value * r[i], value * g[i], value * b[i]};
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path parameters, outDir = "build/TestingOutput/crate_views_i2i/flow";
    uint32_t sampleSize = 128;
    float azimuth = 34.0f, delta = 3.0f;
    std::vector<float> latentCode;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--parameters" && hasValue)
            parameters = argv[++i];
        else if (arg == "--out" && hasValue)
            outDir = argv[++i];
        else if (arg == "--sample-size" && hasValue)
            sampleSize = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--azimuth" && hasValue)
            azimuth = std::stof(argv[++i]);
        else if (arg == "--delta" && hasValue)
            delta = std::stof(argv[++i]);
        else if (arg == "--latent-code" && hasValue) {
            std::stringstream numbers(argv[++i]);
            for (std::string number; std::getline(numbers, number, ',');)
                latentCode.push_back(std::stof(number));
        } else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    if (parameters.empty()) {
        std::cerr << "crate_flow_example: --parameters of a latent crate model needed" << std::endl;
        return 1;
    }
    try {
        HeadlessFrontend frontend;
        VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
        const uint32_t s = sampleSize;
        const size_t pixels = size_t(s) * s;

        // The model, forward only with given latents: its texture, U-Net widths and latent size as saved.
        neural::PhotoEnhancerTrainingConfig config;
        config.forwardOnly = true;
        config.batch = 1;
        config.size = s;
        config.texture = neural::HashTextureShape::fromFile((parameters / "texture.txt").string());
        config.widths.clear();
        for (uint32_t level = 0;; ++level) {
            const auto bias = parameters / ("generator.down." + std::to_string(level) + ".0.bias.bin");
            if (!std::filesystem::exists(bias))
                break;
            config.widths.push_back(uint32_t(std::filesystem::file_size(bias) / sizeof(float)));
        }
        config.gbufferChannels = 4;
        config.latentSize =
            uint32_t(std::filesystem::file_size(parameters / "encoder.head.bias.bin") / sizeof(float) / 2);
        config.latentObjects = 1;
        neural::PhotoEnhancerTraining model(vc, config);
        model.load(parameters.string());
        if (latentCode.empty())
            latentCode.assign(config.latentSize, 0.0f);

        SceneRenderer network(vc, false, 2 * s), geometry(vc, true, s);
        struct View {
            CameraMVP camera;
            std::vector<float> image;                  ///< the model's render, [3][s][s]
            std::vector<uint8_t> depth, u, v, classes; ///< the flow renderer's G-buffers (crate and floor)
        };
        std::vector<View> views;
        for (const float angle : {azimuth, azimuth + delta}) {
            View view;
            view.camera = orbitCamera(angle);
            network.renderer->getCamera()->ubo = view.camera;
            network.renderer->getCamera()->update(0);
            const auto frame = crate::renderFrame(vc, *network.renderer, *network.graph, *network.target, 2 * s);
            const auto inputs = neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{s, s, 1.0f});
            // As crate_training_example --latent sees a view: inverse depth and normals, the crate as the one object.
            std::vector<float> channels(inputs.channels.begin() + 3 * pixels, inputs.channels.begin() + 4 * pixels);
            channels.insert(channels.end(), inputs.normals.begin(), inputs.normals.end());
            std::vector<float> objects(pixels);
            for (size_t q = 0; q < pixels; ++q)
                objects[q] = inputs.classes[q] == float(tree::TREE_CLASS_STEM) ? 1.0f : 0.0f;
            model.setBatch(channels, inputs.coordinates, inputs.classes, {}, {});
            model.setObjects(objects);
            model.setLatents(latentCode);
            model.step();
            view.image = model.image();
            geometry.renderer->getCamera()->ubo = view.camera;
            geometry.renderer->getCamera()->update(0);
            view.depth = geometry.render(vc, GBufferOutput::Depth16);
            view.u = geometry.render(vc, GBufferOutput::CoordinatesU);
            view.v = geometry.render(vc, GBufferOutput::CoordinatesV);
            view.classes = geometry.render(vc, GBufferOutput::Classes);
            views.push_back(std::move(view));
        }
        const View &a = views[0], &b = views[1];
        const auto flow = crate::flow(a.camera, a.depth, b.camera, b.depth, s);

        // a warped onto b, the masked squared differences with and without the warp.
        std::vector<float> warped(3 * pixels, 0.0f);
        double warpedError = 0.0, plainError = 0.0, longest = 0.0;
        size_t valid = 0;
        for (size_t p = 0; p < pixels; ++p) {
            if (!flow.valid[p])
                continue;
            ++valid;
            const float column = float(p % s) + 0.5f, row = float(size_t(p / s)) + 0.5f;
            longest = std::max(longest, double(std::hypot(flow.x[p] - column, flow.y[p] - row)));
            for (uint32_t c = 0; c < 3; ++c) {
                warped[c * pixels + p] = sample(a.image, s, c, flow.x[p], flow.y[p]);
                const double d = warped[c * pixels + p] - b.image[c * pixels + p];
                const double e = a.image[c * pixels + p] - b.image[c * pixels + p];
                warpedError += d * d;
                plainError += e * e;
            }
        }
        // The flow's check: the surface point a's G-buffer shows at the flow's target (nearest pixel) against b's.
        size_t same = 0, crateValid = 0, crateSame = 0; // also on the crate alone (red in the class G-buffer)
        const auto value16 = [](const std::vector<uint8_t>& bgra, size_t p) {
            return float(bgra[4 * p + 2] * 256u + bgra[4 * p + 1]) / 65535.0f;
        };
        for (size_t p = 0; p < pixels; ++p) {
            if (!flow.valid[p])
                continue;
            const size_t q =
                size_t(std::min(flow.y[p], float(s) - 1.0f)) * s + size_t(std::min(flow.x[p], float(s) - 1.0f));
            const auto wrapped = [](float d) { return std::min(std::abs(d), 1.0f - std::abs(d)); };
            const bool sameId = a.u[4 * q] == b.u[4 * p] && a.v[4 * q] == b.v[4 * p];
            const bool close = wrapped(value16(a.u, q) - value16(b.u, p)) < 0.05f &&
                               wrapped(value16(a.v, q) - value16(b.v, p)) < 0.05f;
            same += sameId && close;
            if (b.classes[4 * p + 2] > 127 && b.classes[4 * p + 1] < 64) {
                ++crateValid;
                crateSame += sameId && close;
            }
        }

        // The panel: a | b | flow | a warped onto b | 4 x |warped a - b| | mask.
        const uint32_t columns = 6, width = columns * s;
        std::vector<uint8_t> panel(size_t(width) * s * 3, 0);
        const auto put = [&](uint32_t column, size_t p, std::array<float, 3> rgb) {
            const size_t y = p / s, x = column * s + p % s;
            for (uint32_t c = 0; c < 3; ++c)
                panel[(y * width + x) * 3 + c] = byte(rgb[c]);
        };
        for (size_t p = 0; p < pixels; ++p) {
            put(0, p, {a.image[p], a.image[pixels + p], a.image[2 * pixels + p]});
            put(1, p, {b.image[p], b.image[pixels + p], b.image[2 * pixels + p]});
            const float column = float(p % s) + 0.5f, row = float(size_t(p / s)) + 0.5f;
            put(2, p,
                flow.valid[p] ? flowColour(flow.x[p] - column, flow.y[p] - row, float(longest))
                              : std::array<float, 3>{});
            if (flow.valid[p]) {
                put(3, p, {warped[p], warped[pixels + p], warped[2 * pixels + p]});
                std::array<float, 3> difference{};
                for (uint32_t c = 0; c < 3; ++c)
                    difference[c] = 4.0f * std::abs(warped[c * pixels + p] - b.image[c * pixels + p]);
                put(4, p, difference);
            }
            const float m = flow.valid[p] ? 1.0f : 0.0f;
            put(5, p, {m, m, m});
        }
        std::filesystem::create_directories(outDir);
        writeImage(outDir / "flow_pair.png", panel.data(), width, s, 3);

        const double n = double(std::max<size_t>(valid, 1) * 3);
        std::printf("views at %.1f and %.1f degrees, %ux%u: %zu of %zu pixels valid (%.1f %%), longest flow %.2f px\n",
                    azimuth, azimuth + delta, s, s, valid, pixels, 100.0 * double(valid) / double(pixels), longest);
        std::printf("flow check: %zu of %zu valid pixels find the same surface point in a (%.2f %%)\n", same, valid,
                    100.0 * double(same) / double(std::max<size_t>(valid, 1)));
        std::printf("            on the crate: %zu of %zu (%.2f %%)\n", crateSame, crateValid,
                    100.0 * double(crateSame) / double(std::max<size_t>(crateValid, 1)));
        std::printf("masked MSE b vs a: %.6f without the warp, %.6f with it\n", plainError / n, warpedError / n);
        std::cout << "wrote " << (outDir / "flow_pair.png") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "crate_flow_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
