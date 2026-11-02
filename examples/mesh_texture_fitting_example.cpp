// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Recovers the mesh and texture of a synthetic pumpkin from rendered, masked views: a noise-textured icosphere is
// rendered with differentiable rasterization (the hardware rasterizer decides visibility, Slang compute shaders
// recompute and differentiate the shading and the silhouettes) and fitted by gradient descent. Every training step
// is one pre-recorded command buffer and one queue submit (raster::MeshTraining).
//
// Usage: mesh_texture_fitting_example [--epochs N] [--size S] [--views V]
// Writes target views, progress frames of a held-out view and the fitted mesh with its texture to
// build/TestingOutput/mesh_texture_fitting/.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/raster/mesh_training.hpp"

namespace {

using namespace klartraum;

// Writes images of four floats per pixel side by side as a PPM, premultiplied colours over a white background.
void writePpm(const std::filesystem::path& path, const std::vector<const float*>& images, uint32_t width,
              uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << width * images.size() << " " << height << "\n255\n";
    for (uint32_t y = 0; y < height; ++y)
        for (const float* image : images)
            for (uint32_t x = 0; x < width; ++x) {
                const float* p = &image[4 * (size_t(y) * width + x)];
                for (int c = 0; c < 3; ++c)
                    out.put(char(std::lround(std::clamp(p[c] + (1.0f - p[3]), 0.0f, 1.0f) * 255.0f)));
            }
}

// Writes a texture of four floats per texel as a PPM.
void writeTexture(const std::filesystem::path& path, const std::vector<float>& texels, uint32_t width,
                  uint32_t height) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < size_t(width) * height; ++i)
        for (int c = 0; c < 3; ++c)
            out.put(char(std::lround(std::clamp(texels[4 * i + c], 0.0f, 1.0f) * 255.0f)));
}

} // namespace

int main(int argc, char** argv) {
    uint32_t epochs = 60, size = 128, views = 24;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--epochs" && i + 1 < argc)
            epochs = uint32_t(std::stoul(argv[++i]));
        else if (argument == "--size" && i + 1 < argc)
            size = uint32_t(std::stoul(argv[++i]));
        else if (argument == "--views" && i + 1 < argc)
            views = std::max(3u, uint32_t(std::stoul(argv[++i])));
    }
    const std::filesystem::path output = "build/TestingOutput/mesh_texture_fitting";
    std::filesystem::create_directories(output / "frames");

    HeadlessFrontend frontend;
    VulkanContext& vulkanContext = frontend.getKlartraumEngine().getVulkanContext();
    raster::MeshTrainingOptions options;
    options.width = options.height = size;

    // Training views on three rings (below, at and above the equator); a held-out view between them.
    std::vector<glm::mat4> cameras;
    const uint32_t perRing = views / 3;
    for (const auto& [elevation, start] : {std::pair{-0.25f, 0.0f}, {0.3f, 0.39f}, {0.85f, 0.0f}}) {
        const auto ring = raster::cameraRing(perRing, 3.5f, elevation, 0.8f, 1.0f, start);
        cameras.insert(cameras.end(), ring.begin(), ring.end());
    }
    const std::vector<glm::mat4> heldOut = raster::cameraRing(1, 3.5f, 0.15f, 0.8f, 1.0f, 2.3f);

    const raster::TriangleMesh pumpkin = raster::syntheticPumpkin(4);
    const std::vector<float> truthTexture =
        raster::syntheticPumpkinTexture(options.textureWidth, options.textureHeight);
    const auto render = [&](const raster::TriangleMesh& mesh, const std::vector<float>& texture,
                            const std::vector<glm::mat4>& views) {
        return raster::renderImages(vulkanContext, mesh, texture, options.textureWidth, options.textureHeight, views,
                                    size, size);
    };
    const size_t pixels = size_t(size) * size;
    const std::vector<float> targets = render(pumpkin, truthTexture, cameras);
    const std::vector<float> heldOutTarget = render(pumpkin, truthTexture, heldOut);
    writePpm(output / "targets.ppm", {&targets[0], &targets[4 * pixels * perRing], &targets[8 * pixels * perRing]},
             size, size);

    raster::MeshTraining training(vulkanContext, raster::icosphere(4), cameras, options);
    training.setTargets(targets);
    std::cout << "fitting an icosphere (" << training.mesh().positions.size() << " vertices, " << options.textureWidth
              << "x" << options.textureHeight << " texels) to " << cameras.size() << " views of " << size << "x" << size
              << " pixels" << std::endl;
    double trainingSeconds = 0.0;
    for (uint32_t epoch = 0; epoch <= epochs; ++epoch) {
        // A progress frame: the held-out view's target beside the current fit.
        const std::vector<float> image = render(training.mesh(), training.texture(), heldOut);
        std::ostringstream name;
        name << "epoch_" << std::setw(4) << std::setfill('0') << epoch << ".ppm";
        writePpm(output / "frames" / name.str(), {heldOutTarget.data(), image.data()}, size, size);
        if (epoch == epochs)
            break;
        double loss = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (size_t v = 0; v < cameras.size(); ++v) {
            training.step();
            loss += training.loss();
        }
        trainingSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (epoch % 10 == 0 || epoch + 1 == epochs)
            std::cout << "epoch " << epoch << ": loss " << loss / double(cameras.size()) << std::endl;
    }
    const uint32_t steps = epochs * uint32_t(cameras.size());
    std::cout << steps << " steps in " << trainingSeconds << " s (" << 1000.0 * trainingSeconds / steps
              << " ms per step, including the loss readback)" << std::endl;

    // Held-out metrics: silhouette IoU and PSNR inside the object.
    const std::vector<float> image = render(training.mesh(), training.texture(), heldOut);
    int both = 0, either = 0, count = 0;
    double squaredError = 0.0;
    for (size_t p = 0; p < pixels; ++p) {
        const float *a = &image[4 * p], *b = &heldOutTarget[4 * p];
        both += a[3] > 0.5f && b[3] > 0.5f;
        either += a[3] > 0.5f || b[3] > 0.5f;
        if (a[3] > 0.99f && b[3] > 0.99f)
            for (int c = 0; c < 3; ++c, ++count)
                squaredError += (a[c] - b[c]) * (a[c] - b[c]);
    }
    std::cout << "held-out view: silhouette IoU " << double(both) / either << ", PSNR "
              << 10.0 * std::log10(count / squaredError) << " dB" << std::endl;

    writeTexture(output / "fitted_texture.ppm", training.texture(), options.textureWidth, options.textureHeight);
    raster::writeObj(training.mesh(), output / "fitted.obj", "fitted_texture.ppm");
    raster::writeObj(pumpkin, output / "truth.obj");
    std::cout << "wrote " << output.string() << "/{targets.ppm, frames/, fitted.obj, fitted_texture.ppm, truth.obj}"
              << std::endl;
    return 0;
}
