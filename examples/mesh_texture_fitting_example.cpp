// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Recovers the mesh and texture of a synthetic pumpkin from rendered, masked views: a noise-textured icosphere is
// rendered with differentiable rasterization (the hardware rasterizer decides visibility, Slang compute shaders
// recompute and differentiate the shading and the silhouettes) and fitted by gradient descent. Every training step
// is one pre-recorded command buffer and one queue submit (raster::MeshTraining).
//
// With --scene DIR, fits photographs with masks and camera poses prepared by scripts/mesh_fitting/prepare_data.py
// instead: training on the scene's training views, evaluating on its validation views.
//
// Usage: mesh_texture_fitting_example [--epochs N] [--size S] [--views V] [--scene DIR] [--texture-width W]
//                                     [--laplacian L]
// Writes target views, progress frames of a held-out view and the fitted mesh with its texture to
// build/TestingOutput/mesh_texture_fitting/ (a subdirectory per scene with --scene).

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
#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/raster/fitting_scene.hpp"
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

// Silhouette IoU (coverage above one half) and PSNR inside the object (both covered) of images against targets.
std::pair<double, double> compare(const std::vector<float>& images, const std::vector<float>& targets) {
    int both = 0, either = 0, count = 0;
    double squaredError = 0.0;
    for (size_t p = 0; p < images.size() / 4; ++p) {
        const float *a = &images[4 * p], *b = &targets[4 * p];
        both += a[3] > 0.5f && b[3] > 0.5f;
        either += a[3] > 0.5f || b[3] > 0.5f;
        if (a[3] > 0.99f && b[3] > 0.99f)
            for (int c = 0; c < 3; ++c, ++count)
                squaredError += (a[c] - b[c]) * (a[c] - b[c]);
    }
    return {double(both) / double(std::max(either, 1)), 10.0 * std::log10(count / std::max(squaredError, 1e-12))};
}

} // namespace

int main(int argc, char** argv) {
    uint32_t epochs = 60, size = 128, views = 24, textureWidth = 0;
    float laplacianWeight = -1.0f; // negative: the default of MeshTrainingOptions
    std::filesystem::path sceneDirectory;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--epochs" && i + 1 < argc)
            epochs = uint32_t(std::stoul(argv[++i]));
        else if (argument == "--size" && i + 1 < argc)
            size = uint32_t(std::stoul(argv[++i]));
        else if (argument == "--views" && i + 1 < argc)
            views = std::max(3u, uint32_t(std::stoul(argv[++i])));
        else if (argument == "--scene" && i + 1 < argc)
            sceneDirectory = argv[++i];
        else if (argument == "--texture-width" && i + 1 < argc)
            textureWidth = uint32_t(std::stoul(argv[++i]));
        else if (argument == "--laplacian" && i + 1 < argc)
            laplacianWeight = std::stof(argv[++i]);
    }

    HeadlessFrontend frontend;
    VulkanContext& vulkanContext = frontend.getKlartraumEngine().getVulkanContext();
    raster::MeshTrainingOptions options;
    std::vector<glm::mat4> cameras, heldOut;
    std::vector<float> targets, heldOutTargets;
    // Maps the unit sphere the fitting works in to the scene's coordinates (identity for the synthetic scene).
    glm::mat4 sceneFromUnit(1.0f);
    std::filesystem::path output = "build/TestingOutput/mesh_texture_fitting";
    raster::TriangleMesh truth;

    if (sceneDirectory.empty()) {
        // The synthetic pumpkin, rendered from training views on three rings (below, at and above the equator) and
        // a held-out view between them.
        options.width = options.height = size;
        const uint32_t perRing = views / 3;
        for (const auto& [elevation, start] : {std::pair{-0.25f, 0.0f}, {0.3f, 0.39f}, {0.85f, 0.0f}}) {
            const auto ring = raster::cameraRing(perRing, 3.5f, elevation, 0.8f, 1.0f, start);
            cameras.insert(cameras.end(), ring.begin(), ring.end());
        }
        heldOut = raster::cameraRing(1, 3.5f, 0.15f, 0.8f, 1.0f, 2.3f);
        truth = raster::syntheticPumpkin(4);
        const std::vector<float> texture = raster::syntheticPumpkinTexture(options.textureWidth, options.textureHeight);
        targets = raster::renderImages(vulkanContext, truth, texture, options.textureWidth, options.textureHeight,
                                       cameras, size, size);
        heldOutTargets = raster::renderImages(vulkanContext, truth, texture, options.textureWidth,
                                              options.textureHeight, heldOut, size, size);
    } else {
        // Photographs prepared by scripts/mesh_fitting/prepare_data.py; the enclosing sphere becomes the unit sphere,
        // so the default step sizes and regularizer weight apply at any scene scale.
        const raster::FittingScene scene = raster::loadFittingScene(sceneDirectory);
        options.width = scene.width;
        options.height = scene.height;
        sceneFromUnit = glm::scale(glm::translate(glm::mat4(1.0f), scene.centre), glm::vec3(scene.radius));
        for (const auto& camera : scene.camerasOf(scene.train))
            cameras.push_back(camera * sceneFromUnit);
        for (const auto& camera : scene.camerasOf(scene.validation))
            heldOut.push_back(camera * sceneFromUnit);
        targets = scene.targetsOf(scene.train);
        heldOutTargets = scene.targetsOf(scene.validation);
        output /= sceneDirectory.filename();
        options.textureWidth = 128;
        options.textureHeight = 64;
    }
    if (laplacianWeight >= 0.0f)
        options.laplacianWeight = laplacianWeight;
    if (textureWidth > 0) {
        options.textureWidth = textureWidth;
        options.textureHeight = std::max(1u, textureWidth / 2);
    }
    std::filesystem::create_directories(output / "frames");
    const uint32_t width = options.width, height = options.height;
    const size_t pixels = size_t(width) * height;
    std::vector<const float*> shown;
    for (size_t view = 0; view < cameras.size(); view += std::max<size_t>(1, cameras.size() / 4))
        shown.push_back(&targets[4 * pixels * view]);
    writePpm(output / "targets.ppm", shown, width, height);

    const raster::TriangleMesh start = raster::icosphere(4);
    raster::MeshTraining training(vulkanContext, start, cameras, options);
    training.setTargets(targets);
    const auto render = [&](const std::vector<glm::mat4>& views) {
        return raster::renderImages(vulkanContext, training.mesh(), training.texture(), options.textureWidth,
                                    options.textureHeight, views, width, height);
    };
    const auto [startIou, startPsnr] = compare(render(heldOut), heldOutTargets);
    std::cout << "fitting an icosphere (" << start.positions.size() << " vertices, " << options.textureWidth << "x"
              << options.textureHeight << " texels) to " << cameras.size() << " views of " << width << "x" << height
              << " pixels; " << heldOut.size() << " held-out views, start silhouette IoU " << startIou << std::endl;
    double trainingSeconds = 0.0;
    for (uint32_t epoch = 0; epoch <= epochs; ++epoch) {
        // A progress frame: the first held-out view's target beside the current fit.
        const std::vector<float> image = render({heldOut[0]});
        std::ostringstream name;
        name << "epoch_" << std::setw(4) << std::setfill('0') << epoch << ".ppm";
        writePpm(output / "frames" / name.str(), {heldOutTargets.data(), image.data()}, width, height);
        if (epoch == epochs)
            break;
        double loss = 0.0;
        const auto begin = std::chrono::steady_clock::now();
        for (size_t v = 0; v < cameras.size(); ++v) {
            training.step();
            loss += training.loss();
        }
        trainingSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        if (epoch % 10 == 0 || epoch + 1 == epochs)
            std::cout << "epoch " << epoch << ": loss " << loss / double(cameras.size()) << std::endl;
    }
    const uint32_t steps = epochs * uint32_t(cameras.size());
    std::cout << steps << " steps in " << trainingSeconds << " s (" << 1000.0 * trainingSeconds / steps
              << " ms per step, including the loss readback)" << std::endl;

    const auto [trainingIou, trainingPsnr] = compare(render(cameras), targets);
    std::cout << "training views: silhouette IoU " << trainingIou << ", PSNR inside the object " << trainingPsnr
              << " dB" << std::endl;
    const auto [iou, psnr] = compare(render(heldOut), heldOutTargets);
    std::cout << "held-out views: silhouette IoU " << iou << ", PSNR inside the object " << psnr << " dB" << std::endl;

    raster::TriangleMesh fitted = training.mesh();
    for (auto& p : fitted.positions)
        p = glm::vec3(sceneFromUnit * glm::vec4(p, 1.0f));
    writeTexture(output / "fitted_texture.ppm", training.texture(), options.textureWidth, options.textureHeight);
    raster::writeObj(fitted, output / "fitted.obj", "fitted_texture.ppm");
    if (!truth.positions.empty())
        raster::writeObj(truth, output / "truth.obj");
    std::cout << "wrote " << output.string() << "/{targets.ppm, frames/, fitted.obj, fitted_texture.ppm"
              << (truth.positions.empty() ? "" : ", truth.obj") << "}" << std::endl;
    return 0;
}
