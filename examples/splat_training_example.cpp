// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Trains 3D Gaussian splats (Kerbl et al. 2023) on photographs with masks and camera poses prepared by
// scripts/mesh_fitting/prepare_data.py. The Gaussians start at the surface points of the visual hull carved from the
// masks; every training step (projection, tile sorting, blending, image loss, backward pass and Adam) is one
// pre-recorded command buffer and one queue submit, with densification between steps (splatting::SplatTraining).
//
// Usage: splat_training_example --scene DIR [--steps N] [--display-view N]
// Writes progress frames (a photo turned upright beside the splats seen upright from its side), residuals of four
// training views and the trained Gaussians as a PLY file in scene coordinates to
// build/TestingOutput/splat_training/<scene>/.

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

#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/raster/fitting_scene.hpp"
#include "klartraum/splatting/splat_training.hpp"

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
    uint32_t steps = 6000;
    int displayView = -1;
    std::filesystem::path sceneDirectory;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--scene" && i + 1 < argc)
            sceneDirectory = argv[++i];
        else if (argument == "--steps" && i + 1 < argc)
            steps = uint32_t(std::stoul(argv[++i]));
        else if (argument == "--display-view" && i + 1 < argc)
            displayView = std::stoi(argv[++i]);
    }
    if (sceneDirectory.empty()) {
        std::cerr << "usage: splat_training_example --scene DIR [--steps N] [--display-view N]" << std::endl;
        return 1;
    }

    HeadlessFrontend frontend;
    VulkanContext& vulkanContext = frontend.getKlartraumEngine().getVulkanContext();
    const raster::FittingScene scene = raster::loadFittingScene(sceneDirectory);
    if (scene.points.empty()) {
        std::cerr << sceneDirectory << " has no points.bin: prepare it again with prepare_data.py" << std::endl;
        return 1;
    }
    // Training works in the unit sphere around the object, up on +y (see FittingScene::sceneFromUnitSphere()).
    const glm::mat4 sceneFromUnit = scene.sceneFromUnitSphere(), unitFromScene = glm::inverse(sceneFromUnit);
    std::vector<glm::mat4> cameras, heldOut;
    for (const auto& camera : scene.camerasOf(scene.train))
        cameras.push_back(camera * sceneFromUnit);
    for (const auto& camera : scene.camerasOf(scene.validation))
        heldOut.push_back(camera * sceneFromUnit);
    const std::vector<float> targets = scene.targetsOf(scene.train), heldOutTargets = scene.targetsOf(scene.validation);
    std::vector<glm::vec3> points;
    for (const auto& p : scene.points)
        points.push_back(glm::vec3(unitFromScene * glm::vec4(p, 1.0f)));
    const raster::DisplayView display =
        raster::displayView(scene, displayView >= 0 ? uint32_t(displayView) : scene.validation.at(0));

    const std::filesystem::path output =
        std::filesystem::path("build/TestingOutput/splat_training") / sceneDirectory.filename();
    std::filesystem::remove_all(output / "frames"); // frames of an earlier, longer run would follow this one's
    std::filesystem::create_directories(output / "frames");

    splatting::SplatTrainingOptions options;
    options.raster.width = scene.width;
    options.raster.height = scene.height;
    options.raster.capacity = 1u << 22;
    options.steps = steps;
    options.densifyUntil = steps / 2;
    splatting::SplatRasterOptions displayOptions = options.raster;
    displayOptions.width = displayOptions.height = display.side;
    displayOptions.capacity = 1u << 20;
    splatting::SplatTraining training(vulkanContext, splatting::splatsAt(points, scene.spacing / scene.radius), cameras,
                                      targets, options);
    std::cout << "training " << points.size() << " Gaussians (hull surface points) on " << cameras.size()
              << " views of " << scene.width << "x" << scene.height << " pixels; " << heldOut.size()
              << " held-out views" << std::endl;

    const uint32_t frameEvery = std::max(1u, steps / 150);
    double trainingSeconds = 0.0, loss = 0.0;
    for (uint32_t step = 0; step <= steps; ++step) {
        if (step % frameEvery == 0) {
            const std::vector<float> fit =
                splatting::renderSplats(vulkanContext, training.splats(), {display.camera}, displayOptions);
            std::ostringstream name; // numbered consecutively, as video encoders expect
            name << "frame_" << std::setw(4) << std::setfill('0') << step / frameEvery << ".ppm";
            writePpm(output / "frames" / name.str(), {display.photo.data(), fit.data()}, display.side, display.side);
        }
        if (step == steps)
            break;
        const auto begin = std::chrono::steady_clock::now();
        training.step();
        trainingSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        loss += training.loss();
        if ((step + 1) % 500 == 0) {
            std::cout << "step " << step + 1 << ": loss " << loss / 500.0 << ", " << training.splats().size()
                      << " Gaussians, " << training.instanceCount() << " tile instances" << std::endl;
            loss = 0.0;
        }
    }
    std::cout << steps << " steps in " << trainingSeconds << " s (" << 1000.0 * trainingSeconds / steps
              << " ms per step, including densification and the loss readback)" << std::endl;

    const splatting::SplatSet splats = training.splats();
    const std::vector<float> trainingImages = splatting::renderSplats(vulkanContext, splats, cameras, options.raster);
    const auto [trainingIou, trainingPsnr] = compare(trainingImages, targets);
    std::cout << "training views: silhouette IoU " << trainingIou << ", PSNR inside the object " << trainingPsnr
              << " dB" << std::endl;
    const auto [iou, psnr] =
        compare(splatting::renderSplats(vulkanContext, splats, heldOut, options.raster), heldOutTargets);
    std::cout << "held-out views: silhouette IoU " << iou << ", PSNR inside the object " << psnr << " dB" << std::endl;

    // Residuals of four training views: photo, splats and four times the colour error, one view per row.
    {
        const uint32_t width = scene.width, height = scene.height;
        const size_t pixels = size_t(width) * height;
        std::ofstream out(output / "residuals.ppm", std::ios::binary);
        out << "P6\n" << 3 * width << " " << 4 * height << "\n255\n";
        for (size_t row = 0; row < 4; ++row) {
            const size_t view = row * cameras.size() / 4;
            const float* target = &targets[4 * pixels * view];
            const float* fit = &trainingImages[4 * pixels * view];
            for (uint32_t y = 0; y < height; ++y)
                for (int panel = 0; panel < 3; ++panel)
                    for (uint32_t x = 0; x < width; ++x) {
                        const size_t p = 4 * (size_t(y) * width + x);
                        for (int c = 0; c < 3; ++c) {
                            const float value = panel == 0   ? target[p + c] + 1.0f - target[p + 3]
                                                : panel == 1 ? fit[p + c] + 1.0f - fit[p + 3]
                                                             : 4.0f * std::abs(fit[p + c] - target[p + c]);
                            out.put(char(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f)));
                        }
                    }
        }
    }
    splatting::writeSplatPly(splatting::transformSplats(splats, sceneFromUnit), output / "splats.ply");
    std::cout << "wrote " << output.string() << "/{frames/, residuals.ppm, splats.ply} (" << splats.size()
              << " Gaussians)" << std::endl;
    return 0;
}
