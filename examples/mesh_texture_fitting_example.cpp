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
//                                     [--laplacian L] [--final-lr F]
//                                     [--display-view N]
//                                     [--subdivisions S] [--view-colors] [--view-color-weight W]
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

// The position of the camera of a clip-from-world matrix: the point with clip x, y and w zero.
glm::vec3 cameraCentre(const glm::mat4& clipFromWorld) {
    glm::mat3 rows;
    glm::vec3 offsets;
    for (int r = 0, i = 0; r < 4; ++r) {
        if (r == 2)
            continue;
        rows[0][i] = clipFromWorld[0][r]; // glm: [column][row]
        rows[1][i] = clipFromWorld[1][r];
        rows[2][i] = clipFromWorld[2][r];
        offsets[i++] = -clipFromWorld[3][r];
    }
    return glm::inverse(rows) * offsets;
}

// An image of four floats per pixel with its size.
struct Image {
    std::vector<float> pixels;
    uint32_t width = 0, height = 0;
};

// Turns an image by 90 degrees clockwise.
Image turnClockwise(const Image& image) {
    Image result{std::vector<float>(image.pixels.size()), image.height, image.width};
    for (uint32_t y = 0; y < result.height; ++y)
        for (uint32_t x = 0; x < result.width; ++x)
            for (int c = 0; c < 4; ++c)
                result.pixels[4 * (size_t(y) * result.width + x) + c] =
                    image.pixels[4 * (size_t(image.height - 1 - x) * image.width + y) + c];
    return result;
}

// Centres an image on a transparent square of side `side`.
Image padToSquare(const Image& image, uint32_t side) {
    Image result{std::vector<float>(4 * size_t(side) * side, 0.0f), side, side};
    const uint32_t left = (side - image.width) / 2, top = (side - image.height) / 2;
    for (uint32_t y = 0; y < image.height; ++y)
        std::copy_n(&image.pixels[4 * size_t(y) * image.width], 4 * image.width,
                    &result.pixels[4 * (size_t(y + top) * side + left)]);
    return result;
}

// How many clockwise quarter turns make `up`, seen by `camera` from the point `at`, point up on the screen.
int uprightTurns(const glm::mat4& camera, glm::vec3 at, glm::vec3 up) {
    const glm::vec4 a = camera * glm::vec4(at, 1.0f), b = camera * glm::vec4(at + 0.1f * up, 1.0f);
    glm::vec2 direction = glm::vec2(b) / b.w - glm::vec2(a) / a.w; // y down, like the screen
    int best = 0;
    float bestUp = -2.0f;
    for (int turns = 0; turns < 4; ++turns) {
        const float upness = -glm::normalize(direction).y;
        if (upness > bestUp) {
            bestUp = upness;
            best = turns;
        }
        direction = glm::vec2(-direction.y, direction.x); // a clockwise quarter turn on the screen
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    uint32_t epochs = 0, size = 128, views = 24, textureWidth = 0; // epochs 0: 60, or 150 with --scene
    float laplacianWeight = -1.0f;                                 // negative: the default of MeshTrainingOptions
    float finalLearningRate = 0.1f;
    uint32_t subdivisions = 0; // 0: 4, or 5 with --scene
    bool viewColors = false;
    float viewColorWeight = -1.0f; // negative: the default of MeshTrainingOptions
    int displayView = -1;          // with --scene: the photo whose side of the object the progress frames show
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
        else if (argument == "--subdivisions" && i + 1 < argc)
            subdivisions = uint32_t(std::stoul(argv[++i]));
        else if (argument == "--view-color-weight" && i + 1 < argc)
            viewColorWeight = std::stof(argv[++i]);
        else if (argument == "--view-colors")
            viewColors = true;
        else if (argument == "--display-view" && i + 1 < argc)
            displayView = std::stoi(argv[++i]);
        else if (argument == "--final-lr" && i + 1 < argc)
            finalLearningRate = std::stof(argv[++i]);
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
    // With --scene, the progress frames show a photo (turned upright) beside the fit seen upright from that side.
    Image displayPhoto;
    glm::mat4 displayCamera(1.0f);

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
        // The rotation turns the sphere's pole (+y) to the scene's up. The object rests on the ground, so the part
        // of its enclosing sphere that no camera sees over the horizon faces down; up is opposite its mean
        // direction (or, if the cameras see all around, the mean direction to the cameras). The top of the object
        // then maps to the top of the texture and the unseen underside to its bottom pole.
        glm::vec3 sceneUp(0.0f), unseen(0.0f);
        std::vector<glm::vec3> cameraOffsets;
        for (const auto& camera : scene.cameras) {
            cameraOffsets.push_back(cameraCentre(camera) - scene.centre);
            sceneUp += glm::normalize(cameraOffsets.back());
        }
        for (const auto& direction : raster::icosphere(3).positions) {
            bool seen = false;
            for (const auto& offset : cameraOffsets)
                seen = seen || glm::dot(direction, offset) > scene.radius;
            if (!seen)
                unseen += direction;
        }
        sceneUp = glm::normalize(glm::length(unseen) > 0.0f ? -unseen : sceneUp);
        const glm::vec3 across =
            glm::normalize(glm::cross(sceneUp, std::abs(sceneUp.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1)));
        const glm::mat4 rotation(glm::vec4(across, 0.0f), glm::vec4(sceneUp, 0.0f),
                                 glm::vec4(glm::cross(across, sceneUp), 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        sceneFromUnit = glm::translate(glm::mat4(1.0f), scene.centre) * rotation *
                        glm::scale(glm::mat4(1.0f), glm::vec3(scene.radius));
        for (const auto& camera : scene.camerasOf(scene.train))
            cameras.push_back(camera * sceneFromUnit);
        for (const auto& camera : scene.camerasOf(scene.validation))
            heldOut.push_back(camera * sceneFromUnit);
        targets = scene.targetsOf(scene.train);
        heldOutTargets = scene.targetsOf(scene.validation);
        output /= sceneDirectory.filename();
        // Up: the cameras stand above the ground, so their mean direction from the object. The display camera looks
        // horizontally at the side the display photo shows, from a little above.
        glm::vec3 up(0.0f);
        std::vector<glm::mat4> unitCameras;
        for (const auto& camera : scene.cameras) {
            unitCameras.push_back(camera * sceneFromUnit);
            up += glm::normalize(cameraCentre(unitCameras.back()));
        }
        up = glm::normalize(up);
        const uint32_t shown = displayView >= 0 ? uint32_t(displayView) : scene.validation.at(0);
        const glm::vec3 seen = cameraCentre(unitCameras.at(shown));
        const glm::vec3 side = glm::normalize(seen - glm::dot(seen, up) * up);
        const float elevation = 0.3f;
        const glm::vec3 eye = 3.5f * (std::cos(elevation) * side + std::sin(elevation) * up);
        glm::mat4 projection = glm::perspectiveRH_ZO(0.8f, 1.0f, 0.1f, 20.0f);
        projection[1][1] *= -1.0f; // Vulkan: y down
        displayCamera = projection * glm::lookAtRH(eye, glm::vec3(0.0f), up);
        displayPhoto = {scene.targetsOf({shown}), scene.width, scene.height};
        for (int turn = uprightTurns(unitCameras[shown], glm::vec3(0.0f), up); turn > 0; --turn)
            displayPhoto = turnClockwise(displayPhoto);
        std::cout << "progress frames: view " << shown << " (turned upright) beside the fit seen from its side"
                  << std::endl;
        options.textureWidth = 128;
        options.textureHeight = 64;
    }
    options.viewColors = viewColors;
    if (viewColorWeight >= 0.0f)
        options.viewColorWeight = viewColorWeight;
    if (laplacianWeight >= 0.0f)
        options.laplacianWeight = laplacianWeight;
    // Photographs carry more detail than the synthetic pumpkin: a finer start mesh, trained longer.
    const bool photographs = !sceneDirectory.empty();
    epochs = epochs > 0 ? epochs : (photographs ? 150 : 60);
    subdivisions = subdivisions > 0 ? subdivisions : (photographs ? 5 : 4);
    if (textureWidth > 0) {
        options.textureWidth = textureWidth;
        options.textureHeight = std::max(1u, textureWidth / 2);
    }
    // The learning rates decay over the whole run, so the fit settles instead of jittering between views.
    for (auto* adam : {&options.geometry, &options.texture, &options.viewColor}) {
        adam->decaySteps = epochs * uint32_t(cameras.size());
        adam->finalFactor = finalLearningRate;
    }
    std::filesystem::remove_all(output / "frames"); // frames of an earlier, longer run would follow this one's
    std::filesystem::create_directories(output / "frames");
    const uint32_t width = options.width, height = options.height;
    const size_t pixels = size_t(width) * height;
    std::vector<const float*> shown;
    for (size_t view = 0; view < cameras.size(); view += std::max<size_t>(1, cameras.size() / 4))
        shown.push_back(&targets[4 * pixels * view]);
    writePpm(output / "targets.ppm", shown, width, height);

    const raster::TriangleMesh start = raster::icosphere(subdivisions);
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
        // A progress frame: the first held-out view's target beside the current fit, or with --scene the display
        // photo beside the fit seen upright from its side.
        std::ostringstream name;
        name << "epoch_" << std::setw(4) << std::setfill('0') << epoch << ".ppm";
        if (displayPhoto.pixels.empty()) {
            const std::vector<float> image = render({heldOut[0]});
            writePpm(output / "frames" / name.str(), {heldOutTargets.data(), image.data()}, width, height);
        } else {
            const uint32_t side = std::max(width, height);
            const std::vector<float> fit =
                raster::renderImages(vulkanContext, training.mesh(), training.texture(), options.textureWidth,
                                     options.textureHeight, {displayCamera}, side, side);
            const Image photo = padToSquare(displayPhoto, side);
            writePpm(output / "frames" / name.str(), {photo.pixels.data(), fit.data()}, side, side);
        }
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

    if (const std::vector<float> corrections = training.viewColors(); !corrections.empty()) {
        glm::vec3 low(1e9f), high(-1e9f), mean(0.0f), offsetLow(1e9f), offsetHigh(-1e9f);
        for (size_t v = 0; v < corrections.size() / 8; ++v) {
            const glm::vec3 gain(corrections[8 * v], corrections[8 * v + 1], corrections[8 * v + 2]);
            const glm::vec3 offset(corrections[8 * v + 4], corrections[8 * v + 5], corrections[8 * v + 6]);
            low = glm::min(low, gain);
            high = glm::max(high, gain);
            mean += gain / float(corrections.size() / 8);
            offsetLow = glm::min(offsetLow, offset);
            offsetHigh = glm::max(offsetHigh, offset);
        }
        std::cout << "per-view gains: mean " << mean.r << " " << mean.g << " " << mean.b << ", range " << low.r << "-"
                  << high.r << " " << low.g << "-" << high.g << " " << low.b << "-" << high.b << "; offsets "
                  << offsetLow.r << "-" << offsetHigh.r << std::endl;
    }
    const std::vector<float> trainingImages = render(cameras);
    // Residuals of four training views: target, fit and four times the colour error, one view per row.
    {
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
    const auto [trainingIou, trainingPsnr] = compare(trainingImages, targets);
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
