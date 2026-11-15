// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A toy scene for the photo enhancer pipeline: a crate (a box whose six faces have their own surface coordinates and
// object ids) standing on a floor, rendered by GBufferMeshRenderer from random cameras with all G-buffers. The views
// are written like the frames of tree_bonsai_example --fly-in --gbuffers (frames/classical_*.png and
// cameras_classical.csv), so scripts/neural/prepare_photo_pairs.py and add_surface_coordinates.py take them.
//
// With --orbit the camera circles the crate instead (a turntable), and with --model every view is also enhanced by a
// photo enhancer trained on the crate (neural::PhotoEnhancer: the learnable texture and the generator, on the GPU)
// into frames/enhanced_NNNN.png, under the appearance code of a training image (--appearance) or their mean.
// A latent-conditioned model (crate_training_example --latent) renders the crate with the latent of --latent-code,
// or with --latent-codes every view with its own (a walk through the latent space).
//
//   crate_gbuffer_example --views 180 --size 512 --seed 1 --out-dir build/TestingOutput/crate/flights/crate_a
//   crate_gbuffer_example --orbit --views 240 --model build/TestingOutput/crate/runs/c1/photo_enhancer_best
//   crate_gbuffer_example --orbit --views 120 --size 256 --sample-size 128 --no-floor --out-dir DIR \
//       --parameters build/TestingOutput/crate_views_i2i/runs/v3/parameters

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "crate_scene.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gbuffer_mesh_renderer.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/image_io.hpp"
#include "klartraum/neural/photo_enhancer.hpp"
#include "klartraum/neural/photo_enhancer_training.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/tree_model.hpp"

using namespace klartraum;

namespace {

// The ControlNet's depth of a sample (scripts/neural/prepare_photo_pairs.py --smooth-control 1): its stretched inverse
// depth (channel 3 of the photo enhancer's inputs) in whole 8-bit steps, blurred by a Gaussian of 1 pixel, as a grey
// binary PPM.
void writeControlDepth(const std::filesystem::path& path, const std::vector<float>& channels, uint32_t size) {
    const size_t pixels = size_t(size) * size;
    std::vector<float> grey(pixels), blurred(pixels), result(pixels);
    for (size_t p = 0; p < pixels; ++p)
        grey[p] = std::round(std::clamp(channels[3 * pixels + p], 0.0f, 1.0f) * 255.0f);
    float weights[7], sum = 0.0f;
    for (int k = -3; k <= 3; ++k)
        sum += weights[k + 3] = std::exp(-0.5f * float(k * k));
    for (float& w : weights)
        w /= sum;
    // Separable, with the border pixels repeated.
    const auto clampIndex = [size](int v) { return uint32_t(std::clamp(v, 0, int(size) - 1)); };
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            float v = 0.0f;
            for (int k = -3; k <= 3; ++k)
                v += weights[k + 3] * grey[size_t(y) * size + clampIndex(int(x) + k)];
            blurred[size_t(y) * size + x] = v;
        }
    std::vector<uint8_t> bytes(3 * pixels);
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            float v = 0.0f;
            for (int k = -3; k <= 3; ++k)
                v += weights[k + 3] * blurred[size_t(clampIndex(int(y) + k)) * size + x];
            const uint8_t byte = uint8_t(std::lround(std::clamp(v, 0.0f, 255.0f)));
            bytes[3 * (size_t(y) * size + x)] = bytes[3 * (size_t(y) * size + x) + 1] =
                bytes[3 * (size_t(y) * size + x) + 2] = byte;
        }
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << size << " " << size << "\n255\n";
    file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

// The texture's features of a view (the generator's input after its 11 G-buffer channels: 12 levels of 2): each
// level as an image, its two features as red and green, scaled to the level's largest magnitude (grey: 0), in a
// grid of 4 x 3 levels from coarse to fine.
void writeFeatures(const std::filesystem::path& directory, uint32_t view, const std::vector<float>& input,
                   uint32_t size) {
    const size_t pixels = size_t(size) * size;
    const uint32_t levels = uint32_t(input.size() / pixels - 11) / 2, columns = 4, rows = (levels + 3) / 4;
    std::vector<uint8_t> grid(size_t(columns) * size * rows * size * 3, 0);
    for (uint32_t level = 0; level < levels; ++level) {
        const float* f0 = input.data() + (11 + 2 * level) * pixels;
        const float* f1 = f0 + pixels;
        float largest = 1e-12f;
        for (size_t p = 0; p < pixels; ++p)
            largest = std::max({largest, std::abs(f0[p]), std::abs(f1[p])});
        const uint32_t gx = (level % columns) * size, gy = (level / columns) * size;
        for (uint32_t y = 0; y < size; ++y)
            for (uint32_t x = 0; x < size; ++x) {
                const size_t p = size_t(y) * size + x;
                uint8_t* rgb = &grid[((size_t(gy + y) * columns * size) + gx + x) * 3];
                rgb[0] = uint8_t(std::lround(127.5f + 127.5f * f0[p] / largest));
                rgb[1] = uint8_t(std::lround(127.5f + 127.5f * f1[p] / largest));
                rgb[2] = 128;
            }
    }
    char name[64];
    std::snprintf(name, sizeof(name), "features_%04u.png", view);
    writeImage(directory / name, grid.data(), columns * size, rows * size, 3);
}

// The sample's rendering (its RGB channels) with the background (class 0) white, as a binary PPM.
void writeOnWhite(const std::filesystem::path& path, const neural::PhotoEnhancerInputs& inputs, uint32_t size) {
    const size_t pixels = size_t(size) * size;
    std::vector<uint8_t> bytes(3 * pixels);
    for (size_t p = 0; p < pixels; ++p)
        for (size_t c = 0; c < 3; ++c) {
            const float v = inputs.classes[p] == 0.0f ? 1.0f : inputs.channels[c * pixels + p];
            bytes[3 * p + c] = uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
        }
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << size << " " << size << "\n255\n";
    file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path outDir = "build/TestingOutput/crate/flights/crate_a";
    uint32_t views = 180, size = 512, seed = 1, sampleSize = 256;
    bool orbit = false, fixedView = false, floor = true;
    std::string model;
    std::filesystem::path parameters, qwenBatch;
    bool qwenImageToImage = false;
    float untrained = 0.0f; // > 0: an untrained enhancer, its texture initialised to +-1e-4 x this
    bool features = false;
    int appearanceRow = -1;            // -1: the training codes' mean
    std::vector<float> latentCode;     // the crate's latent for a latent-conditioned model (empty: zeros)
    std::filesystem::path latentCodes; // a file of one latent per line (comma-separated), for view after view
    // The two lights of the shadows G-buffer (--lights), or drawn at random per view (--random-lights SEED).
    std::array<GBufferLight, 2> lights{GBufferLight{glm::vec3(0.45f, 0.8f, 0.35f), 0.05f, 1.0f},
                                       GBufferLight{glm::vec3(-0.6f, 0.5f, -0.3f), 0.05f, 0.5f}};
    bool randomLights = false;
    uint32_t lightSeed = 1;
    bool profile = false;   // time the trained model's every element on the GPU
    bool timeSteps = false; // time the trained model's steps on the CPU (submission to finish)
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--out-dir" && hasValue)
            outDir = argv[++i];
        else if (arg == "--views" && hasValue)
            views = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--size" && hasValue)
            size = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--seed" && hasValue)
            seed = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--orbit")
            orbit = true;
        else if (arg == "--fixed-view")
            fixedView = true;
        else if (arg == "--no-floor")
            floor = false;
        else if (arg == "--model" && hasValue)
            model = argv[++i];
        else if (arg == "--appearance" && hasValue)
            appearanceRow = std::stoi(argv[++i]);
        else if (arg == "--sample-size" && hasValue)
            sampleSize = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--parameters" && hasValue)
            parameters = argv[++i];
        else if (arg == "--qwen-batch" && hasValue)
            qwenBatch = argv[++i];
        else if (arg == "--qwen-image-to-image")
            qwenImageToImage = true;
        else if (arg == "--untrained" && hasValue)
            untrained = std::stof(argv[++i]);
        else if (arg == "--features")
            features = true;
        else if (arg == "--profile")
            profile = true;
        else if (arg == "--time")
            timeSteps = true;
        else if (arg == "--lights" && hasValue) {
            // azimuth and elevation in degrees, angular radius in radians, intensity; for light 1, then light 2
            std::stringstream numbers(argv[++i]);
            std::vector<float> values;
            for (std::string number; std::getline(numbers, number, ',');)
                values.push_back(std::stof(number));
            if (values.size() != 8)
                throw std::invalid_argument("--lights needs 8 numbers");
            for (int k = 0; k < 2; ++k) {
                const float azimuth = glm::radians(values[4 * k]), elevation = glm::radians(values[4 * k + 1]);
                lights[k].towardsLight = glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                                   std::cos(azimuth) * std::cos(elevation));
                lights[k].angularRadius = values[4 * k + 2];
                lights[k].intensity = values[4 * k + 3];
            }
        } else if (arg == "--random-lights" && hasValue) {
            randomLights = true;
            lightSeed = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--latent-codes" && hasValue)
            latentCodes = argv[++i];
        else if (arg == "--latent-code" && hasValue) {
            std::stringstream numbers(argv[++i]);
            latentCode.clear();
            for (std::string number; std::getline(numbers, number, ',');)
                latentCode.push_back(std::stof(number));
        } else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    try {
        HeadlessFrontend frontend;
        VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
        const auto vertices = crate::vertices(floor);
        const VkExtent2D extent{size, size};
        auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        GBufferMeshRenderer renderer(vc, vertices, target, extent);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(renderer.getOutput());
        // A photo enhancer trained in Klartraum (crate_training_example's saved parameters), run forward only.
        std::unique_ptr<neural::PhotoEnhancerTraining> trained;
        // A latent-conditioned model (crate_training_example --latent) and its latent size.
        const uint32_t latentSize = parameters.empty() ? 0 : neural::savedLatentSize(parameters.string());
        // Every view's latent: from --latent-codes (the lines repeating if there are fewer than views), else one.
        std::vector<std::vector<float>> viewLatents;
        if (!latentCodes.empty()) {
            std::ifstream file(latentCodes);
            for (std::string line; std::getline(file, line);) {
                std::stringstream numbers(line);
                std::vector<float> code;
                for (std::string number; std::getline(numbers, number, ',');)
                    code.push_back(std::stof(number));
                if (code.size() != latentSize)
                    throw std::runtime_error("--latent-codes: the model's latents have " + std::to_string(latentSize) +
                                             " numbers");
                viewLatents.push_back(code);
            }
            if (viewLatents.empty())
                throw std::runtime_error("--latent-codes: no latents in " + latentCodes.string());
        }
        if (latentSize > 0 && latentCode.empty())
            latentCode.assign(latentSize, 0.0f);
        if (latentSize > 0 && latentCode.size() != latentSize)
            throw std::runtime_error("--latent-code: the model's latents have " + std::to_string(latentSize) +
                                     " numbers");
        if (!parameters.empty() || untrained > 0.0f) {
            neural::PhotoEnhancerTrainingConfig config;
            config.forwardOnly = true;
            config.profile = profile;
            config.batch = 1;
            config.size = sampleSize;
            // The trained model's texture shape (saved with its parameters), else hash_texture.py's.
            config.texture = !parameters.empty() && std::filesystem::exists(parameters / "texture.txt")
                                 ? neural::HashTextureShape::fromFile((parameters / "texture.txt").string())
                                 : crate::textureShape();
            if (!parameters.empty() && std::filesystem::exists(parameters / "texture.txt")) {
                config.textureMlp = neural::TextureMlp::fromFile((parameters / "texture.txt").string());
                config.albedoTexture = neural::AlbedoTexture::fromFile((parameters / "texture.txt").string());
            }
            const auto codes = parameters / "codes.weight.bin";
            config.appearances =
                std::filesystem::exists(codes) ? uint32_t(std::filesystem::file_size(codes) / sizeof(float) / 8) : 0;
            config.appearanceSize = 8;
            if (untrained > 0.0f)
                config.appearances = 0;
            // The U-Net's levels as saved: the channels of every down block.
            std::vector<uint32_t> widths;
            for (uint32_t level = 0;; ++level) {
                const auto bias = parameters / ("generator.down." + std::to_string(level) + ".0.bias.bin");
                if (parameters.empty() || !std::filesystem::exists(bias))
                    break;
                widths.push_back(uint32_t(std::filesystem::file_size(bias) / sizeof(float)));
            }
            if (widths.size() >= 2)
                config.widths = widths;
            if (latentSize > 0) {
                config.appearances = 0;
                config.gbufferChannels = 4; // inverse depth and normals
                config.latentSize = latentSize;
                config.latentObjects = 1;
                config.latentToDecoder = neural::savedLatentToDecoder(parameters.string());
            }
            trained = std::make_unique<neural::PhotoEnhancerTraining>(vc, config);
            if (untrained > 0.0f) {
                std::mt19937 rng(1);
                for (const auto& name : trained->parameterNames())
                    trained->setParameter(name, crate::initialValue(*trained, name, rng, untrained));
            } else {
                trained->load(parameters.string());
            }
            if (config.appearances > 0) {
                std::vector<float> table(config.appearances * 8);
                std::ifstream(codes, std::ios::binary)
                    .read(reinterpret_cast<char*>(table.data()), std::streamsize(table.size() * sizeof(float)));
                std::vector<float> code(8, 0.0f);
                for (size_t i = 0; i < table.size(); ++i)
                    if (appearanceRow < 0)
                        code[i % 8] += table[i] / float(config.appearances);
                    else if (i / 8 == size_t(appearanceRow))
                        code[i % 8] = table[i];
                trained->setAppearance(code);
            }
        }
        // Qwen-Image's control depths and batch lists: <qwen-batch>/depth_smooth/<view>.ppm and
        // <qwen-batch>/batches/NN.txt, whose photos go to <qwen-batch>/photo/<view>.ppm.
        std::vector<std::string> batchLines;
        if (!qwenBatch.empty()) {
            std::filesystem::create_directories(qwenBatch / "depth_smooth");
            std::filesystem::create_directories(qwenBatch / "photo");
            std::filesystem::create_directories(qwenBatch / "batches");
            if (qwenImageToImage)
                std::filesystem::create_directories(qwenBatch / "input");
        }
        std::unique_ptr<neural::PhotoEnhancer> enhancer;
        if (!model.empty()) {
            enhancer = std::make_unique<neural::PhotoEnhancer>(vc, model, sampleSize, sampleSize);
            if (appearanceRow >= 0) {
                const uint32_t n = enhancer->appearanceSize();
                const auto& codes = enhancer->trainingCodes();
                if (n == 0 || size_t(appearanceRow + 1) * n > codes.size())
                    throw std::runtime_error("--appearance: the model has no such training code");
                enhancer->setAppearance(
                    std::vector<float>(codes.begin() + appearanceRow * n, codes.begin() + (appearanceRow + 1) * n));
                std::cout << "appearance of " << enhancer->trainingIds()[size_t(appearanceRow)] << std::endl;
            }
        }

        std::filesystem::create_directories(outDir / "frames");
        std::ofstream cameras(outDir / "cameras_classical.csv");
        cameras << "frame,eye_x,eye_y,eye_z,look_x,look_y,look_z,near_plane,fov_y_degrees,depth_near,depth_far\n";
        std::mt19937 rng(seed);
        // The two lights of the shadows G-buffer: given, or drawn per view (--random-lights).
        std::mt19937 lightRandom(lightSeed);
        const auto drawLight = [&]() {
            std::uniform_real_distribution<float> unit(0.0f, 1.0f);
            const float azimuth = 6.2831853f * unit(lightRandom),
                        elevation = glm::radians(15.0f + 55.0f * unit(lightRandom));
            GBufferLight light;
            light.towardsLight = glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                           std::cos(azimuth) * std::cos(elevation));
            light.angularRadius = 0.15f * unit(lightRandom);
            light.intensity = 0.5f + 0.7f * unit(lightRandom);
            return light;
        };
        std::ofstream lightLog;
        if (randomLights)
            lightLog.open(outDir / "lights.csv");
        if (lightLog)
            lightLog << "view,light,azimuth_degrees,elevation_degrees,angular_radius,intensity\n";
        if (!randomLights)
            renderer.setLights(lights[0], lights[1], 0.25f);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::normal_distribution<float> jitter(0.0f, 0.12f);
        const std::pair<const char*, GBufferOutput> outputs[] = {
            {"", GBufferOutput::Colour},          {"class_", GBufferOutput::Classes},
            {"depth_", GBufferOutput::Depth},     {"depth16_", GBufferOutput::Depth16},
            {"u_", GBufferOutput::CoordinatesU},  {"v_", GBufferOutput::CoordinatesV},
            {"normals_", GBufferOutput::Normals}, {"shadows_", GBufferOutput::Shadows},
            {"lit_", GBufferOutput::LitShadowed}};
        constexpr uint32_t kWarmUp = 5; // views before --profile and --time start timing
        double stepSeconds = 0.0;       // the latent model's steps after the warm-up
        for (uint32_t view = 0; view < views; ++view) {
            if (profile && trained && view == kWarmUp)
                trained->resetProfile();
            // Any azimuth, 8 to 40 degrees above the crate, 2.2 to 3.8 away, looking near its centre; on the orbit
            // once around at 22 degrees, 3 away; with a fixed view always from azimuth 0.6 at 22 degrees, 3 away.
            const bool steady = orbit || fixedView;
            const float azimuth = fixedView ? 0.6f
                                  : orbit   ? 6.2831853f * float(view) / float(views)
                                            : uniform(rng) * 6.2831853f;
            const float elevation = glm::radians(steady ? 22.0f : 8.0f + 32.0f * uniform(rng));
            const float distance = steady ? 3.0f : 2.2f + 1.6f * uniform(rng);
            const glm::vec3 look = glm::vec3(0.0f, 0.5f, 0.0f) +
                                   (steady ? glm::vec3(0.0f) : glm::vec3(jitter(rng), jitter(rng), jitter(rng)));
            glm::vec3 eye = look + distance * glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                                        std::cos(azimuth) * std::cos(elevation));
            eye.y = std::max(eye.y, 0.15f);
            CameraMVP& ubo = renderer.getCamera()->ubo;
            ubo.model = glm::mat4(1.0f);
            ubo.view = glm::lookAt(eye, look, glm::vec3(0.0f, 1.0f, 0.0f));
            ubo.proj = glm::perspective(glm::radians(crate::kFovDegrees), 1.0f, 0.05f, 100.0f);
            ubo.proj[1][1] *= -1.0f;
            ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
            renderer.getCamera()->update(0);
            if (randomLights) {
                const GBufferLight first = drawLight(), second = drawLight();
                renderer.setLights(first, second, 0.25f);
                for (int k = 0; k < 2; ++k) {
                    const GBufferLight& l = k == 0 ? first : second;
                    const glm::vec3 d = glm::normalize(l.towardsLight);
                    lightLog << view << "," << k + 1 << "," << glm::degrees(std::atan2(d.x, d.z)) << ","
                             << glm::degrees(std::asin(d.y)) << "," << l.angularRadius << "," << l.intensity << "\n";
                }
            }
            neural::GBufferFrame frame;
            frame.width = frame.height = size;
            frame.depthFar = crate::kDepthFar;
            std::vector<uint8_t>* buffers[] = {&frame.colour, &frame.classes, nullptr, &frame.depth16, &frame.u,
                                               &frame.v,      &frame.normals, nullptr, nullptr};
            for (size_t k = 0; k < std::size(outputs); ++k) {
                const auto& [kind, output] = outputs[k];
                renderer.setOutput(output, 0.0f, crate::kDepthFar);
                graph.submitAndWait(vc.getGraphicsQueue(), 0);
                const auto pixels = crate::readImage(vc, target->getImage(0), extent);
                char name[64];
                std::snprintf(name, sizeof(name), "classical_%s%04u.png", kind, view);
                writeImageBgra(outDir / "frames" / name, pixels, size, size);
                if (buffers[k])
                    *buffers[k] = pixels;
            }
            // The sample as the training saw it: the inverse depth stretched up to the nearest point.
            const auto inputs =
                neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{sampleSize, sampleSize, 1.0f});
            if (!qwenBatch.empty()) {
                char id[128];
                std::snprintf(id, sizeof(id), "%s_%04u", outDir.filename().string().c_str(), view);
                const auto control = qwenBatch / "depth_smooth" / (std::string(id) + ".ppm");
                writeControlDepth(control, inputs.channels, sampleSize);
                // For image to image the rendering, its background white, as Qwen-Image's starting image.
                std::string start = "-";
                if (qwenImageToImage) {
                    const auto path = qwenBatch / "input" / (std::string(id) + ".ppm");
                    writeOnWhite(path, inputs, sampleSize);
                    start = path.string();
                }
                batchLines.push_back(start + " " + (qwenBatch / "photo" / (std::string(id) + ".ppm")).string() + " " +
                                     control.string());
            }
            if (enhancer || trained) {
                std::vector<float> image;
                if (trained && latentSize > 0) {
                    // As crate_training_example --latent sees a view: inverse depth and normals, the crate as the
                    // one object, its latent given.
                    const size_t pixels = size_t(sampleSize) * sampleSize;
                    std::vector<float> geometry(inputs.channels.begin() + 3 * pixels,
                                                inputs.channels.begin() + 4 * pixels);
                    geometry.insert(geometry.end(), inputs.normals.begin(), inputs.normals.end());
                    std::vector<float> objects(pixels);
                    for (size_t q = 0; q < pixels; ++q)
                        objects[q] = inputs.classes[q] == float(tree::TREE_CLASS_STEM) ? 1.0f : 0.0f;
                    trained->setBatch(geometry, inputs.coordinates, inputs.classes, {}, {});
                    trained->setObjects(objects);
                    trained->setLatents(viewLatents.empty() ? latentCode : viewLatents[view % viewLatents.size()]);
                    const auto started = std::chrono::steady_clock::now();
                    trained->step();
                    if (view >= kWarmUp)
                        stepSeconds +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                    image = trained->image();
                } else if (trained) {
                    trained->setBatch(inputs.channels, inputs.coordinates, inputs.classes, {}, {0.0f});
                    trained->step();
                    image = trained->image();
                    if (features)
                        writeFeatures(outDir / "frames", view, trained->generatorInput(), sampleSize);
                } else {
                    image = enhancer->run(inputs);
                }
                const size_t pixels = size_t(sampleSize) * sampleSize;
                std::vector<uint8_t> rgb(3 * pixels);
                for (size_t p = 0; p < pixels; ++p)
                    for (size_t c = 0; c < 3; ++c)
                        rgb[3 * p + c] = uint8_t(std::lround(std::clamp(image[c * pixels + p], 0.0f, 1.0f) * 255.0f));
                char name[64];
                std::snprintf(name, sizeof(name), "enhanced_%04u.png", view);
                writeImage(outDir / "frames" / name, rgb.data(), sampleSize, sampleSize, 3);
                // An albedo model's albedo too.
                const auto albedo = trained ? trained->albedo() : std::vector<float>{};
                if (!albedo.empty()) {
                    for (size_t p = 0; p < pixels; ++p)
                        for (size_t c = 0; c < 3; ++c)
                            rgb[3 * p + c] =
                                uint8_t(std::lround(std::clamp(albedo[c * pixels + p], 0.0f, 1.0f) * 255.0f));
                    std::snprintf(name, sizeof(name), "albedo_%04u.png", view);
                    writeImage(outDir / "frames" / name, rgb.data(), sampleSize, sampleSize, 3);
                }
            }
            cameras << view << "," << eye.x << "," << eye.y << "," << eye.z << "," << look.x << "," << look.y << ","
                    << look.z << ",0.05," << crate::kFovDegrees << ",0," << crate::kDepthFar << "\n";
        }
        // Batches of 20 views for qwen_image_example --batch.
        for (size_t first = 0; first < batchLines.size(); first += 20) {
            char name[32];
            std::snprintf(name, sizeof(name), "%02zu.txt", first / 20);
            std::ofstream batch(qwenBatch / "batches" / name);
            for (size_t k = first; k < std::min(first + 20, batchLines.size()); ++k)
                batch << batchLines[k] << "\n";
        }
        if (timeSteps && trained && views > kWarmUp)
            std::printf("model step %.3f ms per view (wall clock, upload to finish, %u views)\n",
                        1e3 * stepSeconds / double(views - kWarmUp), views - kWarmUp);
        // The trained model's GPU time per view: every element in execution order, then per kind.
        if (profile && trained && views > kWarmUp) {
            const auto elements = trained->profileElements();
            double total = 0.0;
            std::cout << "GPU time per view over " << views - kWarmUp << " views:" << std::endl;
            for (size_t i = 0; i < elements.size(); ++i) {
                std::printf("  %3zu %-28s %8.4f ms\n", i, elements[i].first.c_str(), elements[i].second);
                total += elements[i].second;
            }
            std::printf("elements %.3f ms, GPU span %.3f ms\n", total, trained->gpuSpan());
            for (const auto& [kind, milliseconds] : trained->profile())
                std::printf("  %-28s %8.4f ms\n", kind.c_str(), milliseconds);
        }
        std::cout << views << " views in " << outDir << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "crate_gbuffer_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
