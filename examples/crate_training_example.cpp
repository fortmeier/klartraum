// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Trains the crate's photo enhancer entirely in Klartraum (neural::PhotoEnhancerTraining): the views of the crate
// flights of crate_gbuffer_example are rendered again from their cameras (GBufferMeshRenderer) and prepared as the
// photo enhancer's samples (neural::photoEnhancerInputs()), the Qwen-Image photos of the views are the targets, and
// the learnable surface texture, the local U-Net generator and the views' appearance codes are trained with the
// photometric losses on random crops. Every --every steps the averaged parameters are saved and the held-out views
// are rendered with the codes' mean (validation_NNNNNN.png: rendering | photo | enhanced, and the L1 in log.txt).
//
//   crate_training_example --crate build/TestingOutput/crate --out build/TestingOutput/crate/runs/k1

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "crate_scene.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/image_io.hpp"
#include "klartraum/neural/photo_enhancer_training.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kRenderSize = 512, kSampleSize = 256, kChannels = 11;

// One view: its sample (G-buffer channels, coordinates, classes) and its photo [3][size][size] in [0, 1].
struct View {
    std::string id;
    neural::PhotoEnhancerInputs inputs;
    std::vector<float> photo;
};

// A binary PPM of the sample size as [3][height][width] in [0, 1]; empty if missing or of another size.
std::vector<float> readPhoto(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    uint32_t width = 0, height = 0, maximum = 0;
    if (!(file >> magic >> width >> height >> maximum) || magic != "P6" || width != kSampleSize ||
        height != kSampleSize)
        return {};
    file.get();
    std::vector<uint8_t> bytes(size_t(width) * height * 3);
    file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    const size_t pixels = size_t(width) * height;
    std::vector<float> photo(3 * pixels);
    for (size_t p = 0; p < pixels; ++p)
        for (size_t c = 0; c < 3; ++c)
            photo[c * pixels + p] = float(bytes[3 * p + c]) / 255.0f;
    return photo;
}

// {sample: score} of an alignment.json written by scripts/neural/check_photo_alignment.py.
std::map<std::string, float> readAlignment(const std::filesystem::path& path) {
    std::map<std::string, float> scores;
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    const std::string s = text.str();
    for (size_t at = s.find('"'); at != std::string::npos; at = s.find('"', at)) {
        const size_t end = s.find('"', at + 1);
        const size_t colon = s.find(':', end);
        if (end == std::string::npos || colon == std::string::npos)
            break;
        scores[s.substr(at + 1, end - at - 1)] = std::stof(s.substr(colon + 1));
        at = s.find_first_of(",}", colon);
    }
    return scores;
}

// The views of a flight (crate_gbuffer_example's cameras_classical.csv) that have a photo, are aligned well enough
// and show mostly the scene (at most 60 % background, as train_photo_enhancer.usable()).
std::vector<View> loadViews(VulkanContext& vc, GBufferMeshRenderer& renderer, ComputeGraph& graph,
                            OffscreenTarget& target, const std::filesystem::path& flight,
                            const std::filesystem::path& split, float minAlignment) {
    const auto alignment = readAlignment(split / "alignment.json");
    std::ifstream cameras(flight / "cameras_classical.csv");
    std::string line;
    std::getline(cameras, line);
    std::vector<View> views;
    while (std::getline(cameras, line)) {
        std::vector<float> v;
        std::stringstream fields(line);
        for (std::string field; std::getline(fields, field, ',');)
            v.push_back(std::stof(field));
        if (v.size() < 7)
            continue;
        char id[128];
        std::snprintf(id, sizeof(id), "%s_%04d", flight.filename().string().c_str(), int(v[0]));
        View view;
        view.id = id;
        view.photo = readPhoto(split / "photo" / (view.id + ".ppm"));
        const auto score = alignment.find(view.id);
        if (view.photo.empty() || (score != alignment.end() && score->second < minAlignment))
            continue;
        crate::setCamera(renderer.getCamera()->ubo, glm::vec3(v[1], v[2], v[3]), glm::vec3(v[4], v[5], v[6]));
        renderer.getCamera()->update(0);
        const auto frame = crate::renderFrame(vc, renderer, graph, target, kRenderSize);
        view.inputs = neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{kSampleSize, kSampleSize, 1.0f});
        const auto background = std::count(view.inputs.classes.begin(), view.inputs.classes.end(), 0.0f);
        if (double(background) > 0.6 * double(view.inputs.classes.size()))
            continue;
        views.push_back(std::move(view));
    }
    return views;
}

// The surface texture of train_photo_enhancer.py: hash_texture.SurfaceTexture's defaults.
neural::HashTextureShape textureShape() {
    neural::HashTextureShape shape;
    shape.levels = 12;
    shape.features = 2;
    shape.tableSize = 1u << 17;
    shape.classes = {2, 3, 4};
    const double growth = std::exp((std::log(2048.0) - std::log(16.0)) / 11.0);
    for (uint32_t level = 0; level < shape.levels; ++level)
        shape.resolutions.push_back(uint32_t(std::floor(16.0 * std::pow(growth, double(level)))));
    return shape;
}

// The batch of random crops (as train_photo_enhancer.crops()): a view, at full or (30 %) half scale (area mean of
// channels and photo, every second coordinate and class), a random window, flipped half the time.
struct Batch {
    std::vector<float> channels, coordinates, classes, photos, rows;
};

Batch sampleBatch(const std::vector<View>& views, uint32_t count, uint32_t crop, std::mt19937& rng) {
    Batch batch;
    const size_t cropPixels = size_t(crop) * crop;
    batch.channels.resize(count * kChannels * cropPixels);
    batch.coordinates.resize(count * 3 * cropPixels);
    batch.classes.resize(count * cropPixels);
    batch.photos.resize(count * 3 * cropPixels);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (uint32_t k = 0; k < count; ++k) {
        const uint32_t index = uint32_t(rng() % views.size());
        const View& view = views[index];
        batch.rows.push_back(float(index));
        const bool half = uniform(rng) < 0.3f && kSampleSize / 2 >= crop;
        const uint32_t size = half ? kSampleSize / 2 : kSampleSize, step = half ? 2 : 1;
        const uint32_t top = uint32_t(rng() % (size - crop + 1)), left = uint32_t(rng() % (size - crop + 1));
        const bool flip = uniform(rng) < 0.5f;
        const size_t pixels = size_t(kSampleSize) * kSampleSize;
        // A value of a channel of the view at crop pixel (y, x): the mean of its step x step block for averaged
        // planes, its top-left pixel for sampled ones.
        const auto at = [&](const std::vector<float>& plane, size_t channel, uint32_t y, uint32_t x, bool average) {
            const uint32_t sy = (top + y) * step, sx = (left + (flip ? crop - 1 - x : x)) * step;
            if (!average || step == 1)
                return plane[channel * pixels + size_t(sy) * kSampleSize + sx];
            float sum = 0.0f;
            for (uint32_t dy = 0; dy < step; ++dy)
                for (uint32_t dx = 0; dx < step; ++dx)
                    sum += plane[channel * pixels + size_t(sy + dy) * kSampleSize + sx + dx];
            return sum / float(step * step);
        };
        for (uint32_t y = 0; y < crop; ++y)
            for (uint32_t x = 0; x < crop; ++x) {
                const size_t p = size_t(y) * crop + x;
                for (size_t c = 0; c < kChannels; ++c)
                    batch.channels[(k * kChannels + c) * cropPixels + p] = at(view.inputs.channels, c, y, x, true);
                for (size_t c = 0; c < 3; ++c) {
                    batch.coordinates[(k * 3 + c) * cropPixels + p] = at(view.inputs.coordinates, c, y, x, false);
                    batch.photos[(k * 3 + c) * cropPixels + p] = at(view.photo, c, y, x, true);
                }
                batch.classes[k * cropPixels + p] = at(view.inputs.classes, 0, y, x, false);
            }
    }
    return batch;
}

uint8_t byte(float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path crateDir = "build/TestingOutput/crate", outDir = "build/TestingOutput/crate/runs/k1";
    uint32_t steps = 30000, batchSize = 16, crop = 128, appearance = 8, every = 1000;
    float minAlignment = 0.7f;
    bool profile = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--crate" && hasValue)
            crateDir = argv[++i];
        else if (arg == "--out" && hasValue)
            outDir = argv[++i];
        else if (arg == "--steps" && hasValue)
            steps = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--batch" && hasValue)
            batchSize = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--crop" && hasValue)
            crop = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--appearance" && hasValue)
            appearance = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--every" && hasValue)
            every = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--min-alignment" && hasValue)
            minAlignment = std::stof(argv[++i]);
        else if (arg == "--profile")
            profile = true;
        else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    try {
        HeadlessFrontend frontend;
        VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
        std::vector<View> training, validation;
        {
            const VkExtent2D extent{kRenderSize, kRenderSize};
            auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
            GBufferMeshRenderer renderer(vc, crate::vertices(), target, extent);
            ComputeGraph graph(vc, 1);
            graph.compileFrom(renderer.getOutput());
            training = loadViews(vc, renderer, graph, *target, crateDir / "flights/crate_a", crateDir / "data/train",
                                 minAlignment);
            validation = loadViews(vc, renderer, graph, *target, crateDir / "flights/crate_v",
                                   crateDir / "data/validation", minAlignment);
        }
        std::cout << training.size() << " training views, " << validation.size() << " validation views" << std::endl;
        if (training.empty())
            throw std::runtime_error("no training views with photos");
        std::filesystem::create_directories(outDir);
        {
            std::ofstream ids(outDir / "appearance_ids.txt");
            for (const auto& view : training)
                ids << view.id << "\n";
        }

        neural::PhotoEnhancerTrainingConfig config;
        config.batch = batchSize;
        config.size = crop;
        config.texture = textureShape();
        config.appearances = appearance > 0 ? uint32_t(training.size()) : 0;
        config.appearanceSize = appearance;
        config.profile = profile;
        neural::PhotoEnhancerTraining trainer(vc, config);
        // nn.Linear and nn.Conv2d's default initialisation (uniform within 1 / sqrt(fan in)), ChannelNorm's ones and
        // zeros, the texture's +-1e-4, the codes' N(0, 0.1); FiLM's heads and the output layer start at zero.
        std::mt19937 rng(1);
        for (const auto& name : trainer.parameterNames()) {
            const auto dims = trainer.parameter(name)->getDimensions();
            size_t count = 1;
            for (uint32_t d : dims)
                count *= d;
            std::vector<float> values(count, 0.0f);
            const bool bias = name.size() > 5 && name.substr(name.size() - 5) == ".bias";
            const bool norm = name.find(".1.") != std::string::npos || name.find(".4.") != std::string::npos;
            const bool zeroStart = name.find("film.gammas") != std::string::npos ||
                                   name.find("film.betas") != std::string::npos ||
                                   name.find(".out.") != std::string::npos;
            if (name == "texture.table") {
                std::uniform_real_distribution<float> tiny(-1e-4f, 1e-4f);
                for (auto& v : values)
                    v = tiny(rng);
            } else if (name == "codes.weight") {
                std::normal_distribution<float> normal(0.0f, 0.1f);
                for (auto& v : values)
                    v = normal(rng);
            } else if (zeroStart) {
            } else if (norm) {
                if (!bias)
                    std::fill(values.begin(), values.end(), 1.0f);
            } else {
                // The fan in of the layer the parameter belongs to (the weight's dimensions after the first).
                const auto weightDims =
                    trainer.parameter(bias ? name.substr(0, name.size() - 5) + ".weight" : name)->getDimensions();
                uint32_t fanIn = 1;
                for (size_t k = 1; k < weightDims.size(); ++k)
                    fanIn *= weightDims[k];
                std::uniform_real_distribution<float> uniform(-1.0f / std::sqrt(float(fanIn)),
                                                              1.0f / std::sqrt(float(fanIn)));
                for (auto& v : values)
                    v = uniform(rng);
            }
            trainer.setParameter(name, values);
        }

        // Validation: the held-out views at the full sample size, with the codes' mean.
        auto renderConfig = config;
        renderConfig.forwardOnly = true;
        constexpr uint32_t kRenderBatch = 4; // held-out views rendered at a time
        renderConfig.batch = kRenderBatch;
        renderConfig.size = kSampleSize;
        std::unique_ptr<neural::PhotoEnhancerTraining> renderer;
        if (!validation.empty())
            renderer = std::make_unique<neural::PhotoEnhancerTraining>(vc, renderConfig);
        Batch held;
        for (const auto& view : validation) {
            held.channels.insert(held.channels.end(), view.inputs.channels.begin(), view.inputs.channels.end());
            held.coordinates.insert(held.coordinates.end(), view.inputs.coordinates.begin(),
                                    view.inputs.coordinates.end());
            held.classes.insert(held.classes.end(), view.inputs.classes.begin(), view.inputs.classes.end());
            held.photos.insert(held.photos.end(), view.photo.begin(), view.photo.end());
            held.rows.push_back(0.0f);
        }

        std::ofstream log(outDir / "log.txt", std::ios::app);
        double lossSum = 0.0;
        auto started = std::chrono::steady_clock::now();
        for (uint32_t step = 1; step <= steps; ++step) {
            const Batch batch = sampleBatch(training, batchSize, crop, rng);
            trainer.setBatch(batch.channels, batch.coordinates, batch.classes, batch.photos, batch.rows);
            trainer.step();
            lossSum += trainer.loss();
            if (step % 100 == 0) {
                const double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() / 100.0;
                std::ostringstream line;
                line << "step " << step << " loss " << lossSum / 100.0 << " (" << seconds << " s/step)";
                std::cout << line.str() << std::endl;
                log << line.str() << "\n";
                log.flush();
                lossSum = 0.0;
                if (profile) {
                    const auto kinds = trainer.profile();
                    double total = 0.0;
                    for (const auto& [kind, milliseconds] : kinds)
                        total += milliseconds;
                    std::cout << "GPU time per step " << total << " ms:";
                    for (const auto& [kind, milliseconds] : kinds)
                        if (milliseconds > 0.01 * total)
                            std::cout << " " << kind << " " << milliseconds;
                    std::cout << std::endl;
                }
                started = std::chrono::steady_clock::now();
            }
            if (step % every == 0 || step == steps) {
                trainer.save((outDir / "parameters").string());
                if (!renderer)
                    continue;
                renderer->load((outDir / "parameters").string());
                if (config.appearances > 0)
                    renderer->setAppearance(trainer.meanCode());
                // In chunks of kRenderBatch views; the last chunk repeats the first views to fill it.
                const size_t pixels = size_t(kSampleSize) * kSampleSize;
                std::vector<float> image;
                for (size_t first = 0; first < validation.size(); first += kRenderBatch) {
                    Batch chunk;
                    for (uint32_t k = 0; k < kRenderBatch; ++k) {
                        const size_t v = (first + k) % validation.size();
                        const auto take = [&](const std::vector<float>& from, std::vector<float>& to, size_t size) {
                            to.insert(to.end(), from.begin() + v * size, from.begin() + (v + 1) * size);
                        };
                        take(held.channels, chunk.channels, kChannels * pixels);
                        take(held.coordinates, chunk.coordinates, 3 * pixels);
                        take(held.classes, chunk.classes, pixels);
                        chunk.rows.push_back(0.0f);
                    }
                    renderer->setBatch(chunk.channels, chunk.coordinates, chunk.classes, {}, chunk.rows);
                    renderer->step();
                    const auto part = renderer->image();
                    const size_t count = std::min<size_t>(kRenderBatch, validation.size() - first);
                    image.insert(image.end(), part.begin(), part.begin() + count * 3 * pixels);
                }
                double l1 = 0.0;
                for (size_t i = 0; i < image.size(); ++i)
                    l1 += std::abs(image[i] - held.photos[i]);
                l1 /= double(image.size());
                std::ostringstream line;
                line << "validation step " << step << ": L1 " << l1 << " over " << validation.size()
                     << " views (mean code)";
                std::cout << line.str() << std::endl;
                log << line.str() << "\n";
                log.flush();
                // The first six held-out views: rendering | photo | enhanced.
                const size_t shown = std::min<size_t>(6, validation.size());
                const uint32_t width = 3 * kSampleSize;
                std::vector<uint8_t> grid(size_t(width) * kSampleSize * shown * 3);
                for (size_t v = 0; v < shown; ++v)
                    for (size_t p = 0; p < pixels; ++p)
                        for (size_t c = 0; c < 3; ++c) {
                            const size_t y = v * kSampleSize + p / kSampleSize, x = p % kSampleSize;
                            const auto put = [&](size_t column, float value) {
                                grid[(y * width + column * kSampleSize + x) * 3 + c] = byte(value);
                            };
                            put(0, validation[v].inputs.channels[c * pixels + p]);
                            put(1, validation[v].photo[c * pixels + p]);
                            put(2, image[(v * 3 + c) * pixels + p]);
                        }
                char name[64];
                std::snprintf(name, sizeof(name), "validation_%06u.png", step);
                writeImage(outDir / name, grid.data(), width, uint32_t(kSampleSize * shown), 3);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "crate_training_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
