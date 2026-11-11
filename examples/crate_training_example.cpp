// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Trains the crate's photo enhancer entirely in Klartraum (neural::PhotoEnhancerTraining): the views of the crate
// flights of crate_gbuffer_example are rendered again from their cameras (GBufferMeshRenderer) and prepared as the
// photo enhancer's samples (neural::photoEnhancerInputs()), the Qwen-Image photos of the views are the targets (those
// that follow their rendering, neural::PhotoAlignment), and
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
#include "klartraum/neural/photo_alignment.hpp"
#include "klartraum/neural/photo_enhancer_training.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kChannels = 11;
// The samples' size (--sample-size); the G-buffers are rendered at twice that and reduced as in training.
uint32_t sampleSize = 256;

// One view: its sample (G-buffer channels, coordinates, classes) and its photo [3][size][size] in [0, 1].
struct View {
    std::string id;
    neural::PhotoEnhancerInputs inputs;
    std::vector<float> photo;
};

// A binary PPM of the sample size, or a whole multiple of it averaged down (e.g. 256 x 256 photos for 128 x 128
// samples), as [3][size][size] in [0, 1]; empty if missing or of another size.
std::vector<float> readPhoto(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    uint32_t width = 0, height = 0, maximum = 0;
    if (!(file >> magic >> width >> height >> maximum) || magic != "P6" || width != height || width % sampleSize != 0)
        return {};
    file.get();
    std::vector<uint8_t> bytes(size_t(width) * height * 3);
    file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    const uint32_t factor = width / sampleSize;
    const size_t pixels = size_t(sampleSize) * sampleSize;
    std::vector<float> photo(3 * pixels, 0.0f);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            for (size_t c = 0; c < 3; ++c)
                photo[c * pixels + size_t(y / factor) * sampleSize + x / factor] +=
                    float(bytes[(size_t(y) * width + x) * 3 + c]) / (255.0f * float(factor * factor));
    return photo;
}

// Scores by view id from a cache file of lines "<id> <score>" (missing: empty).
std::map<std::string, float> readScores(const std::filesystem::path& path) {
    std::map<std::string, float> scores;
    std::ifstream file(path);
    std::string id;
    for (float score; file >> id >> score;)
        scores[id] = score;
    return scores;
}

// The views of a flight (crate_gbuffer_example's cameras_classical.csv) that have a photo, show mostly the scene (at
// most 60 % background, as train_photo_enhancer.usable()) and whose photo follows the rendering well enough
// (neural::PhotoAlignment; the scores are kept in <split>/alignment_klartraum.txt).
std::vector<View> loadViews(VulkanContext& vc, GBufferMeshRenderer& renderer, ComputeGraph& graph,
                            OffscreenTarget& target, neural::PhotoAlignment& alignment,
                            const std::filesystem::path& flight, const std::filesystem::path& split,
                            float minAlignment) {
    const auto cache = split / "alignment_klartraum.txt";
    auto scores = readScores(cache);
    std::ofstream newScores(cache, std::ios::app);
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
        if (view.photo.empty())
            continue;
        crate::setCamera(renderer.getCamera()->ubo, glm::vec3(v[1], v[2], v[3]), glm::vec3(v[4], v[5], v[6]));
        renderer.getCamera()->update(0);
        const auto frame = crate::renderFrame(vc, renderer, graph, target, 2 * sampleSize);
        view.inputs = neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{sampleSize, sampleSize, 1.0f});
        const auto background = std::count(view.inputs.classes.begin(), view.inputs.classes.end(), 0.0f);
        if (double(background) > 0.6 * double(view.inputs.classes.size()))
            continue;
        if (!scores.count(view.id)) {
            const size_t pixels = size_t(sampleSize) * sampleSize;
            const std::vector<float> depth(view.inputs.channels.begin() + 3 * pixels,
                                           view.inputs.channels.begin() + 4 * pixels);
            scores[view.id] = alignment.score(view.photo, depth, sampleSize, sampleSize);
            newScores << view.id << " " << scores[view.id] << "\n";
        }
        if (scores[view.id] < minAlignment)
            continue;
        views.push_back(std::move(view));
    }
    return views;
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
        const bool half = uniform(rng) < 0.3f && sampleSize / 2 >= crop;
        const uint32_t size = half ? sampleSize / 2 : sampleSize, step = half ? 2 : 1;
        const uint32_t top = uint32_t(rng() % (size - crop + 1)), left = uint32_t(rng() % (size - crop + 1));
        const bool flip = uniform(rng) < 0.5f;
        const size_t pixels = size_t(sampleSize) * sampleSize;
        // A value of a channel of the view at crop pixel (y, x): the mean of its step x step block for averaged
        // planes, its top-left pixel for sampled ones.
        const auto at = [&](const std::vector<float>& plane, size_t channel, uint32_t y, uint32_t x, bool average) {
            const uint32_t sy = (top + y) * step, sx = (left + (flip ? crop - 1 - x : x)) * step;
            if (!average || step == 1)
                return plane[channel * pixels + size_t(sy) * sampleSize + sx];
            float sum = 0.0f;
            for (uint32_t dy = 0; dy < step; ++dy)
                for (uint32_t dx = 0; dx < step; ++dx)
                    sum += plane[channel * pixels + size_t(sy + dy) * sampleSize + sx + dx];
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
    float minAlignment = 0.7f, perceptual = 0.0f, adversarial = 0.0f, featureMatching = 2.0f;
    bool profile = false;
    std::filesystem::path initFrom;
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
        else if (arg == "--sample-size" && hasValue)
            sampleSize = uint32_t(std::stoul(argv[++i]));
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
        else if (arg == "--perceptual" && hasValue)
            perceptual = std::stof(argv[++i]);
        else if (arg == "--adversarial" && hasValue)
            adversarial = std::stof(argv[++i]);
        else if (arg == "--feature-matching" && hasValue)
            featureMatching = std::stof(argv[++i]);
        else if (arg == "--init-from" && hasValue)
            initFrom = argv[++i];
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
            const VkExtent2D extent{2 * sampleSize, 2 * sampleSize};
            auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
            GBufferMeshRenderer renderer(vc, crate::vertices(), target, extent);
            ComputeGraph graph(vc, 1);
            graph.compileFrom(renderer.getOutput());
            neural::PhotoAlignment alignment(vc);
            training = loadViews(vc, renderer, graph, *target, alignment, crateDir / "flights/crate_a",
                                 crateDir / "data/train", minAlignment);
            validation = loadViews(vc, renderer, graph, *target, alignment, crateDir / "flights/crate_v",
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
        config.texture = crate::textureShape();
        config.appearances = appearance > 0 ? uint32_t(training.size()) : 0;
        config.appearanceSize = appearance;
        config.profile = profile;
        config.perceptual = perceptual;
        config.adversarial = adversarial;
        config.featureMatching = featureMatching;
        neural::PhotoEnhancerTraining trainer(vc, config);
        std::mt19937 rng(1);
        for (const auto& name : trainer.parameterNames()) {
            auto values = crate::initialValue(trainer, name, rng);
            // The generator, texture and codes from an earlier run (its saved averages), if given.
            const auto saved = initFrom / (name + ".bin");
            if (!initFrom.empty() && std::filesystem::exists(saved) && name.rfind("discriminator.", 0) != 0) {
                std::ifstream file(saved, std::ios::binary);
                file.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
            }
            trainer.setParameter(name, values);
        }

        // Validation: the held-out views at the full sample size, with the codes' mean.
        auto renderConfig = config;
        renderConfig.forwardOnly = true;
        constexpr uint32_t kRenderBatch = 4; // held-out views rendered at a time
        renderConfig.batch = kRenderBatch;
        renderConfig.size = sampleSize;
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
        double lossSum = 0.0, perceptualSum = 0.0, discriminatorSum = 0.0, adversarialSum = 0.0, matchingSum = 0.0;
        auto started = std::chrono::steady_clock::now();
        for (uint32_t step = 1; step <= steps; ++step) {
            const Batch batch = sampleBatch(training, batchSize, crop, rng);
            trainer.setBatch(batch.channels, batch.coordinates, batch.classes, batch.photos, batch.rows);
            trainer.step();
            lossSum += trainer.loss();
            perceptualSum += trainer.perceptualLoss();
            discriminatorSum += trainer.discriminatorLoss();
            adversarialSum += trainer.adversarialLoss();
            matchingSum += trainer.featureMatchingLoss();
            if (step % 100 == 0) {
                const double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() / 100.0;
                std::ostringstream line;
                line << "step " << step << " loss " << lossSum / 100.0;
                if (perceptual > 0.0f)
                    line << " vgg " << perceptualSum / 100.0;
                if (adversarial > 0.0f)
                    line << " d " << discriminatorSum / 100.0 << " adv " << adversarialSum / 100.0 << " fm "
                         << matchingSum / 100.0;
                line << " (" << seconds << " s/step)";
                perceptualSum = discriminatorSum = adversarialSum = matchingSum = 0.0;
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
                const size_t pixels = size_t(sampleSize) * sampleSize;
                if (config.appearances > 0) {
                    // The first training views with their own codes (training_NNNNNN.png: photo | enhanced): how well
                    // the codes and the texture reproduce the photos they were fitted to.
                    Batch own;
                    for (uint32_t k = 0; k < kRenderBatch; ++k) {
                        const View& view = training[k % training.size()];
                        own.channels.insert(own.channels.end(), view.inputs.channels.begin(),
                                            view.inputs.channels.end());
                        own.coordinates.insert(own.coordinates.end(), view.inputs.coordinates.begin(),
                                               view.inputs.coordinates.end());
                        own.classes.insert(own.classes.end(), view.inputs.classes.begin(), view.inputs.classes.end());
                        own.rows.push_back(float(k % training.size()));
                    }
                    renderer->setBatch(own.channels, own.coordinates, own.classes, {}, own.rows);
                    renderer->step();
                    const auto reproduced = renderer->image();
                    const uint32_t shown = std::min<uint32_t>(kRenderBatch, uint32_t(training.size()));
                    const uint32_t width = 2 * sampleSize;
                    std::vector<uint8_t> grid(size_t(width) * sampleSize * shown * 3);
                    double l1 = 0.0;
                    for (size_t v = 0; v < shown; ++v)
                        for (size_t p = 0; p < pixels; ++p)
                            for (size_t c = 0; c < 3; ++c) {
                                const size_t y = v * sampleSize + p / sampleSize, x = p % sampleSize;
                                const float photo = training[v].photo[c * pixels + p];
                                const float enhanced = reproduced[(v * 3 + c) * pixels + p];
                                l1 += std::abs(photo - enhanced);
                                grid[(y * width + x) * 3 + c] = byte(photo);
                                grid[(y * width + sampleSize + x) * 3 + c] = byte(enhanced);
                            }
                    std::ostringstream line;
                    line << "training step " << step << ": L1 " << l1 / double(shown * 3 * pixels) << " over " << shown
                         << " views (own codes)";
                    std::cout << line.str() << std::endl;
                    log << line.str() << "\n";
                    char name[64];
                    std::snprintf(name, sizeof(name), "training_%06u.png", step);
                    writeImage(outDir / name, grid.data(), width, sampleSize * shown, 3);
                    renderer->setAppearance(trainer.meanCode());
                }
                // In chunks of kRenderBatch views; the last chunk repeats the first views to fill it.
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
                const uint32_t width = 3 * sampleSize;
                std::vector<uint8_t> grid(size_t(width) * sampleSize * shown * 3);
                for (size_t v = 0; v < shown; ++v)
                    for (size_t p = 0; p < pixels; ++p)
                        for (size_t c = 0; c < 3; ++c) {
                            const size_t y = v * sampleSize + p / sampleSize, x = p % sampleSize;
                            const auto put = [&](size_t column, float value) {
                                grid[(y * width + column * sampleSize + x) * 3 + c] = byte(value);
                            };
                            put(0, validation[v].inputs.channels[c * pixels + p]);
                            put(1, validation[v].photo[c * pixels + p]);
                            put(2, image[(v * 3 + c) * pixels + p]);
                        }
                char name[64];
                std::snprintf(name, sizeof(name), "validation_%06u.png", step);
                writeImage(outDir / name, grid.data(), width, uint32_t(sampleSize * shown), 3);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "crate_training_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
