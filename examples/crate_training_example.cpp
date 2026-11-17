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
//
// With --crates DIR many crates train together (crate_paint_example's crates/<id>, six painted views each): every
// crate has its own row of a learned latent table (no encoder), --texture-only-latent keeps the latents out of the
// generator so that only the texture tells the crates apart; the last --held-out-crates crates are left out and their
// codes fitted afterwards on three of their views (--fit-steps), the other three measuring how well a new crate is
// reproduced.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
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

// The geometry channels each sample uploads: the photo enhancer's 11 (RGB, depth, classes), or for the latent
// renderer 4 (inverse depth and normals).
uint32_t kChannels = 11;
// Whether crops are flipped (not for the latent renderer, whose normals would not flip with them).
bool flips = true;
// The samples' size (--sample-size); the G-buffers are rendered at twice that and reduced as in training.
uint32_t sampleSize = 256;

// One view: its sample (G-buffer channels, coordinates, classes) and its photo [3][size][size] in [0, 1].
struct View {
    std::string id;
    glm::vec3 eye, look; ///< the camera
    neural::PhotoEnhancerInputs inputs;
    std::vector<float> photo;
    std::vector<float> objects; ///< for the latent renderer: 1 on the crate, 0 elsewhere
    /// For the temporal loss: a nearby second view's sample and objects, and the flow of its pixels onto this view
    /// ([2][size][size], crate::Flow's x and y) with where this view sees them ([size][size]).
    neural::PhotoEnhancerInputs secondInputs;
    std::vector<float> secondObjects, flow, flowValid;
    int code = -1; ///< with --crates: the row of the view's crate in the latent table
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
// most `maxBackground`, e.g. 60 % as train_photo_enhancer.usable()) and whose photo follows the rendering well enough
// (neural::PhotoAlignment; the scores are kept in <split>/alignment_klartraum.txt).
std::vector<View> loadViews(VulkanContext& vc, GBufferMeshRenderer& renderer, ComputeGraph& graph,
                            OffscreenTarget& target, neural::PhotoAlignment& alignment,
                            const std::filesystem::path& flight, const std::filesystem::path& split, float minAlignment,
                            float maxBackground) {
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
        view.eye = glm::vec3(v[1], v[2], v[3]);
        view.look = glm::vec3(v[4], v[5], v[6]);
        crate::setCamera(renderer.getCamera()->ubo, view.eye, view.look);
        renderer.getCamera()->update(0);
        const auto frame = crate::renderFrame(vc, renderer, graph, target, 2 * sampleSize);
        view.inputs = neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{sampleSize, sampleSize, 1.0f});
        const auto background = std::count(view.inputs.classes.begin(), view.inputs.classes.end(), 0.0f);
        if (double(background) > double(maxBackground) * double(view.inputs.classes.size()))
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
// A frame of a flight of tree_bonsai_example --fly-in --gbuffers: its camera, depth window and G-buffers.
struct FlightFrame {
    int index = 0;
    glm::vec3 eye, look;
    float depthFar = 1.0f;
    neural::GBufferFrame gbuffers;
};

// An RGB binary PPM as BGRA bytes (as a readback gives them); empty if missing.
std::vector<uint8_t> readBgra(const std::filesystem::path& path, uint32_t& width, uint32_t& height) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    uint32_t maximum = 0;
    if (!(file >> magic >> width >> height >> maximum) || magic != "P6")
        return {};
    file.get();
    std::vector<uint8_t> rgb(size_t(width) * height * 3);
    file.read(reinterpret_cast<char*>(rgb.data()), std::streamsize(rgb.size()));
    std::vector<uint8_t> bgra(size_t(width) * height * 4, 255);
    for (size_t p = 0; p < size_t(width) * height; ++p)
        for (size_t c = 0; c < 3; ++c)
            bgra[4 * p + 2 - c] = rgb[3 * p + c];
    return bgra;
}

// The frames of a flight from frame `first` on.
std::vector<FlightFrame> readFlightFrames(const std::filesystem::path& flight, int first) {
    std::ifstream cameras(flight / "cameras_classical.csv");
    std::string line;
    std::getline(cameras, line);
    std::vector<FlightFrame> frames;
    while (std::getline(cameras, line)) {
        std::vector<float> v;
        std::stringstream fields(line);
        for (std::string field; std::getline(fields, field, ',');)
            v.push_back(std::stof(field));
        if (v.size() < 11 || int(v[0]) < first)
            continue;
        FlightFrame frame;
        frame.index = int(v[0]);
        frame.eye = glm::vec3(v[1], v[2], v[3]);
        frame.look = glm::vec3(v[4], v[5], v[6]);
        frame.depthFar = v[10];
        auto& g = frame.gbuffers;
        g.depthFar = frame.depthFar;
        const auto path = [&](const char* kind) {
            char name[64];
            std::snprintf(name, sizeof(name), "classical%s_%04d.ppm", kind, frame.index);
            return flight / "frames" / name;
        };
        g.colour = readBgra(path(""), g.width, g.height);
        uint32_t w = 0, h = 0;
        g.classes = readBgra(path("_class"), w, h);
        g.depth16 = readBgra(path("_depth16"), w, h);
        g.u = readBgra(path("_u"), w, h);
        g.v = readBgra(path("_v"), w, h);
        if (g.colour.empty() || g.classes.empty() || g.depth16.empty() || g.u.empty() || g.v.empty())
            continue;
        frames.push_back(std::move(frame));
    }
    return frames;
}

// A depth16 BGRA image reduced to `size` by taking each block's centre pixel (distances must not be averaged).
std::vector<uint8_t> reduceDepth(const std::vector<uint8_t>& bgra, uint32_t width, uint32_t size) {
    const uint32_t factor = width / size;
    std::vector<uint8_t> out(size_t(size) * size * 4);
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x)
            for (size_t c = 0; c < 4; ++c)
                out[(size_t(y) * size + x) * 4 + c] =
                    bgra[(size_t(y * factor + factor / 2) * width + x * factor + factor / 2) * 4 + c];
    return out;
}

struct Batch {
    std::vector<float> channels, coordinates, classes, photos, rows, objects;
    /// The second views' crops and the flow onto the first views' crops (for the temporal loss).
    std::vector<float> channels2, coordinates2, classes2, objects2, flow, flowValid;
};

Batch sampleBatch(const std::vector<View>& views, uint32_t count, uint32_t crop, std::mt19937& rng) {
    Batch batch;
    const size_t cropPixels = size_t(crop) * crop;
    batch.channels.resize(count * kChannels * cropPixels);
    batch.coordinates.resize(count * 3 * cropPixels);
    batch.classes.resize(count * cropPixels);
    batch.photos.resize(count * 3 * cropPixels);
    if (!views.front().objects.empty())
        batch.objects.resize(count * cropPixels);
    const bool second = !views.front().flow.empty();
    if (second) {
        batch.channels2.resize(count * kChannels * cropPixels);
        batch.coordinates2.resize(count * 3 * cropPixels);
        batch.classes2.resize(count * cropPixels);
        batch.objects2.resize(count * cropPixels);
        batch.flow.resize(count * 2 * cropPixels);
        batch.flowValid.resize(count * cropPixels);
    }
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (uint32_t k = 0; k < count; ++k) {
        const uint32_t index = uint32_t(rng() % views.size());
        const View& view = views[index];
        batch.rows.push_back(float(view.code >= 0 ? uint32_t(view.code) : index));
        // Half-scale crops only without second views, whose flow is in full-scale pixels.
        const bool half = uniform(rng) < 0.3f && sampleSize / 2 >= crop && !second;
        const uint32_t size = half ? sampleSize / 2 : sampleSize, step = half ? 2 : 1;
        const uint32_t top = uint32_t(rng() % (size - crop + 1)), left = uint32_t(rng() % (size - crop + 1));
        const bool flip = flips && uniform(rng) < 0.5f;
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
                    if (!view.photo.empty())
                        batch.photos[(k * 3 + c) * cropPixels + p] = at(view.photo, c, y, x, true);
                }
                batch.classes[k * cropPixels + p] = at(view.inputs.classes, 0, y, x, false);
                if (!batch.objects.empty())
                    batch.objects[k * cropPixels + p] = at(view.objects, 0, y, x, false);
                if (!second)
                    continue;
                // The second view cropped at the same place; its flow in crop pixels, valid only inside the crop.
                for (size_t c = 0; c < kChannels; ++c)
                    batch.channels2[(k * kChannels + c) * cropPixels + p] =
                        at(view.secondInputs.channels, c, y, x, false);
                for (size_t c = 0; c < 3; ++c)
                    batch.coordinates2[(k * 3 + c) * cropPixels + p] =
                        at(view.secondInputs.coordinates, c, y, x, false);
                batch.classes2[k * cropPixels + p] = at(view.secondInputs.classes, 0, y, x, false);
                batch.objects2[k * cropPixels + p] = at(view.secondObjects, 0, y, x, false);
                const float fx = at(view.flow, 0, y, x, false) - float(left),
                            fy = at(view.flow, 1, y, x, false) - float(top);
                batch.flow[(k * 2) * cropPixels + p] = fx;
                batch.flow[(k * 2 + 1) * cropPixels + p] = fy;
                batch.flowValid[k * cropPixels + p] = at(view.flowValid, 0, y, x, false) > 0.0f && fx >= 0.0f &&
                                                              fy >= 0.0f && fx < float(crop) && fy < float(crop)
                                                          ? 1.0f
                                                          : 0.0f;
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
    bool profile = false, floor = true;
    uint32_t latent = 0;
    float klWeight = 1e-3f, freeBits = 0.0f, latentCheck = 0.0f;
    float temporal = 0.0f, temporalDelta = 3.0f; // the temporal loss's weight and the second views' offset in degrees
    uint32_t textureLevels = 12, textureBase = 16, textureFinest = 2048;
    std::filesystem::path initFrom;
    std::vector<uint32_t> widths{32, 64, 128}; // the generator's U-Net levels
    neural::TextureMlp textureMlp;             // the latent-conditioned MLP texture instead (--texture-mlp)
    neural::TextureBasis textureBasis;         // the code-mixed feature texture instead (--texture-basis)
    std::filesystem::path evaluate;            // a saved model to evaluate with and without its texture (--evaluate)
    // A flight of tree_bonsai_example --fly-in --gbuffers instead of the crate (--flight DIR PHOTOS FIRST): its
    // frames from FIRST on with their photos PHOTOS/<frame>.ppm.
    std::filesystem::path flightDir, flightPhotos;
    int flightFirst = 0;
    // After training, render the model along a flight (--render-flight DIR OUT).
    std::filesystem::path renderFlight, renderOut;
    // The GAN without reconstruction (--gan): random views of the crate with random latents against the photos.
    bool gan = false;
    // Many crates (--crates DIR): every crate of crate_paint_example's crates/<id> with its own row of a learned latent
    // table; the last --held-out-crates crates only for fitting their codes after training (--fit-steps).
    std::filesystem::path cratesDir;
    uint32_t heldOutCrates = 10, fitSteps = 1000;
    bool textureOnlyLatent = false; // the latents feed only the texture, not the generator (--texture-only-latent)
    uint32_t ganViews = 800;        // random views rendered for it
    bool unconditional = false;     // a discriminator of the image only
    uint32_t criticScales = 2;
    float discriminatorRate = 2e-4f;
    float networkRate = 1e-4f;     // the generator's (and encoder's) Adam learning rate
    uint32_t criticChannels = 0;   // the conditional discriminator's G-buffer channels (0: all)
    bool augmentGenerator = false; // DiffAugment in the generator's step too
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
        else if (arg == "--texture" && i + 3 < argc) {
            textureLevels = uint32_t(std::stoul(argv[++i]));
            textureBase = uint32_t(std::stoul(argv[++i]));
            textureFinest = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--flight" && i + 3 < argc) {
            flightDir = argv[++i];
            flightPhotos = argv[++i];
            flightFirst = std::stoi(argv[++i]);
        } else if (arg == "--render-flight" && i + 2 < argc) {
            renderFlight = argv[++i];
            renderOut = argv[++i];
        } else if (arg == "--crates" && hasValue) {
            cratesDir = argv[++i];
        } else if (arg == "--held-out-crates" && hasValue) {
            heldOutCrates = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--fit-steps" && hasValue) {
            fitSteps = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--texture-only-latent") {
            textureOnlyLatent = true;
        } else if (arg == "--gan") {
            gan = true;
        } else if (arg == "--gan-views" && hasValue) {
            ganViews = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--evaluate" && hasValue) {
            evaluate = argv[++i];
        } else if (arg == "--texture-mlp" && i + 3 < argc) {
            textureMlp.frequencies = uint32_t(std::stoul(argv[++i]));
            textureMlp.width = uint32_t(std::stoul(argv[++i]));
            textureMlp.outputs = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--texture-mlp-residual" && hasValue) {
            std::stringstream numbers(argv[++i]);
            std::vector<uint32_t> values;
            for (std::string number; std::getline(numbers, number, ',');)
                values.push_back(uint32_t(std::stoul(number)));
            if (values.empty() || values.size() % 2 != 0)
                throw std::invalid_argument("--texture-mlp-residual needs pairs of octaves and width");
            for (size_t k = 0; k < values.size(); k += 2)
                textureMlp.residual.emplace_back(values[k], values[k + 1]);
        } else if (arg == "--texture-basis" && i + 2 < argc) {
            textureBasis.channels = uint32_t(std::stoul(argv[++i]));
            std::stringstream numbers(argv[++i]);
            for (std::string number; std::getline(numbers, number, ',');)
                textureBasis.resolutions.push_back(uint32_t(std::stoul(number)));
        } else if (arg == "--basis-decay" && hasValue) {
            textureBasis.decay = std::stof(argv[++i]);
        } else if (arg == "--basis-smoothness" && hasValue) {
            textureBasis.smoothness = std::stof(argv[++i]);
        } else if (arg == "--perceptual" && hasValue)
            perceptual = std::stof(argv[++i]);
        else if (arg == "--adversarial" && hasValue)
            adversarial = std::stof(argv[++i]);
        else if (arg == "--feature-matching" && hasValue)
            featureMatching = std::stof(argv[++i]);
        else if (arg == "--init-from" && hasValue)
            initFrom = argv[++i];
        else if (arg == "--unconditional")
            unconditional = true;
        else if (arg == "--critic-scales" && hasValue)
            criticScales = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--discriminator-lr" && hasValue)
            discriminatorRate = std::stof(argv[++i]);
        else if (arg == "--lr" && hasValue)
            networkRate = std::stof(argv[++i]);
        else if (arg == "--critic-channels" && hasValue)
            criticChannels = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--augment-generator")
            augmentGenerator = true;
        else if (arg == "--widths" && hasValue) {
            std::stringstream numbers(argv[++i]);
            widths.clear();
            for (std::string number; std::getline(numbers, number, ',');)
                widths.push_back(uint32_t(std::stoul(number)));
        } else if (arg == "--no-floor")
            floor = false;
        else if (arg == "--latent" && hasValue)
            latent = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--kl" && hasValue)
            klWeight = std::stof(argv[++i]);
        else if (arg == "--free-bits" && hasValue)
            freeBits = std::stof(argv[++i]);
        else if (arg == "--latent-check" && hasValue)
            latentCheck = std::stof(argv[++i]);
        else if (arg == "--temporal" && hasValue)
            temporal = std::stof(argv[++i]);
        else if (arg == "--temporal-delta" && hasValue)
            temporalDelta = std::stof(argv[++i]);
        else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    try {
        HeadlessFrontend frontend;
        VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
        std::vector<View> training, validation;
        std::vector<View> fitting; // --crates: the held-out crates' views, for fitting their codes after training
        uint32_t crateCount = 0;
        // A flight's views (--flight): G-buffers and photos from disk, every tenth frame held out; each view's second
        // view for the temporal loss is the flight's next frame (the previous for the last), the flow between them from
        // the frames' distances and cameras.
        const bool fromFlight = !flightDir.empty();
        const size_t samplePixels = size_t(sampleSize) * sampleSize;
        // The latent renderer's sample of a frame: inverse depth and (missing) normals as geometry, every
        // non-background pixel one object (the latent is the whole image's).
        const auto sampleOf = [&](const FlightFrame& frame, neural::PhotoEnhancerInputs& inputs,
                                  std::vector<float>& objects) {
            inputs =
                neural::photoEnhancerInputs(frame.gbuffers, neural::PhotoEnhancerSample{sampleSize, sampleSize, 1.0f});
            if (inputs.normals.empty())
                inputs.normals.assign(3 * samplePixels, 0.0f);
            std::vector<float> geometry(inputs.channels.begin() + 3 * samplePixels,
                                        inputs.channels.begin() + 4 * samplePixels);
            geometry.insert(geometry.end(), inputs.normals.begin(), inputs.normals.end());
            inputs.channels = geometry;
            objects.assign(samplePixels, 0.0f);
            for (size_t q = 0; q < samplePixels; ++q)
                objects[q] = inputs.classes[q] != 0.0f ? 1.0f : 0.0f;
        };
        const auto cameraOf = [](const FlightFrame& frame) {
            CameraMVP camera{};
            crate::setCamera(camera, frame.eye, frame.look);
            return camera;
        };
        if (fromFlight) {
            if (latent == 0)
                throw std::runtime_error("--flight needs --latent");
            const auto frames = readFlightFrames(flightDir, flightFirst);
            for (size_t k = 0; k < frames.size(); ++k) {
                const FlightFrame& frame = frames[k];
                View view;
                char name[32];
                std::snprintf(name, sizeof(name), "%04d", frame.index);
                view.id = name;
                view.eye = frame.eye;
                view.look = frame.look;
                view.photo = readPhoto(flightPhotos / (view.id + ".ppm"));
                if (view.photo.empty())
                    continue;
                sampleOf(frame, view.inputs, view.objects);
                if (temporal > 0.0f) {
                    const FlightFrame& partner = frames[k + 1 < frames.size() ? k + 1 : k - 1];
                    sampleOf(partner, view.secondInputs, view.secondObjects);
                    const auto flow = crate::flow(
                        cameraOf(frame), reduceDepth(frame.gbuffers.depth16, frame.gbuffers.width, sampleSize),
                        cameraOf(partner), reduceDepth(partner.gbuffers.depth16, partner.gbuffers.width, sampleSize),
                        sampleSize, frame.depthFar, partner.depthFar);
                    view.flow = flow.x;
                    view.flow.insert(view.flow.end(), flow.y.begin(), flow.y.end());
                    view.flowValid.assign(flow.valid.begin(), flow.valid.end());
                }
                (k % 10 == 5 ? validation : training).push_back(std::move(view));
            }
            kChannels = 4;
            flips = false;
        } else if (!cratesDir.empty()) {
            // Every crate's views (crates/<id>/cameras_classical.csv, photos in crates/<id>/photo), each with its
            // crate's row; view 3 of every fifth crate held out for validation, the last heldOutCrates crates kept for
            // fitting.
            if (latent == 0)
                throw std::runtime_error("--crates needs --latent");
            const VkExtent2D extent{2 * sampleSize, 2 * sampleSize};
            auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
            GBufferMeshRenderer renderer(vc, crate::vertices(floor), target, extent);
            ComputeGraph graph(vc, 1);
            graph.compileFrom(renderer.getOutput());
            neural::PhotoAlignment alignment(vc);
            std::vector<std::filesystem::path> crates;
            for (const auto& entry : std::filesystem::directory_iterator(cratesDir / "crates"))
                if (entry.is_directory() && std::filesystem::exists(entry.path() / "photo"))
                    crates.push_back(entry.path());
            std::sort(crates.begin(), crates.end());
            crateCount = uint32_t(crates.size());
            if (crateCount <= heldOutCrates)
                throw std::runtime_error("--crates: fewer crates than --held-out-crates");
            for (uint32_t k = 0; k < crateCount; ++k)
                for (auto& view : loadViews(vc, renderer, graph, *target, alignment, crates[k], crates[k], minAlignment,
                                            floor ? 0.6f : 1.0f)) {
                    view.code = int(k);
                    const bool held = view.id.size() > 5 && view.id.substr(view.id.size() - 5) == "_0003";
                    (k >= crateCount - heldOutCrates ? fitting
                     : k % 5 == 0 && held            ? validation
                                                     : training)
                        .push_back(std::move(view));
                }
            std::cout << crateCount << " crates, " << heldOutCrates << " held out (" << fitting.size() << " views)"
                      << std::endl;
        } else {
            const VkExtent2D extent{2 * sampleSize, 2 * sampleSize};
            auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
            GBufferMeshRenderer renderer(vc, crate::vertices(floor), target, extent);
            ComputeGraph graph(vc, 1);
            graph.compileFrom(renderer.getOutput());
            neural::PhotoAlignment alignment(vc);
            training = loadViews(vc, renderer, graph, *target, alignment, crateDir / "flights/crate_a",
                                 crateDir / "data/train", minAlignment, floor ? 0.6f : 1.0f);
            validation = loadViews(vc, renderer, graph, *target, alignment, crateDir / "flights/crate_v",
                                   crateDir / "data/validation", minAlignment, floor ? 0.6f : 1.0f);
        }
        // The latent renderer's geometry: inverse depth and normals; its one object, the crate.
        if (latent > 0 && !fromFlight) {
            kChannels = 4;
            flips = false;
            const size_t pixels = size_t(sampleSize) * sampleSize;
            for (auto* views : {&training, &validation, &fitting})
                for (auto& view : *views) {
                    std::vector<float> geometry(view.inputs.channels.begin() + 3 * pixels,
                                                view.inputs.channels.begin() + 4 * pixels);
                    geometry.insert(geometry.end(), view.inputs.normals.begin(), view.inputs.normals.end());
                    view.inputs.channels = geometry;
                    view.objects.resize(pixels);
                    for (size_t q = 0; q < pixels; ++q)
                        view.objects[q] = view.inputs.classes[q] == float(tree::TREE_CLASS_STEM) ? 1.0f : 0.0f;
                }
        }
        // The GAN's views: random cameras as crate_gbuffer_example's random flights take them (any azimuth, 8 to 40
        // degrees above the crate, 2.2 to 3.8 away, looking near its middle), their samples as the latent renderer sees
        // them; no photos.
        std::vector<View> bank;
        if (gan) {
            if (latent == 0)
                throw std::runtime_error("--gan needs --latent");
            const VkExtent2D extent{2 * sampleSize, 2 * sampleSize};
            auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
            GBufferMeshRenderer renderer(vc, crate::vertices(floor), target, extent);
            ComputeGraph graph(vc, 1);
            graph.compileFrom(renderer.getOutput());
            std::mt19937 cameras(17);
            std::uniform_real_distribution<float> unit(0.0f, 1.0f);
            std::normal_distribution<float> jitter(0.0f, 0.12f);
            const size_t pixels = size_t(sampleSize) * sampleSize;
            for (uint32_t v = 0; v < ganViews; ++v) {
                View view;
                view.id = "random_" + std::to_string(v);
                const float azimuth = unit(cameras) * 6.2831853f;
                const float elevation = glm::radians(8.0f + 32.0f * unit(cameras));
                const float distance = 2.2f + 1.6f * unit(cameras);
                view.look = glm::vec3(0.0f, 0.5f, 0.0f) + glm::vec3(jitter(cameras), jitter(cameras), jitter(cameras));
                view.eye =
                    view.look + distance * glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                                     std::cos(azimuth) * std::cos(elevation));
                view.eye.y = std::max(view.eye.y, 0.15f);
                crate::setCamera(renderer.getCamera()->ubo, view.eye, view.look);
                renderer.getCamera()->update(0);
                const auto frame = crate::renderFrame(vc, renderer, graph, *target, 2 * sampleSize);
                view.inputs =
                    neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{sampleSize, sampleSize, 1.0f});
                std::vector<float> geometry(view.inputs.channels.begin() + 3 * pixels,
                                            view.inputs.channels.begin() + 4 * pixels);
                geometry.insert(geometry.end(), view.inputs.normals.begin(), view.inputs.normals.end());
                view.inputs.channels = geometry;
                view.inputs.normals.clear();
                view.objects.resize(pixels);
                for (size_t q = 0; q < pixels; ++q)
                    view.objects[q] = view.inputs.classes[q] == float(tree::TREE_CLASS_STEM) ? 1.0f : 0.0f;
                bank.push_back(std::move(view));
            }
            std::cout << bank.size() << " random views for the GAN" << std::endl;
        }
        // The temporal loss's second views: each training view's (with --gan: each random view's) camera turned by the
        // offset (alternately to either side) about the vertical through its look point, its sample as the model sees
        // it (without the floor if the views have none), and the flow of its pixels onto the view from G-buffers with
        // the floor.
        std::vector<View>& temporalViews = gan ? bank : training;
        if (temporal > 0.0f && !fromFlight) {
            if (latent == 0)
                throw std::runtime_error("--temporal needs --latent");
            const VkExtent2D extent{2 * sampleSize, 2 * sampleSize}, flowExtent{sampleSize, sampleSize};
            auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
            GBufferMeshRenderer renderer(vc, crate::vertices(floor), target, extent);
            ComputeGraph graph(vc, 1);
            graph.compileFrom(renderer.getOutput());
            auto flowTarget = std::make_shared<OffscreenTarget>(vc, flowExtent, 1);
            GBufferMeshRenderer flowRenderer(vc, crate::vertices(true), flowTarget, flowExtent);
            ComputeGraph flowGraph(vc, 1);
            flowGraph.compileFrom(flowRenderer.getOutput());
            const auto depth = [&](const CameraMVP& camera) {
                flowRenderer.getCamera()->ubo = camera;
                flowRenderer.getCamera()->update(0);
                flowRenderer.setOutput(GBufferOutput::Depth16, 0.0f, crate::kDepthFar);
                flowGraph.submitAndWait(vc.getGraphicsQueue(), 0);
                return crate::readImage(vc, flowTarget->getImage(0), flowExtent);
            };
            const size_t pixels = size_t(sampleSize) * sampleSize;
            for (size_t v = 0; v < temporalViews.size(); ++v) {
                View& view = temporalViews[v];
                const float angle = glm::radians(v % 2 == 0 ? temporalDelta : -temporalDelta);
                const glm::vec3 offset = view.eye - view.look;
                const glm::vec3 eye2 =
                    view.look + glm::vec3(std::cos(angle) * offset.x + std::sin(angle) * offset.z, offset.y,
                                          -std::sin(angle) * offset.x + std::cos(angle) * offset.z);
                CameraMVP first{}, second{};
                crate::setCamera(first, view.eye, view.look);
                crate::setCamera(second, eye2, view.look);
                renderer.getCamera()->ubo = second;
                renderer.getCamera()->update(0);
                const auto frame = crate::renderFrame(vc, renderer, graph, *target, 2 * sampleSize);
                view.secondInputs =
                    neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{sampleSize, sampleSize, 1.0f});
                std::vector<float> geometry(view.secondInputs.channels.begin() + 3 * pixels,
                                            view.secondInputs.channels.begin() + 4 * pixels);
                geometry.insert(geometry.end(), view.secondInputs.normals.begin(), view.secondInputs.normals.end());
                view.secondInputs.channels = geometry;
                view.secondInputs.normals.clear();
                view.secondObjects.resize(pixels);
                for (size_t q = 0; q < pixels; ++q)
                    view.secondObjects[q] = view.secondInputs.classes[q] == float(tree::TREE_CLASS_STEM) ? 1.0f : 0.0f;
                const auto flow = crate::flow(first, depth(first), second, depth(second), sampleSize);
                view.flow = flow.x;
                view.flow.insert(view.flow.end(), flow.y.begin(), flow.y.end());
                view.flowValid.assign(flow.valid.begin(), flow.valid.end());
            }
            double valid = 0.0, length = 0.0;
            for (const auto& view : temporalViews)
                for (size_t p = 0; p < pixels; ++p)
                    if (view.flowValid[p] > 0.0f) {
                        valid += 1.0;
                        length += std::hypot(view.flow[p] - float(p % sampleSize) - 0.5f,
                                             view.flow[pixels + p] - float(size_t(p / sampleSize)) - 0.5f);
                    }
            std::printf("second views at %.1f degrees: %.1f %% of the pixels valid, flow %.2f px on average\n",
                        temporalDelta, 100.0 * valid / double(temporalViews.size() * pixels),
                        length / std::max(valid, 1.0));
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
        config.widths = widths;
        config.texture = textureMlp.frequencies > 0 || textureBasis.enabled()
                             ? crate::denseTextureShape(0, 0)
                             : crate::textureShape(textureLevels, textureBase, textureFinest);
        config.textureMlp = textureMlp;
        config.textureBasis = textureBasis;
        // A flight's texture covers the terrain, the trunks and the leaves.
        if (fromFlight)
            config.texture.classes = {tree::TREE_CLASS_SOIL, tree::TREE_CLASS_STEM, tree::TREE_CLASS_LEAF};
        config.appearances = appearance > 0 && latent == 0 ? uint32_t(training.size()) : 0;
        if (latent > 0) {
            config.gbufferChannels = kChannels;
            config.latentSize = latent;
            config.latentObjects = 1;
            config.klWeight = klWeight;
            config.freeBits = freeBits;
            config.latentCheck = latentCheck;
            config.temporal = temporal;
            config.latentCodes = crateCount;
            config.latentToDecoder = !textureOnlyLatent;
        }
        config.appearanceSize = appearance;
        config.profile = profile;
        config.perceptual = perceptual;
        config.adversarial = adversarial;
        config.featureMatching = featureMatching;
        config.conditionalDiscriminator = !unconditional;
        // The GAN without reconstruction: no encoder, no photo losses, the discriminator on inverse depth and image
        // with the photos' own depth, DiffAugment in the generator step too.
        if (gan) {
            config.randomLatents = true;
            config.l1 = config.colour = 0.0f;
            config.perceptual = 0.0f;
            config.featureMatching = 0.0f;
            config.criticGBufferChannels = 1;
            config.separateRealGBuffer = true;
            config.augmentGeneratorStep = true;
            if (config.adversarial <= 0.0f)
                config.adversarial = 1.0f;
        }
        config.criticScales = criticScales;
        config.discriminatorAdam.learningRate = discriminatorRate;
        config.networkAdam.learningRate = networkRate;
        config.criticGBufferChannels = criticChannels;
        config.augmentGeneratorStep = augmentGenerator;
        // Released before the held-out crates' codes are fitted, so that the GPU holds one training at a time.
        auto trainerOwner = std::make_unique<neural::PhotoEnhancerTraining>(vc, config);
        auto& trainer = *trainerOwner;
        std::mt19937 rng(1);
        std::vector<std::string> fresh; // generator, texture and encoder parameters not taken from --init-from
        for (const auto& name : trainer.parameterNames()) {
            auto values = crate::initialValue(trainer, name, rng);
            // The generator, texture and codes from an earlier run (its saved averages), if given: every parameter of
            // the same shape. A shallower U-Net's up blocks become the last ones of this one (the full-resolution
            // end); the deeper levels keep their fresh initialisation.
            std::string savedName = name;
            const std::string up = "generator.up.";
            if (!initFrom.empty() && name.rfind(up, 0) == 0) {
                uint32_t savedLevels = 0;
                while (std::filesystem::exists(initFrom /
                                               ("generator.down." + std::to_string(savedLevels) + ".0.bias.bin")))
                    ++savedLevels;
                const size_t dot = name.find('.', up.size());
                const int index =
                    std::stoi(name.substr(up.size(), dot - up.size())) - (int(widths.size()) - int(savedLevels));
                savedName = index >= 0 ? up + std::to_string(index) + name.substr(dot) : "";
            }
            const auto saved = initFrom / (savedName + ".bin");
            if (!initFrom.empty() && !savedName.empty() && std::filesystem::exists(saved) &&
                std::filesystem::file_size(saved) == values.size() * sizeof(float) &&
                name.rfind("discriminator.", 0) != 0) {
                std::ifstream file(saved, std::ios::binary);
                file.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
            } else if (!initFrom.empty() && name.rfind("discriminator.", 0) != 0) {
                fresh.push_back(name);
            }
            trainer.setParameter(name, values);
        }
        if (!initFrom.empty()) {
            std::cout << "from " << initFrom << ": all but " << fresh.size() << " parameters, fresh:";
            for (const auto& name : fresh)
                std::cout << " " << name;
            std::cout << std::endl;
        }

        // The latent renderer's validation: renders of the held-out views with the latents their photos encode to, with
        // latents 0, and of one view with latents drawn from N(0, 1).
        constexpr uint32_t kLatentBatch = 4;
        std::unique_ptr<neural::PhotoEnhancerTraining> encodeRenderer, givenRenderer;
        if (latent > 0) {
            auto encodeConfig = config;
            encodeConfig.forwardOnly = true;
            encodeConfig.encodeLatents = !gan;
            encodeConfig.randomLatents = false;
            encodeConfig.batch = kLatentBatch;
            encodeConfig.size = sampleSize;
            encodeConfig.perceptual = encodeConfig.adversarial = 0.0f;
            encodeConfig.latentCodes = 0;
            if (!gan && cratesDir.empty())
                encodeRenderer = std::make_unique<neural::PhotoEnhancerTraining>(vc, encodeConfig);
            auto givenConfig = encodeConfig;
            givenConfig.encodeLatents = false;
            givenRenderer = std::make_unique<neural::PhotoEnhancerTraining>(vc, givenConfig);
        }
        // Renders `views` in chunks (the last filled up with the first views) with latents encoded from the photos or,
        // given `latents` ([views][latent]), with those; the images [views][3][size][size].
        const auto renderLatent = [&](const std::vector<const View*>& views, const std::vector<float>* latents,
                                      neural::PhotoEnhancerTraining* encoding = nullptr) {
            const size_t pixels = size_t(sampleSize) * sampleSize;
            auto& renderer = latents ? *givenRenderer : encoding ? *encoding : *encodeRenderer;
            std::vector<float> images;
            for (size_t first = 0; first < views.size(); first += kLatentBatch) {
                Batch chunk;
                std::vector<float> chunkLatents;
                for (uint32_t k = 0; k < kLatentBatch; ++k) {
                    const size_t v = (first + k) % views.size();
                    const View& view = *views[v];
                    chunk.channels.insert(chunk.channels.end(), view.inputs.channels.begin(),
                                          view.inputs.channels.end());
                    chunk.coordinates.insert(chunk.coordinates.end(), view.inputs.coordinates.begin(),
                                             view.inputs.coordinates.end());
                    chunk.classes.insert(chunk.classes.end(), view.inputs.classes.begin(), view.inputs.classes.end());
                    chunk.photos.insert(chunk.photos.end(), view.photo.begin(), view.photo.end());
                    chunk.objects.insert(chunk.objects.end(), view.objects.begin(), view.objects.end());
                    if (latents)
                        chunkLatents.insert(chunkLatents.end(), latents->begin() + v * latent,
                                            latents->begin() + (v + 1) * latent);
                }
                renderer.setBatch(chunk.channels, chunk.coordinates, chunk.classes, chunk.photos, {});
                renderer.setObjects(chunk.objects);
                if (latents)
                    renderer.setLatents(chunkLatents);
                renderer.step();
                const auto part = renderer.image();
                const size_t count = std::min<size_t>(kLatentBatch, views.size() - first);
                images.insert(images.end(), part.begin(), part.begin() + count * 3 * pixels);
            }
            return images;
        };
        const auto meanL1 = [&](const std::vector<float>& images, const std::vector<const View*>& views) {
            const size_t pixels = size_t(sampleSize) * sampleSize;
            double sum = 0.0;
            for (size_t v = 0; v < views.size(); ++v)
                for (size_t i = 0; i < 3 * pixels; ++i)
                    sum += std::abs(images[v * 3 * pixels + i] - views[v]->photo[i]);
            return sum / double(views.size() * 3 * pixels);
        };
        // Columns of 3-channel images, side by side, rows one after the other.
        const auto writeGrid = [&](const std::filesystem::path& path, const std::vector<std::vector<float>>& columns,
                                   size_t rows) {
            const size_t pixels = size_t(sampleSize) * sampleSize;
            const uint32_t width = uint32_t(columns.size()) * sampleSize;
            std::vector<uint8_t> grid(size_t(width) * sampleSize * rows * 3);
            for (size_t r = 0; r < rows; ++r)
                for (size_t column = 0; column < columns.size(); ++column)
                    for (size_t p = 0; p < pixels; ++p)
                        for (size_t c = 0; c < 3; ++c) {
                            const size_t y = r * sampleSize + p / sampleSize, x = column * sampleSize + p % sampleSize;
                            grid[(y * width + x) * 3 + c] = byte(columns[column][(r * 3 + c) * pixels + p]);
                        }
            writeImage(path, grid.data(), width, uint32_t(sampleSize * rows), 3);
        };

        // --crates: the latent table as saved (the averages, [crates][latent]), each view's row of it, and the mean of
        // the training crates' rows.
        const auto readTable = [&](const std::filesystem::path& parameters) {
            std::vector<float> table(size_t(crateCount) * latent);
            std::ifstream file(parameters / "latents.weight.bin", std::ios::binary);
            file.read(reinterpret_cast<char*>(table.data()), std::streamsize(table.size() * sizeof(float)));
            return table;
        };
        const auto codesOf = [&](const std::vector<const View*>& views, const std::vector<float>& table) {
            std::vector<float> codes;
            for (const auto* view : views)
                codes.insert(codes.end(), table.begin() + size_t(view->code) * latent,
                             table.begin() + size_t(view->code + 1) * latent);
            return codes;
        };
        const auto meanCodeOf = [&](const std::vector<float>& table, size_t views) {
            std::vector<float> mean(latent, 0.0f);
            const uint32_t trained = crateCount - heldOutCrates;
            for (uint32_t k = 0; k < trained; ++k)
                for (uint32_t d = 0; d < latent; ++d)
                    mean[d] += table[size_t(k) * latent + d] / float(trained);
            std::vector<float> codes;
            for (size_t v = 0; v < views; ++v)
                codes.insert(codes.end(), mean.begin(), mean.end());
            return codes;
        };

        // Validation: the held-out views at the full sample size, with the codes' mean.
        auto renderConfig = config;
        renderConfig.forwardOnly = true;
        constexpr uint32_t kRenderBatch = 4; // held-out views rendered at a time
        renderConfig.batch = kRenderBatch;
        renderConfig.size = sampleSize;
        std::unique_ptr<neural::PhotoEnhancerTraining> renderer;
        if (!validation.empty() && latent == 0)
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

        // --evaluate: the saved model on the held-out views with its latents encoded from the photos, once as it is and
        // once with its texture's features zeroed; the texture's features of the first views as images.
        if (!evaluate.empty()) {
            if (latent == 0 || validation.empty())
                throw std::runtime_error("--evaluate needs --latent and validation views");
            auto plainConfig = config;
            plainConfig.forwardOnly = true;
            plainConfig.encodeLatents = true;
            plainConfig.textureOff = true;
            plainConfig.batch = kLatentBatch;
            plainConfig.size = sampleSize;
            plainConfig.perceptual = plainConfig.adversarial = 0.0f;
            neural::PhotoEnhancerTraining plain(vc, plainConfig);
            encodeRenderer->load(evaluate.string());
            plain.load(evaluate.string());
            std::vector<const View*> held, fitted;
            for (const auto& view : validation)
                held.push_back(&view);
            for (const auto& view : training)
                fitted.push_back(&view);
            const auto with = renderLatent(held, nullptr), without = renderLatent(held, nullptr, &plain);
            const auto fittedWith = renderLatent(fitted, nullptr),
                       fittedWithout = renderLatent(fitted, nullptr, &plain);
            std::printf("evaluation of %s: validation L1 %.4f with the texture, %.4f without; training L1 %.4f with, "
                        "%.4f without\n",
                        evaluate.string().c_str(), meanL1(with, held), meanL1(without, held),
                        meanL1(fittedWith, fitted), meanL1(fittedWithout, fitted));
            std::vector<float> photos;
            for (const auto* view : held)
                photos.insert(photos.end(), view->photo.begin(), view->photo.end());
            std::filesystem::create_directories(outDir);
            writeGrid(outDir / "evaluation.png", {photos, with, without}, std::min<size_t>(6, held.size()));
            // The texture's features of the first held-out views (the generator input's channels after the G-buffer's),
            // each channel grey around 0.5, scaled by its largest magnitude over these views.
            renderLatent(
                std::vector<const View*>(held.begin(), held.begin() + std::min<size_t>(kLatentBatch, held.size())),
                nullptr);
            const auto input = encodeRenderer->generatorInput();
            const size_t pixels = size_t(sampleSize) * sampleSize;
            const uint32_t channelsIn = uint32_t(input.size() / (kLatentBatch * pixels));
            const uint32_t features = channelsIn - kChannels - latent;
            std::vector<std::vector<float>> columns(features, std::vector<float>(kLatentBatch * 3 * pixels));
            for (uint32_t f = 0; f < features; ++f) {
                float largest = 1e-12f;
                for (uint32_t k = 0; k < kLatentBatch; ++k)
                    for (size_t p = 0; p < pixels; ++p)
                        largest = std::max(largest, std::abs(input[(k * channelsIn + kChannels + f) * pixels + p]));
                for (uint32_t k = 0; k < kLatentBatch; ++k)
                    for (size_t p = 0; p < pixels; ++p)
                        for (size_t c = 0; c < 3; ++c)
                            columns[f][(k * 3 + c) * pixels + p] =
                                0.5f + 0.5f * input[(k * channelsIn + kChannels + f) * pixels + p] / largest;
            }
            if (features > 0)
                writeGrid(outDir / "texture_features.png", columns, kLatentBatch);
            return 0;
        }

        std::ofstream log(outDir / "log.txt", std::ios::app);
        double lossSum = 0.0, perceptualSum = 0.0, discriminatorSum = 0.0, adversarialSum = 0.0, matchingSum = 0.0,
               klSum = 0.0, checkSum = 0.0, temporalSum = 0.0;
        auto started = std::chrono::steady_clock::now();
        for (uint32_t step = 1; step <= steps; ++step) {
            // With --gan the generated views are random ones and the photos come with their own views' depth.
            const Batch batch = sampleBatch(gan ? bank : training, batchSize, crop, rng);
            if (gan) {
                const Batch real = sampleBatch(training, batchSize, crop, rng);
                trainer.setBatch(batch.channels, batch.coordinates, batch.classes, real.photos, batch.rows);
                trainer.setRealGBuffer(real.channels);
                std::normal_distribution<float> normal(0.0f, 1.0f);
                std::vector<float> latents(batchSize * latent);
                for (auto& v : latents)
                    v = normal(rng);
                trainer.setLatents(latents);
            } else {
                trainer.setBatch(batch.channels, batch.coordinates, batch.classes, batch.photos, batch.rows);
            }
            if (latent > 0)
                trainer.setObjects(batch.objects);
            if (temporal > 0.0f)
                trainer.setSecondView(batch.channels2, batch.coordinates2, batch.classes2, batch.objects2, batch.flow,
                                      batch.flowValid);
            trainer.step();
            lossSum += trainer.loss();
            perceptualSum += trainer.perceptualLoss();
            discriminatorSum += trainer.discriminatorLoss();
            adversarialSum += trainer.adversarialLoss();
            matchingSum += trainer.featureMatchingLoss();
            klSum += trainer.klLoss();
            checkSum += trainer.latentCheckLoss();
            temporalSum += trainer.temporalLoss();
            if (step % 100 == 0) {
                const double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() / 100.0;
                std::ostringstream line;
                line << "step " << step << " loss " << lossSum / 100.0;
                if (perceptual > 0.0f)
                    line << " vgg " << perceptualSum / 100.0;
                if (latent > 0)
                    line << " kl " << klSum / 100.0;
                if (latentCheck > 0.0f)
                    line << " check " << checkSum / 100.0;
                if (temporal > 0.0f)
                    line << " temporal " << temporalSum / 100.0;
                if (config.adversarial > 0.0f)
                    line << " d " << discriminatorSum / 100.0 << " adv " << adversarialSum / 100.0 << " fm "
                         << matchingSum / 100.0;
                line << " (" << seconds << " s/step)";
                perceptualSum = discriminatorSum = adversarialSum = matchingSum = klSum = checkSum = temporalSum = 0.0;
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
                // --gan: 4 fixed random views, each with 4 fixed latents from N(0, 1), as samples_*.png (views in
                // rows), and the samples' spread (mean L1 between the same view's renders with different latents).
                if (gan) {
                    givenRenderer->load((outDir / "parameters").string());
                    std::vector<const View*> views;
                    std::vector<float> latents;
                    std::mt19937 sampler(42);
                    std::normal_distribution<float> normal(0.0f, 1.0f);
                    for (size_t row = 0; row < 4; ++row)
                        for (size_t column = 0; column < 4; ++column) {
                            views.push_back(&bank[(row * 97) % bank.size()]);
                            for (uint32_t d = 0; d < latent; ++d)
                                latents.push_back(normal(sampler));
                        }
                    // The same 4 latents in every row.
                    for (size_t row = 1; row < 4; ++row)
                        std::copy(latents.begin(), latents.begin() + 4 * latent, latents.begin() + row * 4 * latent);
                    const auto images = renderLatent(views, &latents);
                    const size_t pixels = size_t(sampleSize) * sampleSize;
                    double spread = 0.0;
                    for (size_t row = 0; row < 4; ++row)
                        for (size_t column = 0; column + 1 < 4; ++column)
                            for (size_t i = 0; i < 3 * pixels; ++i)
                                spread += std::abs(images[(row * 4 + column) * 3 * pixels + i] -
                                                   images[(row * 4 + column + 1) * 3 * pixels + i]);
                    spread /= double(4 * 3 * 3 * pixels);
                    std::ostringstream line;
                    line << "validation step " << step << ": samples' spread " << spread;
                    std::cout << line.str() << std::endl;
                    log << line.str() << "\n";
                    log.flush();
                    std::vector<std::vector<float>> columns(4);
                    for (size_t column = 0; column < 4; ++column)
                        for (size_t row = 0; row < 4; ++row)
                            columns[column].insert(columns[column].end(),
                                                   images.begin() + (row * 4 + column) * 3 * pixels,
                                                   images.begin() + (row * 4 + column + 1) * 3 * pixels);
                    char name[64];
                    std::snprintf(name, sizeof(name), "samples_%06u.png", step);
                    writeGrid(outDir / name, columns, 4);
                    continue;
                }
                // --crates: the held-out views with their crate's code and with the mean code, eight training views
                // with their codes, and 16 crates on the same view (view 0 has the same camera for every crate) with
                // the mean L1 between consecutive crates (how different the codes make them).
                if (!cratesDir.empty()) {
                    givenRenderer->load((outDir / "parameters").string());
                    const auto table = readTable(outDir / "parameters");
                    std::vector<const View*> held, fitted;
                    for (const auto& view : validation)
                        held.push_back(&view);
                    for (size_t v = 0; v < 8; ++v)
                        fitted.push_back(&training[v * training.size() / 8]);
                    const auto heldCodes = codesOf(held, table), heldMean = meanCodeOf(table, held.size());
                    const auto fittedCodes = codesOf(fitted, table);
                    const auto own = renderLatent(held, &heldCodes), plain = renderLatent(held, &heldMean);
                    const auto trainingImages = renderLatent(fitted, &fittedCodes);
                    const View* front = &training.front();
                    for (const auto& view : training)
                        if (view.id.size() > 5 && view.id.substr(view.id.size() - 5) == "_0000") {
                            front = &view;
                            break;
                        }
                    const uint32_t shownCrates = std::min<uint32_t>(16, crateCount - heldOutCrates);
                    const std::vector<const View*> same(shownCrates, front);
                    const std::vector<float> crateCodes(table.begin(), table.begin() + size_t(shownCrates) * latent);
                    const auto crates = renderLatent(same, &crateCodes);
                    const size_t pixels = size_t(sampleSize) * sampleSize;
                    double spread = 0.0;
                    for (size_t v = 0; v + 1 < shownCrates; ++v)
                        for (size_t i = 0; i < 3 * pixels; ++i)
                            spread += std::abs(crates[v * 3 * pixels + i] - crates[(v + 1) * 3 * pixels + i]);
                    spread /= double(std::max<uint32_t>(1, shownCrates - 1) * 3 * pixels);
                    std::ostringstream line;
                    line << "validation step " << step << ": L1 own code " << meanL1(own, held) << ", mean code "
                         << meanL1(plain, held) << ", training " << meanL1(trainingImages, fitted)
                         << ", crates' spread " << spread;
                    std::cout << line.str() << std::endl;
                    log << line.str() << "\n";
                    log.flush();
                    // validation_*.png: photo | own code | mean code; crates_*.png: the crates in rows of four.
                    std::vector<float> photos;
                    for (const auto* view : held)
                        photos.insert(photos.end(), view->photo.begin(), view->photo.end());
                    char name[64];
                    std::snprintf(name, sizeof(name), "validation_%06u.png", step);
                    writeGrid(outDir / name, {photos, own, plain}, std::min<size_t>(8, held.size()));
                    // Rows of four, the last one filled up with black.
                    const uint32_t gridRows = (shownCrates + 3) / 4;
                    std::vector<std::vector<float>> columns(4);
                    for (uint32_t v = 0; v < 4 * gridRows; ++v) {
                        if (v < shownCrates)
                            columns[v % 4].insert(columns[v % 4].end(), crates.begin() + v * 3 * pixels,
                                                  crates.begin() + (v + 1) * 3 * pixels);
                        else
                            columns[v % 4].resize(columns[v % 4].size() + 3 * pixels, 0.0f);
                    }
                    std::snprintf(name, sizeof(name), "crates_%06u.png", step);
                    writeGrid(outDir / name, columns, gridRows);
                    continue;
                }
                if (latent > 0 && !validation.empty()) {
                    encodeRenderer->load((outDir / "parameters").string());
                    givenRenderer->load((outDir / "parameters").string());
                    std::vector<const View*> held, fitted;
                    for (const auto& view : validation)
                        held.push_back(&view);
                    for (size_t v = 0; v < std::min<size_t>(8, training.size()); ++v)
                        fitted.push_back(&training[v]);
                    const auto encoded = renderLatent(held, nullptr);
                    const std::vector<float> zeros(held.size() * latent, 0.0f);
                    const auto plain = renderLatent(held, &zeros);
                    const auto trainingImages = renderLatent(fitted, nullptr);
                    const std::vector<float> fittedZeros(fitted.size() * latent, 0.0f);
                    const auto trainingPlain = renderLatent(fitted, &fittedZeros);
                    // Eight crates from N(0, 1) on the first held-out view.
                    std::mt19937 sampler(42);
                    std::normal_distribution<float> normal(0.0f, 1.0f);
                    std::vector<float> drawn(8 * latent);
                    for (auto& v : drawn)
                        v = normal(sampler);
                    const std::vector<const View*> same(8, held.front());
                    const auto samples = renderLatent(same, &drawn);
                    double spread = 0.0; // mean L1 between consecutive samples
                    const size_t pixels = size_t(sampleSize) * sampleSize;
                    for (size_t v = 0; v + 1 < 8; ++v)
                        for (size_t i = 0; i < 3 * pixels; ++i)
                            spread += std::abs(samples[v * 3 * pixels + i] - samples[(v + 1) * 3 * pixels + i]);
                    spread /= double(7 * 3 * pixels);
                    std::ostringstream line;
                    line << "validation step " << step << ": L1 encoded " << meanL1(encoded, held) << ", latents 0 "
                         << meanL1(plain, held) << ", training encoded " << meanL1(trainingImages, fitted)
                         << ", training latents 0 " << meanL1(trainingPlain, fitted) << ", samples' spread " << spread;
                    std::cout << line.str() << std::endl;
                    log << line.str() << "\n";
                    log.flush();
                    // validation_*.png: photo | encoded | latents 0 for the first six held-out views; samples_*.png:
                    // the eight drawn crates in two rows.
                    std::vector<float> photos;
                    for (const auto* view : held)
                        photos.insert(photos.end(), view->photo.begin(), view->photo.end());
                    char name[64];
                    std::snprintf(name, sizeof(name), "validation_%06u.png", step);
                    writeGrid(outDir / name, {photos, encoded, plain}, std::min<size_t>(6, held.size()));
                    std::vector<float> top(samples.begin(), samples.begin() + 4 * 3 * pixels),
                        bottom(samples.begin() + 4 * 3 * pixels, samples.end());
                    std::vector<std::vector<float>> columns;
                    for (size_t v = 0; v < 4; ++v) {
                        std::vector<float> column(top.begin() + v * 3 * pixels, top.begin() + (v + 1) * 3 * pixels);
                        column.insert(column.end(), bottom.begin() + v * 3 * pixels,
                                      bottom.begin() + (v + 1) * 3 * pixels);
                        columns.push_back(column);
                    }
                    std::snprintf(name, sizeof(name), "samples_%06u.png", step);
                    writeGrid(outDir / name, columns, 2);
                    continue;
                }
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
        // --crates: the held-out crates' codes fitted on their first three views with everything else fixed (the
        // network's and texture's learning rates 0, no temporal loss, no GAN), each starting at the training crates'
        // mean code; then their other three views rendered with the fitted codes and with the mean code
        // (new_crates.png: photo | fitted | mean, one view per crate). The whole table goes to crate_codes.txt, one
        // crate per line.
        if (!cratesDir.empty()) {
            auto table = readTable(outDir / "parameters");
            if (!fitting.empty() && fitSteps > 0) {
                trainerOwner.reset();
                auto fitConfig = config;
                fitConfig.networkAdam.learningRate = 0.0f;
                fitConfig.textureAdam.learningRate = 0.0f;
                fitConfig.temporal = 0.0f;
                fitConfig.adversarial = 0.0f;
                neural::PhotoEnhancerTraining fitter(vc, fitConfig);
                fitter.load((outDir / "parameters").string());
                const auto mean = meanCodeOf(table, 1);
                for (uint32_t k = crateCount - heldOutCrates; k < crateCount; ++k)
                    std::copy(mean.begin(), mean.end(), table.begin() + size_t(k) * latent);
                fitter.setParameter("latents.weight", table);
                std::vector<View> fitViews, testViews;
                for (const auto& view : fitting) {
                    const int index = std::stoi(view.id.substr(view.id.size() - 4));
                    (index < 3 ? fitViews : testViews).push_back(view);
                }
                double fitLoss = 0.0;
                for (uint32_t step = 1; step <= fitSteps; ++step) {
                    const Batch batch = sampleBatch(fitViews, batchSize, crop, rng);
                    fitter.setBatch(batch.channels, batch.coordinates, batch.classes, batch.photos, batch.rows);
                    fitter.setObjects(batch.objects);
                    fitter.step();
                    fitLoss += fitter.loss();
                    if (step % 100 == 0) {
                        std::ostringstream line;
                        line << "fit step " << step << " loss " << fitLoss / 100.0;
                        std::cout << line.str() << std::endl;
                        log << line.str() << "\n";
                        fitLoss = 0.0;
                    }
                }
                fitter.parameter("latents.weight")->getDataBuffer(0).memcopyTo(table);
                givenRenderer->load((outDir / "parameters").string());
                std::vector<const View*> tests;
                for (const auto& view : testViews)
                    tests.push_back(&view);
                const auto fittedCodes = codesOf(tests, table), meanCodes = meanCodeOf(table, tests.size());
                const auto fitted = renderLatent(tests, &fittedCodes), plain = renderLatent(tests, &meanCodes);
                std::ostringstream line;
                line << "new crates: L1 " << meanL1(fitted, tests) << " with codes fitted on 3 views, "
                     << meanL1(plain, tests) << " with the mean code (" << tests.size() << " views of " << heldOutCrates
                     << " crates)";
                std::cout << line.str() << std::endl;
                log << line.str() << "\n";
                log.flush();
                std::vector<const View*> shown;
                std::vector<float> photos, shownFitted, shownPlain;
                const size_t pixels = size_t(sampleSize) * sampleSize;
                for (size_t v = 0; v < tests.size(); ++v)
                    if (tests[v]->id.substr(tests[v]->id.size() - 4) == "0003") {
                        photos.insert(photos.end(), tests[v]->photo.begin(), tests[v]->photo.end());
                        shownFitted.insert(shownFitted.end(), fitted.begin() + v * 3 * pixels,
                                           fitted.begin() + (v + 1) * 3 * pixels);
                        shownPlain.insert(shownPlain.end(), plain.begin() + v * 3 * pixels,
                                          plain.begin() + (v + 1) * 3 * pixels);
                    }
                writeGrid(outDir / "new_crates.png", {photos, shownFitted, shownPlain}, photos.size() / (3 * pixels));
            }
            std::ofstream codes(outDir / "crate_codes.txt");
            for (uint32_t k = 0; k < crateCount; ++k) {
                for (uint32_t d = 0; d < latent; ++d)
                    codes << (d > 0 ? "," : "") << table[size_t(k) * latent + d];
                codes << "\n";
            }
        }
        // --render-flight: the trained model along a (finely rendered) flight, latents 0, OUT/frame_<frame>.ppm.
        if (!renderFlight.empty()) {
            if (!givenRenderer)
                throw std::runtime_error("--render-flight needs --latent");
            givenRenderer->load((outDir / "parameters").string());
            const auto frames = readFlightFrames(renderFlight, flightFirst);
            std::vector<View> views(frames.size());
            std::vector<const View*> pointers;
            for (size_t k = 0; k < frames.size(); ++k) {
                sampleOf(frames[k], views[k].inputs, views[k].objects);
                views[k].photo.assign(3 * samplePixels, 0.0f);
                pointers.push_back(&views[k]);
            }
            const std::vector<float> zeros(views.size() * latent, 0.0f);
            const auto images = renderLatent(pointers, &zeros);
            std::filesystem::create_directories(renderOut);
            for (size_t k = 0; k < frames.size(); ++k) {
                std::vector<uint8_t> rgb(3 * samplePixels);
                for (size_t q = 0; q < samplePixels; ++q)
                    for (size_t c = 0; c < 3; ++c)
                        rgb[3 * q + c] = byte(images[(k * 3 + c) * samplePixels + q]);
                char name[64];
                std::snprintf(name, sizeof(name), "frame_%04d.png", frames[k].index);
                writeImage(renderOut / name, rgb.data(), sampleSize, sampleSize, 3);
            }
            std::cout << frames.size() << " flight frames rendered to " << renderOut << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "crate_training_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
