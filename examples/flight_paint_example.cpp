// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Consistent Qwen-Image photos along a flight (tree_bonsai_example --fly-in --gbuffers): every frame's photo is
// generated from the previous frame's photo warped onto it by the geometric flow (each pixel's point from its
// distance and the camera, projected into the previous frame, kept where that frame sees it), the newly visible parts
// from the classical rendering, so that consecutive photos show the same world, the way crate_paint_example paints
// the crate. The flight's frames are <flight>/frames/classical_<frame>.ppm, classical_depth16_<frame>.ppm and its
// cameras <flight>/cameras_classical.csv (eye, look, near plane, vertical field of view, depth window).
//
//   --check FLIGHT              the flow's check: each rendering warped onto the next frame against that frame's own
//                               rendering (mean squared difference where valid; small if the flow is right)
//   --start FLIGHT FRAME PHOTOS OUT  the start image of FRAME: the photo of the frame before it (PHOTOS/<frame>.ppm)
//                               warped onto it where valid, elsewhere its rendering; OUT/start_<frame>.ppm and the
//                               depth control OUT/control_<frame>.ppm, and the share taken from the photo
//   --consistency FLIGHT PHOTOS  how well consecutive photos agree: each warped onto the next, the mean squared
//                               difference where valid

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

// A frame of the flight: its camera and depth window, and its images as RGB bytes.
struct Frame {
    int index = 0;
    glm::vec3 eye, look;
    float nearPlane = 0.0f, fov = 40.0f, depthFar = 1.0f;
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> colour, depth16;
    glm::mat4 viewProj{1.0f};

    // The distance at pixel p (0: nothing drawn).
    float distance(size_t p) const { return float(depth16[3 * p] * 256u + depth16[3 * p + 1]) / 65535.0f * depthFar; }
};

std::vector<uint8_t> readPpm(const std::filesystem::path& path, uint32_t& width, uint32_t& height) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    uint32_t maximum = 0;
    if (!(file >> magic >> width >> height >> maximum) || magic != "P6")
        return {};
    file.get();
    std::vector<uint8_t> bytes(size_t(width) * height * 3);
    file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    return bytes;
}

void writePpm(const std::filesystem::path& path, const std::vector<uint8_t>& rgb, uint32_t width, uint32_t height) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
}

std::map<int, Frame> readFlight(const std::filesystem::path& flight) {
    std::ifstream cameras(flight / "cameras_classical.csv");
    std::string line;
    std::getline(cameras, line);
    std::map<int, Frame> frames;
    while (std::getline(cameras, line)) {
        std::vector<float> v;
        std::stringstream fields(line);
        for (std::string field; std::getline(fields, field, ',');)
            v.push_back(std::stof(field));
        if (v.size() < 11)
            continue;
        Frame frame;
        frame.index = int(v[0]);
        frame.eye = glm::vec3(v[1], v[2], v[3]);
        frame.look = glm::vec3(v[4], v[5], v[6]);
        frame.nearPlane = v[7];
        frame.fov = v[8];
        frame.depthFar = v[10];
        char name[64];
        std::snprintf(name, sizeof(name), "classical_%04d.ppm", frame.index);
        frame.colour = readPpm(flight / "frames" / name, frame.width, frame.height);
        uint32_t w = 0, h = 0;
        std::snprintf(name, sizeof(name), "classical_depth16_%04d.ppm", frame.index);
        frame.depth16 = readPpm(flight / "frames" / name, w, h);
        if (frame.colour.empty() || frame.depth16.empty())
            continue;
        // As the renderers set the camera: Vulkan's clip space with y flipped.
        glm::mat4 proj = glm::perspective(glm::radians(frame.fov), float(frame.width) / float(frame.height),
                                          frame.nearPlane, 100.0f);
        proj[1][1] *= -1.0f;
        frame.viewProj = proj * glm::lookAt(frame.eye, frame.look, glm::vec3(0.0f, 1.0f, 0.0f));
        frames[frame.index] = std::move(frame);
    }
    return frames;
}

// For every pixel of frame b: where frame a sees its point (pixel coordinates in a, centres at i + 0.5) and whether it
// does (inside a, a's own distance there agreeing within 2 %).
struct Flow {
    std::vector<float> x, y;
    std::vector<uint8_t> valid;
};

Flow flowOnto(const Frame& a, const Frame& b) {
    const size_t pixels = size_t(b.width) * b.height;
    Flow flow{std::vector<float>(pixels), std::vector<float>(pixels), std::vector<uint8_t>(pixels, 0)};
    const glm::mat4 toWorldB = glm::inverse(b.viewProj);
    for (uint32_t row = 0; row < b.height; ++row)
        for (uint32_t column = 0; column < b.width; ++column) {
            const size_t p = size_t(row) * b.width + column;
            const float distanceB = b.distance(p);
            if (distanceB <= 0.0f)
                continue;
            const glm::vec4 far = toWorldB * glm::vec4((column + 0.5f) / b.width * 2.0f - 1.0f,
                                                       (row + 0.5f) / b.height * 2.0f - 1.0f, 1.0f, 1.0f);
            const glm::vec3 point = b.eye + glm::normalize(glm::vec3(far) / far.w - b.eye) * distanceB;
            const glm::vec4 clip = a.viewProj * glm::vec4(point, 1.0f);
            if (clip.w <= 0.0f)
                continue;
            const float x = (clip.x / clip.w + 1.0f) * 0.5f * a.width, y = (clip.y / clip.w + 1.0f) * 0.5f * a.height;
            flow.x[p] = x;
            flow.y[p] = y;
            if (x < 0.0f || y < 0.0f || x >= float(a.width) || y >= float(a.height))
                continue;
            const float seen = a.distance(size_t(y) * a.width + size_t(x)), expected = glm::length(point - a.eye);
            flow.valid[p] = seen > 0.0f && std::abs(seen - expected) < 0.02f * expected;
        }
    return flow;
}

// Fills the pixels without a value (`known` 0) from their known neighbours (the mean of the known 3 x 3 neighbours,
// repeated `rounds` times), so that gaps of a few pixels (sporadic occlusion-test failures in foliage) do not leave
// single rendering pixels between the photo's; larger regions keep `known` 0.
void fillGaps(std::vector<uint8_t>& image, std::vector<uint8_t>& known, uint32_t width, uint32_t height, int rounds) {
    for (int round = 0; round < rounds; ++round) {
        auto nextImage = image;
        auto nextKnown = known;
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                const size_t p = size_t(y) * width + x;
                if (known[p])
                    continue;
                float sum[3] = {0, 0, 0};
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = int(x) + dx, yy = int(y) + dy;
                        if (xx < 0 || yy < 0 || xx >= int(width) || yy >= int(height))
                            continue;
                        const size_t q = size_t(yy) * width + size_t(xx);
                        if (!known[q])
                            continue;
                        for (size_t c = 0; c < 3; ++c)
                            sum[c] += image[3 * q + c];
                        ++count;
                    }
                // Only gaps: at least half of the neighbours known.
                if (count < 4)
                    continue;
                for (size_t c = 0; c < 3; ++c)
                    nextImage[3 * p + c] = uint8_t(std::lround(sum[c] / float(count)));
                nextKnown[p] = 1;
            }
        image = std::move(nextImage);
        known = std::move(nextKnown);
    }
}

// Image a warped onto frame b (bilinear) where the flow is valid, `fallback` elsewhere.
std::vector<uint8_t> warp(const std::vector<uint8_t>& imageA, const Frame& a, const Frame& b, const Flow& flow,
                          const std::vector<uint8_t>& fallback, size_t* taken = nullptr) {
    std::vector<uint8_t> out = fallback;
    size_t count = 0;
    for (size_t p = 0; p < size_t(b.width) * b.height; ++p) {
        if (!flow.valid[p])
            continue;
        const float fx = std::clamp(flow.x[p] - 0.5f, 0.0f, float(a.width - 1)),
                    fy = std::clamp(flow.y[p] - 0.5f, 0.0f, float(a.height - 1));
        const uint32_t x0 = uint32_t(fx), y0 = uint32_t(fy), x1 = std::min(x0 + 1, a.width - 1),
                       y1 = std::min(y0 + 1, a.height - 1);
        const float tx = fx - float(x0), ty = fy - float(y0);
        for (size_t c = 0; c < 3; ++c) {
            const auto at = [&](uint32_t x, uint32_t y) { return float(imageA[3 * (size_t(y) * a.width + x) + c]); };
            const float v =
                (1 - ty) * ((1 - tx) * at(x0, y0) + tx * at(x1, y0)) + ty * ((1 - tx) * at(x0, y1) + tx * at(x1, y1));
            out[3 * p + c] = uint8_t(std::lround(std::clamp(v, 0.0f, 255.0f)));
        }
        ++count;
    }
    if (taken)
        *taken = count;
    return out;
}

double maskedDifference(const std::vector<uint8_t>& x, const std::vector<uint8_t>& y, const Flow& flow) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t p = 0; p < flow.valid.size(); ++p) {
        if (!flow.valid[p])
            continue;
        for (size_t c = 0; c < 3; ++c) {
            const double d = (double(x[3 * p + c]) - double(y[3 * p + c])) / 255.0;
            sum += d * d;
        }
        ++count;
    }
    return count ? sum / double(3 * count) : -1.0;
}

// Qwen's depth control: the inverse distance stretched to the nearest point, blurred (as crate_gbuffer_example's).
std::vector<uint8_t> depthControl(const Frame& f) {
    const size_t pixels = size_t(f.width) * f.height;
    float nearest = 1e9f;
    for (size_t p = 0; p < pixels; ++p)
        if (f.distance(p) > 0.0f)
            nearest = std::min(nearest, f.distance(p));
    std::vector<float> grey(pixels, 0.0f), blurred(pixels, 0.0f);
    for (size_t p = 0; p < pixels; ++p)
        if (f.distance(p) > 0.0f)
            grey[p] = std::round(std::clamp(nearest / f.distance(p), 0.0f, 1.0f) * 255.0f);
    float weights[7], sum = 0.0f;
    for (int k = -3; k <= 3; ++k)
        sum += weights[k + 3] = std::exp(-0.5f * float(k * k));
    for (float& w : weights)
        w /= sum;
    const auto cx = [&](int i) { return uint32_t(std::clamp(i, 0, int(f.width) - 1)); };
    const auto cy = [&](int i) { return uint32_t(std::clamp(i, 0, int(f.height) - 1)); };
    for (uint32_t y = 0; y < f.height; ++y)
        for (uint32_t x = 0; x < f.width; ++x) {
            float v = 0.0f;
            for (int k = -3; k <= 3; ++k)
                v += weights[k + 3] * grey[size_t(y) * f.width + cx(int(x) + k)];
            blurred[size_t(y) * f.width + x] = v;
        }
    std::vector<uint8_t> out(3 * pixels);
    for (uint32_t y = 0; y < f.height; ++y)
        for (uint32_t x = 0; x < f.width; ++x) {
            float v = 0.0f;
            for (int k = -3; k <= 3; ++k)
                v += weights[k + 3] * blurred[size_t(cy(int(y) + k)) * f.width + x];
            for (size_t c = 0; c < 3; ++c)
                out[3 * (size_t(y) * f.width + x) + c] = uint8_t(std::lround(std::clamp(v, 0.0f, 255.0f)));
        }
    return out;
}

std::string photoName(int frame) {
    char name[32];
    std::snprintf(name, sizeof(name), "%04d.ppm", frame);
    return name;
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path flight, photos, out;
    std::string mode;
    int frame = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--check" && i + 1 < argc) {
            mode = "check";
            flight = argv[++i];
        } else if (arg == "--start" && i + 4 < argc) {
            mode = "start";
            flight = argv[++i];
            frame = std::stoi(argv[++i]);
            photos = argv[++i];
            out = argv[++i];
        } else if (arg == "--consistency" && i + 2 < argc) {
            mode = "consistency";
            flight = argv[++i];
            photos = argv[++i];
        } else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    const auto frames = readFlight(flight);
    if (frames.size() < 2) {
        std::cerr << "flight_paint_example: no frames in " << flight << std::endl;
        return 1;
    }
    if (mode == "check" || mode == "consistency") {
        double sum = 0.0, unwarped = 0.0, valid = 0.0;
        size_t pairs = 0;
        for (auto it = frames.begin(); std::next(it) != frames.end(); ++it) {
            const Frame &a = it->second, &b = std::next(it)->second;
            std::vector<uint8_t> imageA = a.colour, imageB = b.colour;
            if (mode == "consistency") {
                uint32_t w = 0, h = 0;
                imageA = readPpm(photos / photoName(a.index), w, h);
                imageB = readPpm(photos / photoName(b.index), w, h);
                if (imageA.empty() || imageB.empty())
                    continue;
            }
            const Flow flow = flowOnto(a, b);
            size_t taken = 0;
            const auto warped = warp(imageA, a, b, flow, imageB, &taken);
            Flow identity{flow.x, flow.y, flow.valid};
            sum += maskedDifference(warped, imageB, flow);
            unwarped += maskedDifference(imageA, imageB, identity);
            valid += double(taken) / double(b.width * b.height);
            ++pairs;
        }
        std::printf("%s over %zu consecutive pairs: masked MSE %.5f warped, %.5f without the warp, %.1f %% valid\n",
                    mode.c_str(), pairs, sum / double(pairs), unwarped / double(pairs), 100.0 * valid / double(pairs));
        return 0;
    }
    if (mode == "start") {
        const auto it = frames.find(frame);
        if (it == frames.end())
            return 1;
        const Frame& b = it->second;
        std::filesystem::create_directories(out);
        std::vector<uint8_t> start = b.colour;
        size_t taken = 0;
        if (it != frames.begin()) {
            const Frame& a = std::prev(it)->second;
            uint32_t w = 0, h = 0;
            const auto photo = readPpm(photos / photoName(a.index), w, h);
            if (!photo.empty()) {
                const Flow flow = flowOnto(a, b);
                start = warp(photo, a, b, flow, b.colour, &taken);
                std::vector<uint8_t> known = flow.valid;
                fillGaps(start, known, b.width, b.height, 4);
            }
        }
        writePpm(out / ("start_" + photoName(frame)), start, b.width, b.height);
        writePpm(out / ("control_" + photoName(frame)), depthControl(b), b.width, b.height);
        std::printf("frame %d: %.1f %% from the previous photo\n", frame,
                    100.0 * double(taken) / double(b.width * b.height));
        return 0;
    }
    return 1;
}
