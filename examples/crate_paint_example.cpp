// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Projective painting of the crate (as TEXTure paints meshes): a UV atlas of the crate's six faces, filled photo by
// photo, so that Qwen-Image photos of different views show the same crate. Views come from flights of
// crate_gbuffer_example (cameras_classical.csv); photos are named <flight>_<frame>.ppm.
//
//   --start FLIGHT OUT    every view's start image for image-to-image: the atlas where it is painted, elsewhere the
//                         crate rendered on white; writes OUT/start/<id>.ppm, OUT/control/<id>.ppm (Qwen's depth
//                         control) and OUT/batch.txt (lines "start photo control", photos to OUT/photo/<id>.ppm)
//   --bake FLIGHT PHOTOS  bakes the flight's photos into the atlas: a texel takes a photo's colour where the photo sees
//                         it more frontally than the view that painted it so far (the crate's outline band left out)
//   --views a,b,...       only these frames of the flight
//   --consistency FLIGHT PHOTOS  how well the flight's photos agree: each against its nearest other view, warped onto
//                         it by the geometric flow, mean squared difference on the crate
//   --self-test           bakes a synthetic flight (each face its own colour) and checks the start image of a new view
//
//   crate_paint_example --atlas build/TestingOutput/crate_painted/atlas --bake FLIGHT PHOTO_DIR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
#include "klartraum/offscreen_target.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kSize = 256;   // photos, start images and G-buffers
constexpr uint32_t kTexels = 128; // per face side

// The atlas: per face kTexels x kTexels colours (RGB in [0, 1]) and how frontally the painting view saw each texel
// (0: not painted).
struct Atlas {
    std::vector<float> colour = std::vector<float>(6 * kTexels * kTexels * 3, 0.0f);
    std::vector<float> confidence = std::vector<float>(6 * kTexels * kTexels, 0.0f);

    size_t texel(uint32_t face, uint32_t x, uint32_t y) const { return (size_t(face) * kTexels + y) * kTexels + x; }

    // The atlas with its holes filled: unpainted texels take the mean of their painted neighbours, repeated
    // `rounds` times (oblique views paint only every few texels). Confidences stay as painted.
    Atlas filled(int rounds) const {
        Atlas result = *this;
        for (int round = 0; round < rounds; ++round) {
            Atlas next = result;
            for (uint32_t face = 0; face < 6; ++face)
                for (uint32_t y = 0; y < kTexels; ++y)
                    for (uint32_t x = 0; x < kTexels; ++x) {
                        const size_t t = texel(face, x, y);
                        if (result.confidence[t] > 0.0f)
                            continue;
                        float sum[3] = {0, 0, 0}, count = 0;
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int xx = int(x) + dx, yy = int(y) + dy;
                                if (xx < 0 || yy < 0 || xx >= int(kTexels) || yy >= int(kTexels))
                                    continue;
                                const size_t n = texel(face, uint32_t(xx), uint32_t(yy));
                                if (result.confidence[n] <= 0.0f)
                                    continue;
                                for (size_t c = 0; c < 3; ++c)
                                    sum[c] += result.colour[n * 3 + c];
                                count += 1;
                            }
                        if (count < 1)
                            continue;
                        for (size_t c = 0; c < 3; ++c)
                            next.colour[t * 3 + c] = sum[c] / count;
                        next.confidence[t] = 1e-3f; // filled, not painted
                    }
            result = std::move(next);
        }
        return result;
    }

    void load(const std::filesystem::path& directory) {
        std::ifstream file(directory / "atlas.bin", std::ios::binary);
        if (!file)
            return;
        file.read(reinterpret_cast<char*>(colour.data()), std::streamsize(colour.size() * sizeof(float)));
        file.read(reinterpret_cast<char*>(confidence.data()), std::streamsize(confidence.size() * sizeof(float)));
    }

    // atlas.bin and atlas.png (the faces +x, -x, +y in the first row, -y, +z, -z in the second; unpainted grey).
    void save(const std::filesystem::path& directory) const {
        std::filesystem::create_directories(directory);
        std::ofstream file(directory / "atlas.bin", std::ios::binary);
        file.write(reinterpret_cast<const char*>(colour.data()), std::streamsize(colour.size() * sizeof(float)));
        file.write(reinterpret_cast<const char*>(confidence.data()),
                   std::streamsize(confidence.size() * sizeof(float)));
        const uint32_t width = 3 * kTexels, height = 2 * kTexels;
        std::vector<uint8_t> image(size_t(width) * height * 3, 128);
        for (uint32_t face = 0; face < 6; ++face)
            for (uint32_t y = 0; y < kTexels; ++y)
                for (uint32_t x = 0; x < kTexels; ++x) {
                    const size_t t = texel(face, x, y);
                    if (confidence[t] <= 0.0f)
                        continue;
                    // v grows upwards: the image's rows run downwards.
                    const size_t px = (face % 3) * kTexels + x, py = (face / 3) * kTexels + (kTexels - 1 - y);
                    for (size_t c = 0; c < 3; ++c)
                        image[(py * width + px) * 3 + c] =
                            uint8_t(std::lround(std::clamp(colour[t * 3 + c], 0.0f, 1.0f) * 255.0f));
                }
        writeImage(directory / "atlas.png", image.data(), width, height, 3);
    }
};

// One view's G-buffers at kSize: per pixel whether it shows the crate, its face and texel, and how frontally the
// camera sees it; the rendering on white; the depth control image.
struct ViewBuffers {
    std::vector<int> face;             // -1 off the crate
    std::vector<uint32_t> tx, ty;      // the texel
    std::vector<float> facing;         // cos of the angle between the normal and the direction to the camera
    std::vector<uint8_t> interior;     // 1 on the crate at least 3 pixels from its outline
    std::vector<uint8_t> onWhite;      // RGB, the lit crate on white
    std::vector<uint8_t> depthControl; // RGB, Qwen's depth control (near bright, blurred)
    std::vector<uint8_t> depth16;      // BGRA, the distance as crate::flow() takes it
    CameraMVP camera{};
};

// A flight's cameras: frame index, eye, look.
struct Camera {
    int frame;
    glm::vec3 eye, look;
};

std::vector<Camera> readFlight(const std::filesystem::path& flight) {
    std::ifstream file(flight / "cameras_classical.csv");
    std::string line;
    std::getline(file, line);
    std::vector<Camera> cameras;
    while (std::getline(file, line)) {
        std::vector<float> v;
        std::stringstream fields(line);
        for (std::string field; std::getline(fields, field, ',');)
            v.push_back(std::stof(field));
        if (v.size() >= 7)
            cameras.push_back({int(v[0]), glm::vec3(v[1], v[2], v[3]), glm::vec3(v[4], v[5], v[6])});
    }
    return cameras;
}

class Painter {
public:
    explicit Painter(VulkanContext& vc)
        : vc(vc) {
        const VkExtent2D extent{kSize, kSize};
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        renderer = std::make_unique<GBufferMeshRenderer>(vc, crate::vertices(false), target, extent);
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(renderer->getOutput());
    }

    ViewBuffers render(const Camera& camera) {
        CameraMVP mvp{};
        crate::setCamera(mvp, camera.eye, camera.look);
        renderer->getCamera()->ubo = mvp;
        renderer->getCamera()->update(0);
        const auto u = draw(GBufferOutput::CoordinatesU), v = draw(GBufferOutput::CoordinatesV);
        const auto classes = draw(GBufferOutput::Classes), normals = draw(GBufferOutput::Normals);
        const auto colour = draw(GBufferOutput::Colour), depth = draw(GBufferOutput::Depth16);
        const size_t pixels = size_t(kSize) * kSize;
        ViewBuffers view;
        view.depth16 = depth;
        view.camera = mvp;
        view.face.assign(pixels, -1);
        view.tx.assign(pixels, 0);
        view.ty.assign(pixels, 0);
        view.facing.assign(pixels, 0.0f);
        view.interior.assign(pixels, 0);
        view.onWhite.assign(3 * pixels, 255);
        view.depthControl.assign(3 * pixels, 0);
        const glm::mat4 toWorld = glm::inverse(mvp.proj * mvp.view);
        const auto value16 = [](const std::vector<uint8_t>& bgra, size_t p) {
            return float(bgra[4 * p + 2] * 256u + bgra[4 * p + 1]) / 65535.0f;
        };
        std::vector<uint8_t> crateMask(pixels, 0);
        float nearest = 1e9f;
        for (size_t p = 0; p < pixels; ++p) {
            // The crate's class colour is red (tree::TREE_CLASS_STEM); the floor is not drawn.
            const bool crate = classes[4 * p + 2] > 127 && classes[4 * p + 1] < 64;
            if (!crate)
                continue;
            crateMask[p] = 1;
            const uint32_t id = uint32_t(u[4 * p]) * 256u + v[4 * p];
            view.face[p] = std::clamp(int(std::lround(float(id) / 65535.0f * 6.0f - 0.5f)), 0, 5);
            view.tx[p] = std::min(uint32_t(value16(u, p) * kTexels), kTexels - 1);
            view.ty[p] = std::min(uint32_t(value16(v, p) * kTexels), kTexels - 1);
            const glm::vec3 n = glm::vec3(normals[4 * p + 2], normals[4 * p + 1], normals[4 * p]) / 127.5f - 1.0f;
            const uint32_t column = uint32_t(p % kSize), row = uint32_t(p / kSize);
            const glm::vec4 far = toWorld * glm::vec4((column + 0.5f) / kSize * 2.0f - 1.0f,
                                                      (row + 0.5f) / kSize * 2.0f - 1.0f, 1.0f, 1.0f);
            const glm::vec3 ray = glm::normalize(glm::vec3(far) / far.w - camera.eye);
            view.facing[p] = std::max(0.0f, -glm::dot(glm::normalize(n), ray));
            for (size_t c = 0; c < 3; ++c)
                view.onWhite[3 * p + c] = colour[4 * p + 2 - c];
            nearest = std::min(nearest, value16(depth, p));
        }
        // 3 pixels inside the outline.
        for (size_t p = 0; p < pixels; ++p) {
            if (!crateMask[p])
                continue;
            const int x = int(p % kSize), y = int(p / kSize);
            bool inside = true;
            for (int dy = -3; dy <= 3 && inside; ++dy)
                for (int dx = -3; dx <= 3 && inside; ++dx) {
                    const int xx = x + dx, yy = y + dy;
                    inside = xx >= 0 && yy >= 0 && xx < int(kSize) && yy < int(kSize) &&
                             crateMask[size_t(yy) * kSize + size_t(xx)];
                }
            view.interior[p] = inside;
        }
        // The depth control as crate_gbuffer_example's: inverse depth stretched to the nearest point, blurred.
        std::vector<float> grey(pixels, 0.0f), blurred(pixels, 0.0f);
        for (size_t p = 0; p < pixels; ++p)
            if (crateMask[p])
                grey[p] = std::round(std::clamp(nearest / std::max(value16(depth, p), 1e-6f), 0.0f, 1.0f) * 255.0f);
        float weights[7], sum = 0.0f;
        for (int k = -3; k <= 3; ++k)
            sum += weights[k + 3] = std::exp(-0.5f * float(k * k));
        for (float& w : weights)
            w /= sum;
        const auto clampIndex = [](int i) { return uint32_t(std::clamp(i, 0, int(kSize) - 1)); };
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                float value = 0.0f;
                for (int k = -3; k <= 3; ++k)
                    value += weights[k + 3] * grey[size_t(y) * kSize + clampIndex(int(x) + k)];
                blurred[size_t(y) * kSize + x] = value;
            }
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                float value = 0.0f;
                for (int k = -3; k <= 3; ++k)
                    value += weights[k + 3] * blurred[size_t(clampIndex(int(y) + k)) * kSize + x];
                for (size_t c = 0; c < 3; ++c)
                    view.depthControl[3 * (size_t(y) * kSize + x) + c] =
                        uint8_t(std::lround(std::clamp(value, 0.0f, 255.0f)));
            }
        return view;
    }

private:
    std::vector<uint8_t> draw(GBufferOutput output) {
        renderer->setOutput(output, 0.0f, crate::kDepthFar);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return crate::readImage(vc, target->getImage(0), VkExtent2D{kSize, kSize});
    }

    VulkanContext& vc;
    std::shared_ptr<OffscreenTarget> target;
    std::unique_ptr<GBufferMeshRenderer> renderer;
    std::unique_ptr<ComputeGraph> graph;
};

// The crate's surface point of every texel: per face its two triangles (crate::vertices()), the texel centre's (u, v)
// located in one of them and its position interpolated; NaN where no triangle covers it.
struct TexelPoints {
    std::vector<glm::vec3> point = std::vector<glm::vec3>(6 * kTexels * kTexels, glm::vec3(NAN));
    std::array<glm::vec3, 6> normal{};
};

TexelPoints texelPoints() {
    TexelPoints result;
    const auto vertices = crate::vertices(false);
    for (size_t k = 0; k + 2 < vertices.size(); k += 3) {
        const auto& a = vertices[k];
        const auto& b = vertices[k + 1];
        const auto& c = vertices[k + 2];
        const int face =
            std::clamp(int(std::lround(float(uint32_t(a.normal.w + 0.5f)) / 65535.0f * 6.0f - 0.5f)), 0, 5);
        result.normal[size_t(face)] = glm::vec3(a.normal);
        const glm::vec2 ua(a.uv), ub(b.uv), uc(c.uv);
        const float det = (ub.x - ua.x) * (uc.y - ua.y) - (uc.x - ua.x) * (ub.y - ua.y);
        if (std::abs(det) < 1e-12f)
            continue;
        for (uint32_t y = 0; y < kTexels; ++y)
            for (uint32_t x = 0; x < kTexels; ++x) {
                const glm::vec2 q((x + 0.5f) / kTexels, (y + 0.5f) / kTexels);
                const float l1 = ((ub.x - q.x) * (uc.y - q.y) - (uc.x - q.x) * (ub.y - q.y)) / det;
                const float l2 = ((uc.x - q.x) * (ua.y - q.y) - (ua.x - q.x) * (uc.y - q.y)) / det;
                const float l3 = 1.0f - l1 - l2;
                if (l1 < -1e-4f || l2 < -1e-4f || l3 < -1e-4f)
                    continue;
                result.point[(size_t(face) * kTexels + y) * kTexels + x] =
                    l1 * glm::vec3(a.pos) + l2 * glm::vec3(b.pos) + l3 * glm::vec3(c.pos);
            }
    }
    return result;
}

// Bakes `photo` (RGB bytes, kSize x kSize) into the atlas from the texels' side: every texel's surface point projected
// into the view, taken where the view sees it (its distance agrees with the view's depth, the pixel lies in the
// crate's interior) more frontally than the texel's painter so far; the colour sampled bilinearly. Returns the texels
// painted.
size_t bake(Atlas& atlas, const TexelPoints& points, const ViewBuffers& view, const std::vector<uint8_t>& photo,
            glm::vec3 eye) {
    const glm::mat4 toClip = view.camera.proj * view.camera.view;
    const auto distance16 = [&](uint32_t x, uint32_t y) {
        const size_t p = size_t(y) * kSize + x;
        return float(view.depth16[4 * p + 2] * 256u + view.depth16[4 * p + 1]) / 65535.0f * crate::kDepthFar;
    };
    size_t painted = 0;
    for (uint32_t face = 0; face < 6; ++face) {
        for (size_t t = size_t(face) * kTexels * kTexels; t < size_t(face + 1) * kTexels * kTexels; ++t) {
            const glm::vec3 point = points.point[t];
            if (std::isnan(point.x))
                continue;
            const glm::vec3 towards = eye - point;
            const float facing = glm::dot(points.normal[face], glm::normalize(towards));
            // Seen too obliquely, a photo smears the face: such views do not paint.
            if (facing < 0.3f || facing <= atlas.confidence[t] + 0.02f)
                continue;
            const glm::vec4 clip = toClip * glm::vec4(point, 1.0f);
            if (clip.w <= 0.0f)
                continue;
            const float x = (clip.x / clip.w + 1.0f) * 0.5f * kSize, y = (clip.y / clip.w + 1.0f) * 0.5f * kSize;
            if (x < 1.0f || y < 1.0f || x >= kSize - 1.0f || y >= kSize - 1.0f)
                continue;
            const uint32_t px = uint32_t(x), py = uint32_t(y);
            if (!view.interior[size_t(py) * kSize + px])
                continue;
            const float seen = distance16(px, py), expected = glm::length(towards);
            if (std::abs(seen - expected) > 0.02f * expected)
                continue;
            // Bilinear, pixel centres at i + 0.5.
            const float fx = x - 0.5f, fy = y - 0.5f;
            const uint32_t x0 = uint32_t(std::max(fx, 0.0f)), y0 = uint32_t(std::max(fy, 0.0f));
            const uint32_t x1 = std::min(x0 + 1, kSize - 1), y1 = std::min(y0 + 1, kSize - 1);
            const float ax = fx - float(x0), ay = fy - float(y0);
            for (size_t c = 0; c < 3; ++c) {
                const auto at = [&](uint32_t xx, uint32_t yy) {
                    return float(photo[3 * (size_t(yy) * kSize + xx) + c]);
                };
                const float value = (1 - ay) * ((1 - ax) * at(x0, y0) + ax * at(x1, y0)) +
                                    ay * ((1 - ax) * at(x0, y1) + ax * at(x1, y1));
                atlas.colour[t * 3 + c] = value / 255.0f;
            }
            atlas.confidence[t] = facing;
            ++painted;
        }
    }
    return painted;
}

// The view's start image: background white, the crate from the (hole-filled) atlas where painted (the nearest painted
// texel within 2 texels), elsewhere the atlas's mean colour, lightly shaded. Returns the share of the crate's pixels
// taken from the atlas.
float startImage(const Atlas& atlas, const ViewBuffers& view, std::vector<uint8_t>& image) {
    const size_t pixels = size_t(kSize) * kSize;
    image = view.onWhite;
    // Unpainted surface: the painted texels' mean colour, shaded only lightly (a dark, flat face reads as an opening).
    float mean[3] = {0.55f, 0.36f, 0.22f};
    {
        double sum[3] = {0, 0, 0}, count = 0;
        for (size_t t = 0; t < atlas.confidence.size(); ++t)
            if (atlas.confidence[t] > 0.0f) {
                for (size_t c = 0; c < 3; ++c)
                    sum[c] += atlas.colour[t * 3 + c];
                count += 1;
            }
        if (count > 0)
            for (size_t c = 0; c < 3; ++c)
                mean[c] = float(sum[c] / count);
    }
    float brightest = 1.0f;
    for (size_t p = 0; p < pixels; ++p)
        if (view.face[p] >= 0)
            brightest =
                std::max(brightest, float(view.onWhite[3 * p] + view.onWhite[3 * p + 1] + view.onWhite[3 * p + 2]));
    for (size_t p = 0; p < pixels; ++p)
        if (view.face[p] >= 0) {
            const float lit =
                float(view.onWhite[3 * p] + view.onWhite[3 * p + 1] + view.onWhite[3 * p + 2]) / brightest;
            for (size_t c = 0; c < 3; ++c)
                image[3 * p + c] = uint8_t(std::lround(std::clamp(mean[c] * (0.8f + 0.2f * lit), 0.0f, 1.0f) * 255.0f));
        }
    size_t crate = 0, fromAtlas = 0;
    for (size_t p = 0; p < pixels; ++p) {
        if (view.face[p] < 0)
            continue;
        ++crate;
        bool found = false;
        for (int r = 0; r <= 2 && !found; ++r)
            for (int dy = -r; dy <= r && !found; ++dy)
                for (int dx = -r; dx <= r && !found; ++dx) {
                    const int x = int(view.tx[p]) + dx, y = int(view.ty[p]) + dy;
                    if (x < 0 || y < 0 || x >= int(kTexels) || y >= int(kTexels))
                        continue;
                    const size_t t = atlas.texel(uint32_t(view.face[p]), uint32_t(x), uint32_t(y));
                    if (atlas.confidence[t] <= 0.0f)
                        continue;
                    for (size_t c = 0; c < 3; ++c)
                        image[3 * p + c] =
                            uint8_t(std::lround(std::clamp(atlas.colour[t * 3 + c], 0.0f, 1.0f) * 255.0f));
                    found = true;
                }
        fromAtlas += found;
    }
    return crate > 0 ? float(fromAtlas) / float(crate) : 0.0f;
}

// The mean squared difference (RGB in [0, 1]) between photo b and photo a warped onto b's view by the geometric flow,
// over b's crate interior where a sees the same point; -1 if they share too few pixels.
double warpedDifference(const ViewBuffers& a, const std::vector<uint8_t>& photoA, const ViewBuffers& b,
                        const std::vector<uint8_t>& photoB) {
    const auto flow = crate::flow(a.camera, a.depth16, b.camera, b.depth16, kSize);
    double sum = 0.0;
    size_t count = 0;
    for (size_t p = 0; p < size_t(kSize) * kSize; ++p) {
        if (!b.interior[p] || !flow.valid[p])
            continue;
        const size_t q =
            size_t(std::min(flow.y[p], float(kSize) - 1.0f)) * kSize + size_t(std::min(flow.x[p], float(kSize) - 1.0f));
        if (!a.interior[q])
            continue;
        for (size_t c = 0; c < 3; ++c) {
            const double d = (double(photoA[3 * q + c]) - double(photoB[3 * p + c])) / 255.0;
            sum += d * d;
        }
        ++count;
    }
    return count > 500 ? sum / double(3 * count) : -1.0;
}

void writePpm(const std::filesystem::path& path, const std::vector<uint8_t>& rgb) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << kSize << " " << kSize << "\n255\n";
    file.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
}

// A binary PPM of kSize x kSize (photos of larger sizes averaged down), RGB bytes; empty if missing.
std::vector<uint8_t> readPpm(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    uint32_t width = 0, height = 0, maximum = 0;
    if (!(file >> magic >> width >> height >> maximum) || magic != "P6" || width != height || width % kSize != 0)
        return {};
    file.get();
    std::vector<uint8_t> bytes(size_t(width) * height * 3);
    file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    const uint32_t factor = width / kSize;
    std::vector<uint8_t> out(size_t(kSize) * kSize * 3);
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x)
            for (size_t c = 0; c < 3; ++c) {
                uint32_t sum = 0;
                for (uint32_t dy = 0; dy < factor; ++dy)
                    for (uint32_t dx = 0; dx < factor; ++dx)
                        sum += bytes[(size_t(y * factor + dy) * width + x * factor + dx) * 3 + c];
                out[(size_t(y) * kSize + x) * 3 + c] = uint8_t(sum / (factor * factor));
            }
    return out;
}

std::string viewId(const std::filesystem::path& flight, int frame) {
    char id[128];
    std::snprintf(id, sizeof(id), "%s_%04d", flight.filename().string().c_str(), frame);
    return id;
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path atlasDir = "build/TestingOutput/crate_painted/atlas", startFlight, startOut, bakeFlight,
                          bakePhotos;
    std::vector<int> only;
    bool selfTest = false;
    std::filesystem::path consistencyFlight, consistencyPhotos;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--atlas" && i + 1 < argc)
            atlasDir = argv[++i];
        else if (arg == "--start" && i + 2 < argc) {
            startFlight = argv[++i];
            startOut = argv[++i];
        } else if (arg == "--bake" && i + 2 < argc) {
            bakeFlight = argv[++i];
            bakePhotos = argv[++i];
        } else if (arg == "--views" && i + 1 < argc) {
            std::stringstream numbers(argv[++i]);
            for (std::string number; std::getline(numbers, number, ',');)
                only.push_back(std::stoi(number));
        } else if (arg == "--consistency" && i + 2 < argc) {
            consistencyFlight = argv[++i];
            consistencyPhotos = argv[++i];
        } else if (arg == "--self-test")
            selfTest = true;
        else {
            std::cerr << "unknown argument " << arg << std::endl;
            return 1;
        }
    }
    try {
        HeadlessFrontend frontend;
        VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
        Painter painter(vc);
        const TexelPoints points = texelPoints();
        const auto wanted = [&](int frame) { return only.empty() || std::count(only.begin(), only.end(), frame) > 0; };

        if (selfTest) {
            // Each face painted in its own colour from a view that sees it; a new view must show each face's colour.
            Atlas atlas;
            const std::array<std::array<uint8_t, 3>, 6> colours{
                {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255}, {255, 0, 255}}};
            const glm::vec3 look(0.0f, 0.5f, 0.0f);
            // Four side views and one from above (a view paints only faces it sees at a cosine of 0.3 or more).
            for (const glm::vec3 eye : {glm::vec3(3, 2, 2), glm::vec3(-3, 2, -2), glm::vec3(2, 2, -3),
                                        glm::vec3(-2, 2, 3), glm::vec3(0.4f, 4.0f, 0.6f)}) {
                const auto view = painter.render({0, eye, look});
                std::vector<uint8_t> photo(3 * kSize * kSize, 255);
                for (size_t p = 0; p < size_t(kSize) * kSize; ++p)
                    if (view.face[p] >= 0)
                        for (size_t c = 0; c < 3; ++c)
                            photo[3 * p + c] = colours[size_t(view.face[p])][c];
                bake(atlas, points, view, photo, eye);
            }
            const auto check = painter.render({0, glm::vec3(2.5f, 1.8f, 2.5f), look});
            std::vector<uint8_t> image;
            const float share = startImage(atlas, check, image);
            size_t right = 0, total = 0;
            for (size_t p = 0; p < size_t(kSize) * kSize; ++p)
                if (check.face[p] >= 0 && check.interior[p]) {
                    ++total;
                    const auto& want = colours[size_t(check.face[p])];
                    // Bilinear sampling blends neighbouring faces' colours at the faces' borders a little.
                    int off = 0;
                    for (size_t c = 0; c < 3; ++c)
                        off = std::max(off, std::abs(int(image[3 * p + c]) - int(want[c])));
                    right += off <= 64;
                }
            std::printf("self-test: %.1f %% of the crate from the atlas, %zu of %zu interior pixels show their face's "
                        "colour (within 64 of 255)\n",
                        100.0f * share, right, total);
            return share > 0.95f && right >= total * 98 / 100 ? 0 : 1;
        }

        // How well the photos of a flight agree: every photo against its nearest other view (by camera position),
        // warped onto it.
        if (!consistencyFlight.empty()) {
            std::vector<Camera> cameras;
            std::vector<ViewBuffers> views;
            std::vector<std::vector<uint8_t>> photos;
            for (const auto& camera : readFlight(consistencyFlight)) {
                auto photo = readPpm(consistencyPhotos / (viewId(consistencyFlight, camera.frame) + ".ppm"));
                if (photo.empty() || !wanted(camera.frame))
                    continue;
                cameras.push_back(camera);
                views.push_back(painter.render(camera));
                photos.push_back(std::move(photo));
            }
            double sum = 0.0, angles = 0.0;
            size_t pairs = 0;
            for (size_t i = 0; i < cameras.size(); ++i) {
                size_t best = i;
                float bestDistance = 1e9f;
                for (size_t j = 0; j < cameras.size(); ++j)
                    if (j != i && glm::length(cameras[j].eye - cameras[i].eye) < bestDistance) {
                        bestDistance = glm::length(cameras[j].eye - cameras[i].eye);
                        best = j;
                    }
                if (best == i)
                    continue;
                const double d = warpedDifference(views[best], photos[best], views[i], photos[i]);
                if (d < 0.0)
                    continue;
                sum += d;
                angles +=
                    glm::degrees(std::acos(std::clamp(glm::dot(glm::normalize(cameras[i].eye - cameras[i].look),
                                                               glm::normalize(cameras[best].eye - cameras[best].look)),
                                                      -1.0f, 1.0f)));
                ++pairs;
            }
            std::printf("consistency of %s: warped MSE %.5f over %zu nearest-neighbour pairs (%.1f degrees apart on "
                        "average)\n",
                        consistencyFlight.string().c_str(), pairs ? sum / double(pairs) : 0.0, pairs,
                        pairs ? angles / double(pairs) : 0.0);
            return 0;
        }

        Atlas atlas;
        atlas.load(atlasDir);
        if (!bakeFlight.empty()) {
            for (const auto& camera : readFlight(bakeFlight)) {
                if (!wanted(camera.frame))
                    continue;
                const auto id = viewId(bakeFlight, camera.frame);
                const auto photo = readPpm(bakePhotos / (id + ".ppm"));
                if (photo.empty()) {
                    std::cerr << "no photo " << id << std::endl;
                    continue;
                }
                const size_t painted = bake(atlas, points, painter.render(camera), photo, camera.eye);
                std::cout << id << ": " << painted << " texels painted" << std::endl;
            }
            atlas.save(atlasDir);
            size_t total = 0;
            for (float c : atlas.confidence)
                total += c > 0.0f;
            std::printf("atlas: %.1f %% of the texels painted\n",
                        100.0 * double(total) / double(atlas.confidence.size()));
        }
        if (!startFlight.empty()) {
            for (const char* sub : {"start", "control", "photo"})
                std::filesystem::create_directories(startOut / sub);
            std::ofstream batch(startOut / "batch.txt", std::ios::app);
            const Atlas holeFilled = atlas.filled(6);
            for (const auto& camera : readFlight(startFlight)) {
                if (!wanted(camera.frame))
                    continue;
                const auto id = viewId(startFlight, camera.frame);
                const auto view = painter.render(camera);
                std::vector<uint8_t> image;
                const float share = startImage(holeFilled, view, image);
                writePpm(startOut / "start" / (id + ".ppm"), image);
                writePpm(startOut / "control" / (id + ".ppm"), view.depthControl);
                batch << (startOut / "start" / (id + ".ppm")).string() << " "
                      << (startOut / "photo" / (id + ".ppm")).string() << " "
                      << (startOut / "control" / (id + ".ppm")).string() << "\n";
                std::printf("%s: %.1f %% of the crate from the atlas\n", id.c_str(), 100.0f * share);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "crate_paint_example: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
