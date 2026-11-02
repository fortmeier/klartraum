// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/fitting_scene.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "klartraum/raster/mesh.hpp"

namespace klartraum::raster {

namespace {

std::vector<float> readFloats(const std::filesystem::path& path, size_t count) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("loadFittingScene: cannot open " + path.string());
    if (size_t(in.tellg()) != count * sizeof(float))
        throw std::runtime_error("loadFittingScene: " + path.string() + " does not hold " + std::to_string(count) +
                                 " floats");
    std::vector<float> values(count);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(float)));
    return values;
}

void writeFloats(const std::filesystem::path& path, const float* values, size_t count) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("writeFittingScene: cannot open " + path.string());
    out.write(reinterpret_cast<const char*>(values), std::streamsize(count * sizeof(float)));
}

} // namespace

std::vector<float> FittingScene::targetsOf(const std::vector<uint32_t>& indices) const {
    const size_t viewSize = 4 * size_t(width) * height;
    std::vector<float> values;
    values.reserve(viewSize * indices.size());
    for (const uint32_t view : indices)
        values.insert(values.end(), targets.begin() + view * viewSize, targets.begin() + (view + 1) * viewSize);
    return values;
}

std::vector<glm::mat4> FittingScene::camerasOf(const std::vector<uint32_t>& indices) const {
    std::vector<glm::mat4> result;
    for (const uint32_t view : indices)
        result.push_back(cameras.at(view));
    return result;
}

glm::mat4 FittingScene::sceneFromUnitSphere() const {
    glm::vec3 up(0.0f), unseen(0.0f);
    std::vector<glm::vec3> offsets;
    for (const auto& camera : cameras) {
        offsets.push_back(cameraPosition(camera) - centre);
        up += glm::normalize(offsets.back());
    }
    for (const auto& direction : icosphere(3).positions) {
        bool seen = false;
        for (const auto& offset : offsets)
            seen = seen || glm::dot(direction, offset) > radius;
        if (!seen)
            unseen += direction;
    }
    up = glm::normalize(glm::length(unseen) > 0.0f ? -unseen : up);
    const glm::vec3 across =
        glm::normalize(glm::cross(up, std::abs(up.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1)));
    const glm::mat4 rotation(glm::vec4(across, 0.0f), glm::vec4(up, 0.0f), glm::vec4(glm::cross(across, up), 0.0f),
                             glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    return glm::translate(glm::mat4(1.0f), centre) * rotation * glm::scale(glm::mat4(1.0f), glm::vec3(radius));
}

glm::vec3 cameraPosition(const glm::mat4& clipFromWorld) {
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

DisplayView displayView(const FittingScene& scene, uint32_t view) {
    const glm::mat4 sceneFromUnit = scene.sceneFromUnitSphere();
    const glm::mat4 camera = scene.cameras.at(view) * sceneFromUnit;
    const glm::vec3 up(0.0f, 1.0f, 0.0f), seen = cameraPosition(camera);
    const glm::vec3 side = glm::normalize(seen - glm::dot(seen, up) * up);
    const float elevation = 0.3f;
    glm::mat4 projection = glm::perspectiveRH_ZO(0.8f, 1.0f, 0.1f, 20.0f);
    projection[1][1] *= -1.0f; // Vulkan: y down
    DisplayView result;
    result.camera =
        projection * glm::lookAtRH(3.5f * (std::cos(elevation) * side + std::sin(elevation) * up), glm::vec3(0.0f), up);
    // Quarter turns that make the up direction point up on the screen.
    const glm::vec4 a = camera * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), b = camera * glm::vec4(0.1f * up, 1.0f);
    glm::vec2 direction = glm::vec2(b) / b.w - glm::vec2(a) / a.w; // y down, like the screen
    int turns = 0;
    float best = -2.0f;
    for (int k = 0; k < 4; ++k) {
        if (-glm::normalize(direction).y > best) {
            best = -glm::normalize(direction).y;
            turns = k;
        }
        direction = glm::vec2(-direction.y, direction.x); // a clockwise quarter turn on the screen
    }
    std::vector<float> photo = scene.targetsOf({view});
    uint32_t width = scene.width, height = scene.height;
    for (; turns > 0; --turns) { // turn clockwise
        std::vector<float> turned(photo.size());
        for (uint32_t y = 0; y < width; ++y)
            for (uint32_t x = 0; x < height; ++x)
                for (int c = 0; c < 4; ++c)
                    turned[4 * (size_t(y) * height + x) + c] = photo[4 * (size_t(height - 1 - x) * width + y) + c];
        photo.swap(turned);
        std::swap(width, height);
    }
    result.side = std::max(width, height);
    result.photo.assign(4 * size_t(result.side) * result.side, 0.0f);
    const uint32_t left = (result.side - width) / 2, top = (result.side - height) / 2;
    for (uint32_t y = 0; y < height; ++y)
        std::copy_n(&photo[4 * size_t(y) * width], 4 * width,
                    &result.photo[4 * (size_t(y + top) * result.side + left)]);
    return result;
}

FittingScene loadFittingScene(const std::filesystem::path& directory) {
    std::ifstream text(directory / "scene.txt");
    if (!text)
        throw std::runtime_error("loadFittingScene: cannot open " + (directory / "scene.txt").string());
    std::map<std::string, std::vector<double>> entries;
    std::string line;
    while (std::getline(text, line)) {
        std::istringstream words(line);
        std::string key;
        if (!(words >> key))
            continue;
        auto& values = entries[key];
        for (double value; words >> value;)
            values.push_back(value);
    }
    const auto single = [&](const char* key) {
        const auto found = entries.find(key);
        if (found == entries.end() || found->second.size() != 1)
            throw std::runtime_error(std::string("loadFittingScene: scene.txt needs one value for ") + key);
        return found->second[0];
    };
    FittingScene scene;
    scene.width = uint32_t(single("width"));
    scene.height = uint32_t(single("height"));
    const uint32_t views = uint32_t(single("views"));
    scene.radius = float(single("radius"));
    const auto& centre = entries["centre"];
    if (centre.size() != 3)
        throw std::runtime_error("loadFittingScene: scene.txt needs three values for centre");
    scene.centre = glm::vec3(centre[0], centre[1], centre[2]);
    for (const auto& [key, indices] : {std::pair{"train", &scene.train}, {"validation", &scene.validation}})
        for (const double index : entries[key]) {
            if (index < 0 || index >= views)
                throw std::runtime_error(std::string("loadFittingScene: view index out of range in ") + key);
            indices->push_back(uint32_t(index));
        }
    if (entries.count("points") && std::filesystem::exists(directory / "points.bin")) {
        const std::vector<float> coordinates = readFloats(directory / "points.bin", 3 * size_t(single("points")));
        for (size_t i = 0; i < coordinates.size(); i += 3)
            scene.points.emplace_back(coordinates[i], coordinates[i + 1], coordinates[i + 2]);
        scene.spacing = float(single("spacing"));
    }
    scene.targets = readFloats(directory / "targets.bin", 4 * size_t(views) * scene.width * scene.height);
    const std::vector<float> matrices = readFloats(directory / "cameras.bin", 16 * size_t(views));
    for (uint32_t view = 0; view < views; ++view)
        scene.cameras.push_back(glm::make_mat4(&matrices[16 * size_t(view)]));
    return scene;
}

void writeFittingScene(const FittingScene& scene, const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    writeFloats(directory / "targets.bin", scene.targets.data(), scene.targets.size());
    std::vector<float> matrices;
    for (const auto& camera : scene.cameras)
        matrices.insert(matrices.end(), glm::value_ptr(camera), glm::value_ptr(camera) + 16);
    writeFloats(directory / "cameras.bin", matrices.data(), matrices.size());
    std::ofstream text(directory / "scene.txt");
    if (!text)
        throw std::runtime_error("writeFittingScene: cannot open " + (directory / "scene.txt").string());
    text << "width " << scene.width << "\nheight " << scene.height << "\nviews " << scene.cameras.size() << "\ntrain";
    for (const uint32_t view : scene.train)
        text << " " << view;
    text << "\nvalidation";
    for (const uint32_t view : scene.validation)
        text << " " << view;
    text.precision(9);
    text << "\ncentre " << scene.centre.x << " " << scene.centre.y << " " << scene.centre.z << "\nradius "
         << scene.radius << "\n";
}

} // namespace klartraum::raster
