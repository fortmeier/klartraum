// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/fitting_scene.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include <glm/gtc/type_ptr.hpp>

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
