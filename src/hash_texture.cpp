// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/neural/hash_texture.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum::neural {

namespace {

struct HashTexturePushConstants {
    uint32_t width, height, levels, features, tableSize, grids, outputOffset, batch;
    uint32_t classes[4];
    uint32_t resolutions[16];
};

} // namespace

HashTextureShape HashTextureShape::fromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("HashTextureShape: cannot read " + path);
    HashTextureShape shape;
    shape.resolutions.clear();
    shape.classes.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream words(line);
        std::string key;
        words >> key;
        uint32_t value;
        if (key == "width")
            words >> shape.width;
        else if (key == "height")
            words >> shape.height;
        else if (key == "levels")
            words >> shape.levels;
        else if (key == "features")
            words >> shape.features;
        else if (key == "table")
            words >> shape.tableSize;
        else if (key == "resolutions")
            while (words >> value)
                shape.resolutions.push_back(value);
        else if (key == "classes")
            while (words >> value)
                shape.classes.push_back(value);
    }
    if (shape.width == 0 || shape.height == 0 || shape.resolutions.size() != shape.levels || shape.classes.empty())
        throw std::runtime_error("HashTextureShape: incomplete shape in " + path);
    return shape;
}

namespace {

HashTexturePushConstants pushConstantsOf(const HashTextureShape& shape) {
    if (shape.width == 0 || shape.height == 0 || shape.levels == 0 || shape.levels > 16 ||
        shape.resolutions.size() != shape.levels || shape.features == 0 || shape.features > 4 ||
        shape.classes.empty() || shape.classes.size() > 4 || (shape.tableSize & (shape.tableSize - 1)) != 0)
        throw std::invalid_argument("hashTexture: unsupported shape");
    for (uint32_t resolution : shape.resolutions)
        if (resolution == 0 || resolution > (1u << 20))
            throw std::invalid_argument("hashTexture: resolutions must be between 1 and 2^20");
    HashTexturePushConstants constants{shape.width,
                                       shape.height,
                                       shape.levels,
                                       shape.features,
                                       shape.tableSize,
                                       uint32_t(shape.classes.size()),
                                       shape.outputOffset,
                                       shape.batch,
                                       {},
                                       {}};
    for (size_t g = 0; g < shape.classes.size(); ++g)
        constants.classes[g] = shape.classes[g];
    for (size_t l = 0; l < shape.levels; ++l)
        constants.resolutions[l] = shape.resolutions[l];
    return constants;
}

} // namespace

ComputeGraphElementPtr hashTexture(VulkanContext& vulkanContext, const HashTextureShape& shape) {
    auto operation =
        vulkanContext.create<GeneralComputation<HashTexturePushConstants>>("shaders/neural/hash_texture.comp.spv");
    operation->setPushConstants({pushConstantsOf(shape)});
    operation->setGroupCount((shape.width + 15) / 16, (shape.height + 15) / 16, shape.batch);
    return operation;
}

ComputeGraphElementPtr hashTextureBackward(VulkanContext& vulkanContext, const HashTextureShape& shape) {
    auto operation = vulkanContext.create<GeneralComputation<HashTexturePushConstants>>(
        "shaders/neural/hash_texture_backward.comp.spv");
    operation->setPushConstants({pushConstantsOf(shape)});
    operation->setGroupCount((shape.width + 15) / 16, (shape.height + 15) / 16, shape.batch);
    return operation;
}

} // namespace klartraum::neural
