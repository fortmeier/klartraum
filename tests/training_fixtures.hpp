// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Reference tensors of the training elements, computed by PyTorch: scripts/neural/make_element_fixtures.py writes
// every case to build/TestingOutput/training_fixtures/ (<case>.txt lists each tensor with its shape,
// <case>_<tensor>.bin holds its floats). Tests that need a case skip without it.

#ifndef KLARTRAUM_TESTS_TRAINING_FIXTURES_HPP
#define KLARTRAUM_TESTS_TRAINING_FIXTURES_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::test {

inline const std::string kFixtures = "build/TestingOutput/training_fixtures/";

// A fixture case: its tensors' shapes and values.
struct Fixture {
    std::map<std::string, std::vector<uint32_t>> shapes;

    explicit Fixture(const std::string& name)
        : name(name) {
        std::ifstream file(kFixtures + name + ".txt");
        for (std::string line; std::getline(file, line);) {
            std::istringstream words(line);
            std::string tensor;
            words >> tensor;
            std::vector<uint32_t> shape;
            for (uint32_t d; words >> d;)
                shape.push_back(d);
            shapes[tensor] = shape;
        }
    }

    bool exists() const { return !shapes.empty(); }

    std::vector<float> values(const std::string& tensor) const {
        std::ifstream file(kFixtures + name + "_" + tensor + ".bin", std::ios::binary | std::ios::ate);
        std::vector<float> data(size_t(file.tellg()) / sizeof(float));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
        return data;
    }

    std::string name;
};

inline std::shared_ptr<TensorElement<float>> tensor(VulkanContext& vc, const std::vector<uint32_t>& shape) {
    return vc.create<TensorElement<float>>(shape);
}

inline std::vector<float> read(const std::shared_ptr<TensorElement<float>>& t) {
    std::vector<float> values(t->getDataElementCount());
    t->getDataBuffer(0).memcopyTo(values);
    return values;
}

// The largest difference relative to the reference's largest magnitude.
inline float relativeError(const std::vector<float>& actual, const std::vector<float>& expected) {
    EXPECT_EQ(actual.size(), expected.size());
    float worst = 0.0f, largest = 1e-12f;
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        worst = std::max(worst, std::abs(actual[i] - expected[i]));
        largest = std::max(largest, std::abs(expected[i]));
    }
    return worst / largest;
}

} // namespace klartraum::test

#endif // KLARTRAUM_TESTS_TRAINING_FIXTURES_HPP
