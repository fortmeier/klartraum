// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - ConstructorWithIndividualDimensions: a TensorElement can be created from width,
 *   height, depth and batch
 * - ConstructorWithDimensionsVector: a TensorElement can be created from a vector of
 *   dimensions
 * - TensorElementSinglePathConstruction: a TensorElementSinglePath can be created from
 *   width, height, depth and batch
 * - TensorElementSinglePathConstructionWithVector: a TensorElementSinglePath can be
 *   created from a vector of dimensions
 * - TensorElementSinglePathGetDimensions: a TensorElementSinglePath returns the
 *   dimensions it was created with
 * - SharesPhysicalStorageAndRetainsLogicalSize: compatible tensors can share physical
 *   storage while keeping their logical sizes
 * - RejectsUndersizedSharedStorage: a tensor cannot share storage that is smaller than
 *   its logical allocation
 **/

#include <gtest/gtest.h>

#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"

using namespace klartraum;

class TensorElementTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vulkanContext;
};

TEST_F(TensorElementTest, ConstructorWithIndividualDimensions) {
    // Test creating TensorElement with individual dimension parameters
    auto tensorElement =
        std::make_unique<TensorElement<float>>(*vulkanContext, 224, 224, 3, 1 // width, height, depth, batch
        );
    EXPECT_NE(tensorElement, nullptr);
}

TEST_F(TensorElementTest, ConstructorWithDimensionsVector) {
    // Test creating TensorElement with dimensions vector
    std::vector<uint32_t> dims = {128, 128, 4, 2};

    EXPECT_NO_THROW({
        auto tensorElement = std::make_unique<TensorElement<float>>(*vulkanContext, dims);
        EXPECT_NE(tensorElement, nullptr);
    });
}

TEST_F(TensorElementTest, TensorElementSinglePathConstruction) {
    // Test creating TensorElementSinglePath with individual dimensions
    auto tensorElementSinglePath =
        std::make_unique<TensorElementSinglePath<float>>(*vulkanContext, 128, 128, 3, 1 // width, height, depth, batch
        );
    EXPECT_NE(tensorElementSinglePath, nullptr);
}

TEST_F(TensorElementTest, TensorElementSinglePathConstructionWithVector) {
    // Test creating TensorElementSinglePath with dimensions vector
    std::vector<uint32_t> dims = {64, 64, 1, 4};

    EXPECT_NO_THROW({
        auto tensorElementSinglePath = std::make_unique<TensorElementSinglePath<float>>(*vulkanContext, dims);
        EXPECT_NE(tensorElementSinglePath, nullptr);
    });
}

TEST_F(TensorElementTest, TensorElementSinglePathGetDimensions) {
    // Test getting dimensions from TensorElementSinglePath
    std::vector<uint32_t> expectedDims = {32, 32, 2, 1};
    auto tensorElementSinglePath = std::make_unique<TensorElementSinglePath<float>>(*vulkanContext, expectedDims);

    auto actualDims = tensorElementSinglePath->getDimensions();
    EXPECT_EQ(actualDims, expectedDims);
}

TEST_F(TensorElementTest, SharesPhysicalStorageAndRetainsLogicalSize) {
    auto large = std::make_shared<TensorElement<float>>(*vulkanContext, std::vector<uint32_t>{16});
    auto small = std::make_shared<TensorElement<float>>(*vulkanContext, std::vector<uint32_t>{4});

    small->shareDataStorageWith(*large);
    large->_setup(*vulkanContext, 1);
    small->_setup(*vulkanContext, 1);

    EXPECT_EQ(large->getDataVkBuffer(0), small->getDataVkBuffer(0));
    EXPECT_EQ(large->getBufferMemSize(), 16 * sizeof(float));
    EXPECT_EQ(small->getBufferMemSize(), 4 * sizeof(float));
    EXPECT_EQ(small->getStorageCapacityBytes(), 16 * sizeof(float));
    EXPECT_EQ(large->getStorageIdentity(), small->getStorageIdentity());
}

TEST_F(TensorElementTest, RejectsUndersizedSharedStorage) {
    auto small = std::make_shared<TensorElement<float>>(*vulkanContext, std::vector<uint32_t>{4});
    auto large = std::make_shared<TensorElement<float>>(*vulkanContext, std::vector<uint32_t>{16});

    EXPECT_THROW(large->shareDataStorageWith(*small), std::invalid_argument);
}
