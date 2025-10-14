#include <gtest/gtest.h>

#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/glfw_frontend.hpp"

using namespace klartraum;

class TensorElementTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<GlfwFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<GlfwFrontend> frontend;
    VulkanContext* vulkanContext;
};

TEST_F(TensorElementTest, ConstructorWithIndividualDimensions) {
    // Test creating TensorElement with individual dimension parameters
    auto tensorElement = std::make_unique<TensorElement<float>>(
        *vulkanContext,
        224, 224, 3, 1 // width, height, depth, batch
    );
    EXPECT_NE(tensorElement, nullptr);
}

TEST_F(TensorElementTest, ConstructorWithDimensionsVector) {
    // Test creating TensorElement with dimensions vector
    std::vector<uint32_t> dims = {128, 128, 4, 2};

    EXPECT_NO_THROW({
        auto tensorElement = std::make_unique<TensorElement<float>>(
            *vulkanContext,
            dims);
        EXPECT_NE(tensorElement, nullptr);
    });
}

TEST_F(TensorElementTest, TensorElementSinglePathConstruction) {
    // Test creating TensorElementSinglePath with individual dimensions
    auto tensorElementSinglePath = std::make_unique<TensorElementSinglePath<float>>(
        *vulkanContext,
        128, 128, 3, 1 // width, height, depth, batch
    );
    EXPECT_NE(tensorElementSinglePath, nullptr);
}

TEST_F(TensorElementTest, TensorElementSinglePathConstructionWithVector) {
    // Test creating TensorElementSinglePath with dimensions vector
    std::vector<uint32_t> dims = {64, 64, 1, 4};

    EXPECT_NO_THROW({
        auto tensorElementSinglePath = std::make_unique<TensorElementSinglePath<float>>(
            *vulkanContext,
            dims);
        EXPECT_NE(tensorElementSinglePath, nullptr);
    });
}

TEST_F(TensorElementTest, TensorElementSinglePathGetDimensions) {
    // Test getting dimensions from TensorElementSinglePath
    std::vector<uint32_t> expectedDims = {32, 32, 2, 1};
    auto tensorElementSinglePath = std::make_unique<TensorElementSinglePath<float>>(
        *vulkanContext,
        expectedDims);

    auto actualDims = tensorElementSinglePath->getDimensions();
    EXPECT_EQ(actualDims, expectedDims);
}