#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/computegraph/tensorelement.hpp"

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
    EXPECT_NO_THROW({
        auto tensorElement = std::make_unique<TensorElement<float>>(
            *vulkanContext, 
            224, 224, 3, 1,  // width, height, depth, batch
            224 * 224 * 3 * 1  // total data elements
        );
        EXPECT_NE(tensorElement, nullptr);
    });
}

TEST_F(TensorElementTest, ConstructorWithDimensionsVector) {
    // Test creating TensorElement with dimensions vector
    std::vector<uint32_t> dims = {128, 128, 4, 2};
    uint32_t totalElements = 128 * 128 * 4 * 2;
    
    EXPECT_NO_THROW({
        auto tensorElement = std::make_unique<TensorElement<float>>(
            *vulkanContext,
            dims,
            totalElements
        );
        EXPECT_NE(tensorElement, nullptr);
    });
}
