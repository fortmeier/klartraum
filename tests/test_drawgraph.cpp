#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/drawgraph.hpp"

using namespace klartraum;

class Drawable : public DrawGraphElement {
    virtual const char* getName() const {
        return "Drawable";
    }
};

class BlurOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "BlurOp";
    }

};

class NoiseOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "NoiseOp";
    }

};

class AddOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "AddOp";
    }

};

class CopyOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "CopyOp";
    }

};



TEST(DrawGraph, create) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    /*
    STEP 1: create the drawgraph elements
    */
    
    auto drawable = std::make_shared<Drawable>();
    auto blur = std::make_shared<BlurOp>();
    blur->set_input(drawable);
    auto noise = std::make_shared<NoiseOp>();
    noise->set_input(blur);

    auto add = std::make_shared<AddOp>();
    add->set_input(blur, 0);
    add->set_input(noise, 1);

    auto copy = std::make_shared<CopyOp>();
    copy->set_input(add);

    
    /*
    STEP 2: create the drawgraph backend
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph();
    drawgraph.compile_from(copy);

    exit(0);
    
    // klartraum::VulkanBuffer<float> buffer(vulkanKernel, 7);
    // std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    // buffer.memcopy_from(data);
    // std::vector<float> data2(7);
    // buffer.memcopy_to(data2);
    // for (int i = 0; i < 7; i++) {
    //     EXPECT_EQ(data[i], data2[i]);
    // }
    return;
}