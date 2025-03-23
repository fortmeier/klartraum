#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "klartraum/glfw_frontend.hpp"
//#include "klartraum/vulkan_buffer.hpp"



class DrawGraphElement {
public:
    virtual void set_input(DrawGraphElement* input, int index = 0) {
        inputs[index] = input;
    }
    //virtual DrawGraphElement& get_output() = 0;

    virtual const char* getName() const = 0;

    std::map<int, DrawGraphElement*> inputs;
};

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

class DrawGraph {
public:
    void compile_from(DrawGraphElement* element) {
        // traverse the graph and create vulkan objects
       backwards(element);
    }

    void backwards(DrawGraphElement* element, bool mainpath = true) {
        // do not traverse if element already in graph
        if(visited.find(element) != visited.end()) {
            return;
        }

        visited.insert(element);

        std::cout << element->getName() << " mainpath = " << mainpath << std::endl;
        for(auto& input : element->inputs) {
            backwards(input.second, mainpath);
            mainpath = false;
        }
        
    }

    std::set<DrawGraphElement*> visited;

    typedef std::vector<DrawGraphElement*> Branch;
    std::vector<Branch> branches;

};

TEST(DrawGraph, create) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    /*
    STEP 1: create the drawgraph elements
    */
    
    auto& drawable = Drawable();
    auto& blur = BlurOp();
    blur.set_input(&drawable);

    auto& blur2 = BlurOp();
    blur2.set_input(&blur);

    auto& add = AddOp();
    add.set_input(&blur, 0);
    add.set_input(&blur2, 1);


    auto& copy = CopyOp();
    copy.set_input(&add);
    
    /*
    STEP 2: create the drawgraph backend
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph();
    drawgraph.compile_from(&copy);

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