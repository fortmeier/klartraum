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

class DrawGraph {
public:
    typedef std::map<DrawGraphElement*, std::vector<DrawGraphElement*>> EdgeList;

    void fill_edges(EdgeList& edges, EdgeList& incoming, DrawGraphElement* element) {
        for(auto& input : element->inputs) {
            // check if input already in graph
            auto it = find(edges[element].begin(), edges[element].end(), input.second);
            if(it == edges[element].end()) {
                // if not, add it
                edges[element].push_back(input.second);
                incoming[input.second].push_back(element);
                std::cout << "edge: " << element->getName() << " -> " << input.second->getName() << std::endl;
                fill_edges(edges, incoming, input.second);
            }
        }
    }

    void compile_from(DrawGraphElement* element) {
        // traverse the graph and create vulkan objects
        //backwards(element);

        // Use Kahn's algorithm to find the execution order
    
        auto& L = ordered_elements;
        std::queue<DrawGraphElement*> S;
        S.push(element);

        //std::set<std::pair<DrawGraphElement*, DrawGraphElement*>> remove_edges;
        EdgeList edges;
        EdgeList incoming_edges;
        fill_edges(edges, incoming_edges, element);

        while (!S.empty())
        {
            // remove a node n from S
            auto n = S.front();
            S.pop();

            std::cout << "node: " << n->getName() << std::endl;

            L.push_back(n);
            for(auto input = n->inputs.begin(); input != n->inputs.end(); input++) {
                // note the convention that N and M are iterators
                auto m = input->second;
                // first check if the input node is still in the graph
                auto M = find(edges[n].begin(), edges[n].end(), m);
                if(M != edges[n].end()) {
                    // if yes, remove edge N->M from the graph
                    edges[n].erase(M);
                    auto N = find(incoming_edges[m].begin(), incoming_edges[m].end(), n);
                    incoming_edges[m].erase(N);
                    
                    // if m has no other incoming edges then
                    // insert m into S
                    if(incoming_edges[m].empty()) {
                        S.push(m);
                    }
                }


            }
        }
        size_t sum_edges = 0;
        for(auto& element : edges) {
            std::cout << element.first->getName() << " -> " << element.second.size() << std::endl;
            sum_edges += element.second.size();
        }
        std::cout << "edges size: " << sum_edges << std::endl;

        size_t sum_incoming_edges = 0;
        for(auto& element : incoming_edges) {
            std::cout << element.first->getName() << " <- " << element.second.size() << std::endl;
            sum_incoming_edges += element.second.size();
        }
        std::cout << "incoming edges size: " << sum_incoming_edges << std::endl;

        for(auto& element : L) {
            std::cout << element->getName() << std::endl;
        }

        std::cout << "" << std::endl;
        
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

    std::vector<DrawGraphElement*> ordered_elements;

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

    auto& blur2 = NoiseOp();
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