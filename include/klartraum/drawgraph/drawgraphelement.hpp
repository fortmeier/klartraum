#ifndef KLARTRAUM_DRAWGRAPHELEMENT_HPP
#define KLARTRAUM_DRAWGRAPHELEMENT_HPP

#include <map>
#include <memory>

namespace klartraum {

class DrawGraphElement;

typedef std::shared_ptr<DrawGraphElement> DrawGraphElementPtr;

class DrawGraphElement {
public:
    virtual void set_input(DrawGraphElementPtr input, int index = 0) {
        inputs[index] = input;
    }
    //virtual DrawGraphElement& get_output() = 0;

    virtual void _setup(VkDevice& device) {};

    virtual void _record(VkCommandBuffer commandBuffer) {};

    virtual const char* getName() const = 0;

    std::map<int, DrawGraphElementPtr> inputs;
};



} // namespace klartraum

#endif // KLARTRAUM_DRAWGRAPHELEMENT_HPP