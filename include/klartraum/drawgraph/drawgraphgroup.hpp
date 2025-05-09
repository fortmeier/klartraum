#ifndef KLARTRAUM_COMPUTEGRAPHGROUP_HPP
#define KLARTRAUM_COMPUTEGRAPHGROUP_HPP

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

class DrawGraphGroup : public virtual DrawGraphElement {
public:
    virtual void setInput(DrawGraphElementPtr input, int index = 0) {
        inputs[index] = input;
    }

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        // Default implementation does nothing
    }

    virtual std::string getType() const {
        return "DrawGraphGroup";
    }

protected:
    std::unique_ptr<DrawGraphElement> inElement;
    std::unique_ptr<DrawGraphElement> outElement;
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPHGROUP_HPP



