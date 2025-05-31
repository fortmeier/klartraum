#ifndef KLARTRAUM_COMPUTEGRAPHGROUP_HPP
#define KLARTRAUM_COMPUTEGRAPHGROUP_HPP

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

class DrawGraphGroup : public virtual DrawGraphElement {
public:
    virtual const char* getType() const {
        return "DrawGraphGroup";
    }

protected:
    std::unique_ptr<DrawGraphElement> inElement;
    std::unique_ptr<DrawGraphElement> outElement;
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPHGROUP_HPP



