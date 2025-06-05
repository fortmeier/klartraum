#ifndef KLARTRAUM_COMPUTEGRAPHGROUP_HPP
#define KLARTRAUM_COMPUTEGRAPHGROUP_HPP

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

class DrawGraphGroup : public virtual DrawGraphElement {
public:

    virtual const char* getType() const {
        return "DrawGraphGroup";
    }

    virtual std::map<int, DrawGraphElementPtr> getInputs() const {
        return outputElements;
    }

protected:
    // this contains the graph elements that are the output of the group
    // the drawgraph compilation traversal will use this to traverse from
    // the successor elements back through all the elements of the group 
    std::map<int, DrawGraphElementPtr> outputElements;
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPHGROUP_HPP



