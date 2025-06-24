#ifndef KLARTRAUM_COMPUTEGRAPHGROUP_HPP
#define KLARTRAUM_COMPUTEGRAPHGROUP_HPP

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

class ComputeGraphGroup : public virtual ComputeGraphElement {
public:

    virtual const char* getType() const {
        return "ComputeGraphGroup";
    }

    virtual std::map<int, ComputeGraphElementPtr> getInputs() const {
        return outputElements;
    }

protected:
    // this contains the graph elements that are the output of the group
    // the computegraph compilation traversal will use this to traverse from
    // the successor elements back through all the elements of the group 
    std::map<int, ComputeGraphElementPtr> outputElements;
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPHGROUP_HPP



