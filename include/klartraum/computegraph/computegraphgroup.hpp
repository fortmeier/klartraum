// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_COMPUTEGRAPHGROUP_HPP
#define KLARTRAUM_COMPUTEGRAPH_COMPUTEGRAPHGROUP_HPP

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

/**
 * @brief Base class for elements built from several graph elements, such as OnnxNetwork or the Gaussian splatting
 * backends.
 *
 * getInputs() returns the group's output elements, so compiling a graph traverses
 * from the group through its output elements back to all elements of the group.
 */
class ComputeGraphGroup : public virtual ComputeGraphElement {
public:
    virtual const char* getType() const { return "ComputeGraphGroup"; }

    virtual std::map<int, ComputeGraphElementPtr> getInputs() const { return outputElements; }

protected:
    // this contains the graph elements that are the output of the group
    // the computegraph compilation traversal will use this to traverse from
    // the successor elements back through all the elements of the group
    std::map<int, ComputeGraphElementPtr> outputElements;
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_COMPUTEGRAPHGROUP_HPP
