// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_GRAPH_EXPORT_HPP
#define KLARTRAUM_COMPUTEGRAPH_GRAPH_EXPORT_HPP

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"

namespace klartraum {

/**
 * @brief A group of elements to mark in an exported graph, e.g. an ONNX network or a splatting backend.
 *
 * Its members are the elements `element` depends on, without the `boundary` elements and what they depend on
 * (the group's inputs from outside).
 */
struct ExportGroup {
    std::string name;
    ComputeGraphElementPtr element;
    std::vector<ComputeGraphElementPtr> boundary;
};

/**
 * @brief Writes the structure of a compiled graph as JSON, for drawing it.
 *
 * The elements in recording order (`id`), with name, type, the groups containing them and a `key` that is the same
 * for an element in every graph it is part of (e.g. a tensor passed from one graph of a GraphChain to the next), and
 * the edges as [predecessor id, element id]. With profiling enabled, each element also gets `ms`, its mean GPU time
 * (ComputeGraph::getElementTimesMs()).
 */
inline void writeGraphJson(const std::filesystem::path& path, const std::string& name, const ComputeGraph& graph,
                           const std::vector<ExportGroup>& groups) {
    const std::vector<double> times = graph.getElementTimesMs();
    bool timed = false;
    for (double time : times) {
        timed = timed || time > 0.0;
    }
    const auto& elements = graph.getElements();
    std::map<ComputeGraphElementPtr, size_t> index;
    for (size_t i = 0; i < elements.size(); ++i) {
        index[elements[i]] = i;
    }
    auto ancestors = [&](const ComputeGraphElementPtr& element, std::set<ComputeGraphElementPtr>& result) {
        std::vector<ComputeGraphElementPtr> stack{element};
        while (!stack.empty()) {
            const auto current = stack.back();
            stack.pop_back();
            for (const auto& predecessor : graph.getPredecessors(current)) {
                if (result.insert(predecessor).second) {
                    stack.push_back(predecessor);
                }
            }
        }
    };
    std::vector<std::set<ComputeGraphElementPtr>> members;
    for (const auto& group : groups) {
        if (!index.count(group.element)) {
            throw std::runtime_error("writeGraphJson: group " + group.name + " is not part of the graph");
        }
        std::set<ComputeGraphElementPtr> inside, outside;
        ancestors(group.element, inside);
        for (const auto& boundary : group.boundary) {
            outside.insert(boundary);
            ancestors(boundary, outside);
        }
        for (const auto& element : outside) {
            inside.erase(element);
        }
        members.push_back(inside);
    }
    auto escape = [](const std::string& text) {
        std::string result;
        for (char c : text) {
            if (c == '"' || c == '\\') {
                result += '\\';
                result += c;
            } else if (static_cast<unsigned char>(c) < 0x20) {
                result += ' ';
            } else {
                result += c;
            }
        }
        return result;
    };

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("writeGraphJson: cannot write " + path.string());
    }
    out << "{\n  \"name\": \"" << escape(name) << "\",\n  \"groups\": [";
    for (size_t g = 0; g < groups.size(); ++g) {
        out << (g ? ", " : "") << "{\"name\": \"" << escape(groups[g].name)
            << "\", \"element\": " << index.at(groups[g].element) << "}";
    }
    out << "],\n  \"elements\": [\n";
    for (size_t i = 0; i < elements.size(); ++i) {
        char key[32];
        std::snprintf(key, sizeof(key), "%p", static_cast<const void*>(elements[i].get()));
        out << "    {\"id\": " << i << ", \"key\": \"" << key << "\", \"name\": \"" << escape(elements[i]->getName())
            << "\", \"type\": \"" << escape(elements[i]->getType()) << "\", ";
        if (timed) {
            out << "\"ms\": " << times[i] << ", ";
        }
        out << "\"groups\": [";
        bool first = true;
        for (size_t g = 0; g < groups.size(); ++g) {
            if (members[g].count(elements[i])) {
                out << (first ? "" : ", ") << g;
                first = false;
            }
        }
        out << "]}" << (i + 1 < elements.size() ? "," : "") << "\n";
    }
    out << "  ],\n  \"edges\": [\n";
    bool first = true;
    for (size_t i = 0; i < elements.size(); ++i) {
        for (const auto& predecessor : graph.getPredecessors(elements[i])) {
            out << (first ? "    " : ",\n    ") << "[" << index.at(predecessor) << ", " << i << "]";
            first = false;
        }
    }
    out << "\n  ]\n}\n";
}

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_GRAPH_EXPORT_HPP
