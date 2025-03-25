#ifndef KLARTRAUM_DRAWGRAPH_HPP
#define KLARTRAUM_DRAWGRAPH_HPP

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <queue>

#include "klartraum/drawgraphelement.hpp"

namespace klartraum {

class DrawGraph {
public:

    typedef std::map<DrawGraphElementPtr, std::vector<DrawGraphElementPtr>> EdgeList;

    void fill_edges(EdgeList& edges, EdgeList& incoming, DrawGraphElementPtr element) {
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

    void compile_from(DrawGraphElementPtr element) {
        // traverse the graph and create vulkan objects
        //backwards(element);

        // Use Kahn's algorithm to find the execution order
    
        auto& L = ordered_elements;
        std::queue<DrawGraphElementPtr> S;
        S.push(element);

        //std::set<std::pair<DrawGraphElementPtr, DrawGraphElementPtr>> remove_edges;
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

    void backwards(DrawGraphElementPtr element, bool mainpath = true) {
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

    std::vector<DrawGraphElementPtr> ordered_elements;

    std::set<DrawGraphElementPtr> visited;

    typedef std::vector<DrawGraphElementPtr> Branch;
    std::vector<Branch> branches;

};

} // namespace klartraum

#endif // KLARTRAUM_DRAWGRAPH_HPP