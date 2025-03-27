#ifndef KLARTRAUM_DRAWGRAPH_HPP
#define KLARTRAUM_DRAWGRAPH_HPP

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <queue>

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

class DrawGraph {
public:
    DrawGraph(VulkanKernel& vulkanKernel) : device(vulkanKernel.getDevice())
    {

        // create the command pool
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = vulkanKernel.getQueueFamilyIndices().graphicsAndComputeFamily.value();
    
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }    

  
    }

    void compileFrom(DrawGraphElementPtr element) {
        computeOrder(element);

        for(auto& element : ordered_elements) {
            element->_setup(device);
        }

        commandBuffers.resize(ordered_elements.size());

        // create the command buffers
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = commandBuffers.size();
    
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        for(auto& commandBuffer : commandBuffers) {
            // reset the command buffer before recording
            vkResetCommandBuffer(commandBuffer, 0);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0; // Optional
            beginInfo.pInheritanceInfo = nullptr; // Optional
        
            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }
            
            // record the command buffer
            element->_record(commandBuffer);

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }        

            // for now, all command buffers will be submitted to the same queue without any synchronization
            // this is okay since we sorted the elements in the graph before and the queue is
            // processing them one after another (assumption!!!)
            submit_infos.push_back(getSubmitInfoForElement(element, &commandBuffer));
        }
    }


    /*
    * Submit the graph to the graphics queue
    *
    * The submit infos will have to be prepared before by calling compile_from
    */
    VkFence& submitTo(VkQueue graphicsQueue) {
        // TODO implement the fence to return
        VkFence fence = VK_NULL_HANDLE;
        if (vkQueueSubmit(graphicsQueue, submit_infos.size(), submit_infos.data(), fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit the graph elements!");
        }
        return fence;
    }

private:
    VkDevice device;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<DrawGraphElementPtr> ordered_elements;

    std::vector<VkSubmitInfo> submit_infos;

    VkSubmitInfo getSubmitInfoForElement(DrawGraphElementPtr element, VkCommandBuffer* pCommandBuffer) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 0;
        submitInfo.pCommandBuffers = pCommandBuffer;
       
        // submitInfo.waitSemaphoreCount = 1;
        // submitInfo.pWaitSemaphores = waitSemaphores;
        // submitInfo.pWaitDstStageMask = waitStages;
    
    
        // submitInfo.signalSemaphoreCount = 1;
        // submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];
        return submitInfo;
    }

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

    void computeOrder(DrawGraphElementPtr element) {
        // Use Kahn's algorithm to find the execution order

        auto& L = ordered_elements;
        L.clear();

        std::queue<DrawGraphElementPtr> S;
        S.push(element);

        EdgeList edges;
        EdgeList incoming_edges;
        fill_edges(edges, incoming_edges, element);

        while (!S.empty())
        {
            // remove a node n from S
            auto n = S.front();
            S.pop();

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
            sum_edges += element.second.size();
        }

        size_t sum_incoming_edges = 0;
        for(auto& element : incoming_edges) {
            sum_incoming_edges += element.second.size();
        }

        if (sum_edges > 0 || sum_incoming_edges > 0) {
            throw std::runtime_error("graph has cycles!");
        }

        std::reverse(L.begin(), L.end());

    }        
};

} // namespace klartraum

#endif // KLARTRAUM_DRAWGRAPH_HPP