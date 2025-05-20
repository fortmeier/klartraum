#ifndef KLARTRAUM_DRAWGRAPH_HPP
#define KLARTRAUM_DRAWGRAPH_HPP

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <queue>

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

class SubmitInfoWrapper {
public:
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkSemaphore> signalSemaphores;
    VkSubmitInfo submitInfo{};
    std::vector<VkPipelineStageFlags> waitStages; //{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
};


class DrawGraph {
public:
    DrawGraph(VulkanKernel& vulkanKernel, uint32_t numberPaths) : vulkanKernel(vulkanKernel), numberPaths(numberPaths) 
    {
        auto& device = vulkanKernel.getDevice();

        all_path_submit_infos.resize(numberPaths);
        all_path_submit_info_wrappers.resize(numberPaths);
        allRenderFinishedSemaphores.resize(numberPaths);

        // create the command pool
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = vulkanKernel.getQueueFamilyIndices().graphicsAndComputeFamily.value();
    
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }    

  
    }

    virtual ~DrawGraph() {
        auto& device = vulkanKernel.getDevice();

        // clear the outputs of all elements
        // otherwise we will have dangling pointers in the graph
        clearOutputs();

        // destroy the semaphores
        for(auto& semaphores: allRenderFinishedSemaphores) {
            for(auto& semaphore_list: semaphores) {
                for(auto& semaphore: semaphore_list.second) {
                    vkDestroySemaphore(device, semaphore.second, nullptr);
                }
            }
        }

        for(auto& semaphores: graphFinishedSemaphores) {
            vkDestroySemaphore(device, semaphores, nullptr);
        }

        for(auto& buffer: commandBuffers) {
            vkFreeCommandBuffers(device, commandPool, 1, &buffer);
        }
        // destroy the command pool
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    void compileFrom(DrawGraphElementPtr element) {
        auto& device = vulkanKernel.getDevice();
        
        computeOrder(element);

        updateOutputs();

        createRenderFinishedSemaphores();

        createGraphFinishedSemaphores();

        for(auto& element : ordered_elements) {
            element->_setup(vulkanKernel, numberPaths);
        }

        commandBuffers.resize(ordered_elements.size() * numberPaths);

        // create the command buffers
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)(commandBuffers.size());
    
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        for(uint32_t pathId = 0; pathId < numberPaths; pathId++) {
            for(size_t i = 0; i < ordered_elements.size(); i++) {
                auto& element = ordered_elements[i];
                VkCommandBuffer& commandBuffer = commandBuffers[i * numberPaths + pathId];                
                recordCommandBuffer(commandBuffer, element, pathId);
                // for now, all command buffers will be submitted to the same queue without any synchronization
                // this is okay since we sorted the elements in the graph before and the queue is
                // processing them one after another (assumption!!!)
                SubmitInfoWrapperList& submitInfoWrappers = all_path_submit_info_wrappers[pathId];
                // TODO: this is the time to grok move semantics
                SubmitInfoWrapper submitInfoWrapper;
                submitInfoWrappers.push_back(submitInfoWrapper);
                SubmitInfoWrapper& submitInfoWrapper2 = submitInfoWrappers.back();
                getSubmitInfoForElement(submitInfoWrapper2, pathId, element, &commandBuffer);
                SubmitInfoList& submit_infos = all_path_submit_infos[pathId];
                submit_infos.push_back(submitInfoWrapper2.submitInfo);
            }
        }
    }


    /*
    * Submit the graph to the graphics queue
    *
    * The submit infos will have to be prepared before by calling compile_from
    */
   VkSemaphore submitTo(VkQueue graphicsQueue, uint32_t pathId, VkFence fence = VK_NULL_HANDLE) {
        auto& submit_infos = all_path_submit_infos[pathId];

        // the following seems not to work if there are multiple paths in the graph
        // if (vkQueueSubmit(graphicsQueue, submit_infos.size(), submit_infos.data(), nullptr) != VK_SUCCESS) {
        //     throw std::runtime_error("failed to submit the graph elements!");
        // }
        // instead we have to submit them one by one
        // this is not optimal but it works for now
        // in the future, we will merge command buffers of consecutive elements
        // and submit them together

        if (vkQueueSubmit(graphicsQueue, (uint32_t)submit_infos.size(), submit_infos.data(), fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit the graph elements!");
        }

        return graphFinishedSemaphores[pathId];
    }

    void submitAndWait(VkQueue graphicsQueue, uint32_t pathId) {
        auto& device = vulkanKernel.getDevice();
    
        
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    
        VkFence fence;
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create fence!");
        }
        auto finishSemaphore = submitTo(graphicsQueue, pathId, fence);

        VkResult waitResult = vkWaitForFences(device, 1, &fence, true, UINT64_MAX);
        if (waitResult != VK_SUCCESS) {
            throw std::runtime_error("failed to wait for fence!");
        }

        vkDestroyFence(device, fence, nullptr);        
        return;
    }

private:
    VulkanKernel& vulkanKernel;
    uint32_t numberPaths;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<DrawGraphElementPtr> ordered_elements;

    typedef std::vector<VkSubmitInfo> SubmitInfoList;
    typedef std::vector<SubmitInfoWrapper> SubmitInfoWrapperList;

    std::vector<SubmitInfoList> all_path_submit_infos;
    std::vector<SubmitInfoWrapperList> all_path_submit_info_wrappers;

    typedef std::map<DrawGraphElementPtr, VkSemaphore> SemaphoreMap;
    typedef std::map<DrawGraphElementPtr, SemaphoreMap> SemaphoreMapMap;
    
    std::vector<SemaphoreMapMap> allRenderFinishedSemaphores;

    std::vector<VkSemaphore> graphFinishedSemaphores;


    void recordCommandBuffer(VkCommandBuffer commandBuffer, DrawGraphElementPtr element, uint32_t pathId) {
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
        element->_record(commandBuffer, pathId);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }        
    }

    void getSubmitInfoForElement(SubmitInfoWrapper& submitInfoWrapper, uint32_t pathId, DrawGraphElementPtr element, VkCommandBuffer* pCommandBuffer) {
        
        auto& submitInfo = submitInfoWrapper.submitInfo;
        auto& waitSemaphores = submitInfoWrapper.waitSemaphores;
        auto& waitStages = submitInfoWrapper.waitStages;
        auto& signalSemaphores = submitInfoWrapper.signalSemaphores;
        
        if(element->renderWaitSemaphores.find(pathId) != element->renderWaitSemaphores.end()) {
            waitSemaphores.push_back(element->renderWaitSemaphores[pathId]);
            waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
        
       
        // for the element, we want to find the semaphores that connect the
        // element to its inputs (given that the element has inputs)
        // for that we have to first get the SemphoreMapMap for the pathId
        // and then find the SemaphoreMap for the element, which
        // contains the semaphore for the connection between the element and its input element
        auto& renderFinishedSemaphores = allRenderFinishedSemaphores[pathId];
        for(auto& input : element->inputs) {
            auto& inputElement = input.second;

            auto input_output_map_iter = renderFinishedSemaphores.find(inputElement);
            if(input_output_map_iter != renderFinishedSemaphores.end()) {
                // would be better if it would be a map
                // now find the element in the output of the input element
                auto element_iter = renderFinishedSemaphores[inputElement].find(element);
                if(element_iter != renderFinishedSemaphores[inputElement].end()) {
                    // Only push back if the semaphore is not already in waitSemaphores
                    if (std::find(waitSemaphores.begin(), waitSemaphores.end(), element_iter->second) == waitSemaphores.end()) {
                        waitSemaphores.push_back(element_iter->second);
                        waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                    }
                } else {
                    throw std::runtime_error("failed to find the element in the output of the input element!");
                }
            }
        }
        
        auto input_output_map_iter = renderFinishedSemaphores.find(element);
        if(input_output_map_iter != renderFinishedSemaphores.end()) {
            for(auto& outputElement : element->outputs) {
                auto element_iter = renderFinishedSemaphores[element].find(outputElement);
                if(element_iter != renderFinishedSemaphores[element].end()) {
                    // Only push back if the semaphore is not already in signalSemaphores
                    if (std::find(signalSemaphores.begin(), signalSemaphores.end(), element_iter->second) == signalSemaphores.end()) {
                        signalSemaphores.push_back(element_iter->second);
                    }
                } else {
                    throw std::runtime_error("failed to find the semaphore connecting element to the output element!");
                }
            }
        }        

        // if it does not have any inputs, we can just use the graph finish semaphore
        if(element->outputs.size() == 0)
        {
            signalSemaphores.push_back(graphFinishedSemaphores[pathId]);
        }
        
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = pCommandBuffer;
       
        submitInfo.waitSemaphoreCount = (uint32_t)waitSemaphores.size();
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
    
        submitInfo.signalSemaphoreCount = (uint32_t)signalSemaphores.size();
        submitInfo.pSignalSemaphores = signalSemaphores.data();
    }

    void createRenderFinishedSemaphores() {
        auto& device = vulkanKernel.getDevice();
        auto& config = vulkanKernel.getConfig();

        // create the render finished semaphores
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
        for (uint32_t i = 0; i < numberPaths; i++) {
            // get the render finished semaphores for this path
            auto& renderFinishedSemaphores = allRenderFinishedSemaphores[i];
            // create mulitple semaphores for each element in the path
            // (one for each output of the element)
            for(auto& element : ordered_elements) {
                for(auto& output_element : element->outputs) {
                    VkSemaphore* finishSemaphore = &renderFinishedSemaphores[element][output_element];
                    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, finishSemaphore) != VK_SUCCESS) {
                        throw std::runtime_error("failed to create render finished semaphore!");
                    }
                }
            }
        }        
    }

    void createGraphFinishedSemaphores() {
        auto& device = vulkanKernel.getDevice();
        auto& config = vulkanKernel.getConfig();

        // create the render finished semaphores
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        graphFinishedSemaphores.resize(numberPaths);
    
        for (uint32_t i = 0; i < numberPaths; i++) {
            VkSemaphore* graphFinishedSemaphore = &graphFinishedSemaphores[i];
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, graphFinishedSemaphore) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graph finished semaphore!");
            }
        }        
    }

    void updateOutputs() {
        // first, clear the outputs of all elements
        // to make sure that we have a clean slate
        for(auto& element : ordered_elements) {
            element->outputs.clear();
        }

        // now, update the outputs of all elements
        for(auto& element : ordered_elements) {
            for(auto& input : element->inputs) {
                auto& inputElement = input.second;
                // Only add if element is not already in outputs
                if (std::find(inputElement->outputs.begin(), inputElement->outputs.end(), element) == inputElement->outputs.end()) {
                    inputElement->outputs.push_back(element);
                }
            }
        }
    }

    void clearOutputs() {
        for(auto& element : ordered_elements) {
            element->outputs.clear();
        }
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