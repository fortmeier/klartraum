#ifndef KLARTRAUM_COMPUTEGRAPH_HPP
#define KLARTRAUM_COMPUTEGRAPH_HPP

#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

class SubmitInfoWrapper {
public:
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkSemaphore> signalSemaphores;
    VkSubmitInfo submitInfo{};
    std::vector<VkPipelineStageFlags> waitStages; //{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
};

class ComputeGraph {
public:
    ComputeGraph(VulkanContext& vulkanContext, uint32_t numberPaths) : vulkanContext(vulkanContext), numberPaths(numberPaths) {
        auto& device = vulkanContext.getDevice();

        all_path_submit_infos.resize(numberPaths);
        all_path_submit_info_wrappers.resize(numberPaths);
        allRenderFinishedSemaphores.resize(numberPaths);

        // create the command pool
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = vulkanContext.getQueueFamilyIndices().graphicsAndComputeFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    virtual ~ComputeGraph() {
        auto& device = vulkanContext.getDevice();

        // clear the outputs of all elements
        // otherwise we will have dangling pointers in the graph
        clearOutputs();

        // destroy the semaphores
        for (auto& semaphores : allRenderFinishedSemaphores) {
            for (auto& semaphore_list : semaphores) {
                for (auto& semaphore : semaphore_list.second) {
                    vkDestroySemaphore(device, semaphore.second, nullptr);
                }
            }
        }

        for (auto& semaphores : graphFinishedSemaphores) {
            vkDestroySemaphore(device, semaphores, nullptr);
        }

        for (auto& buffer : commandBuffers) {
            vkFreeCommandBuffers(device, commandPool, 1, &buffer);
        }
        vkDestroyCommandPool(device, commandPool, nullptr);

        if (profilingQueryPool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, profilingQueryPool_, nullptr);
        }
    }

    void compileFrom(ComputeGraphElementPtr element) {
        auto& device = vulkanContext.getDevice();

        computeOrder(element);

        updateOutputs();

        createRenderFinishedSemaphores();

        createGraphFinishedSemaphores();

        for (auto& element : ordered_elements) {
            element->_setup(vulkanContext, numberPaths);
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

        // Create timestamp query pool when profiling is enabled.
        // Each element gets two queries: one at the top of its CB (start)
        // and one at the bottom (end).  The pool is reset inside each CB so
        // it is re-used correctly every frame.
        if (profilingEnabled_) {
            profilingTimestampPeriodNs_ = vulkanContext.getTimestampPeriod();
            profilingAccum_.assign(ordered_elements.size(), {0.0, 0ULL});

            VkQueryPoolCreateInfo qi{};
            qi.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = 2u * (uint32_t)ordered_elements.size();
            vkCreateQueryPool(device, &qi, nullptr, &profilingQueryPool_);
        }

        for (uint32_t pathId = 0; pathId < numberPaths; pathId++) {
            for (size_t i = 0; i < ordered_elements.size(); i++) {
                auto& element = ordered_elements[i];
                VkCommandBuffer& commandBuffer = commandBuffers[i * numberPaths + pathId];
                recordCommandBuffer(commandBuffer, element, pathId, (uint32_t)i);
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
        auto& device = vulkanContext.getDevice();

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

        // Drain graphFinishedSemaphores[pathId] so repeated submitAndWait calls
        // don't double-signal it.
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo drainInfo{};
        drainInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        drainInfo.waitSemaphoreCount   = 1;
        drainInfo.pWaitSemaphores      = &finishSemaphore;
        drainInfo.pWaitDstStageMask    = &waitStage;
        drainInfo.signalSemaphoreCount = 0;
        drainInfo.commandBufferCount   = 0;
        vkQueueSubmit(graphicsQueue, 1, &drainInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        // GPU is now idle: accumulate timestamp results if profiling is on.
        readAndAccumulateTimestamps_();
    }

private:
    VulkanContext& vulkanContext;
    uint32_t numberPaths;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<ComputeGraphElementPtr> ordered_elements;

    typedef std::vector<VkSubmitInfo> SubmitInfoList;
    typedef std::vector<SubmitInfoWrapper> SubmitInfoWrapperList;

    std::vector<SubmitInfoList> all_path_submit_infos;
    std::vector<SubmitInfoWrapperList> all_path_submit_info_wrappers;

    typedef std::map<ComputeGraphElementPtr, VkSemaphore> SemaphoreMap;
    typedef std::map<ComputeGraphElementPtr, SemaphoreMap> SemaphoreMapMap;

    std::vector<SemaphoreMapMap> allRenderFinishedSemaphores;

    std::vector<VkSemaphore> graphFinishedSemaphores;

    // ---- Profiling -------------------------------------------------------
    bool         profilingEnabled_          = false;
    VkQueryPool  profilingQueryPool_        = VK_NULL_HANDLE;
    float        profilingTimestampPeriodNs_ = 1.0f;
    // Per ordered_element: {accumulated nanoseconds, sample count}
    std::vector<std::pair<double, uint64_t>> profilingAccum_;

public:
    // Call before compileFrom().
    void enableProfiling() { profilingEnabled_ = true; }

    // Returns {elementName, meanTimeMs} for every element in execution order.
    // Only meaningful after at least one submitAndWait() or after an explicit
    // readAndAccumulateTimestamps_() call following vkQueueWaitIdle().
    std::vector<std::pair<std::string, float>> getProfilingResults() const {
        std::vector<std::pair<std::string, float>> out;
        for (size_t i = 0; i < ordered_elements.size(); ++i) {
            auto& [totalNs, count] = profilingAccum_[i];
            float meanMs = (count > 0) ? float(totalNs / double(count)) * 1e-6f : 0.f;
            std::string label = ordered_elements[i]->getName();
            if (label.empty()) label = ordered_elements[i]->getType();
            out.push_back({label, meanMs});
        }
        return out;
    }

    // Read timestamp results from the GPU (GPU must be idle).
    // Accumulates into profilingAccum_ for mean computation.
    void readAndAccumulateTimestamps_() {
        if (!profilingEnabled_ || profilingQueryPool_ == VK_NULL_HANDLE) return;
        uint32_t n = (uint32_t)ordered_elements.size();
        std::vector<uint64_t> ts(2u * n, 0ULL);
        VkResult r = vkGetQueryPoolResults(
            vulkanContext.getDevice(), profilingQueryPool_,
            0, 2u * n,
            sizeof(uint64_t) * 2u * n, ts.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);     // don't use WAIT_BIT: GPU must already be idle
        if (r != VK_SUCCESS && r != VK_NOT_READY) return;
        for (uint32_t i = 0; i < n; ++i) {
            if (ts[2*i+1] >= ts[2*i]) {
                profilingAccum_[i].first  += double(ts[2*i+1] - ts[2*i]) * profilingTimestampPeriodNs_;
                profilingAccum_[i].second += 1;
            }
        }
    }

private:
    void recordCommandBuffer(VkCommandBuffer commandBuffer,
                             ComputeGraphElementPtr element,
                             uint32_t pathId,
                             uint32_t elementIdx = 0) {
        vkResetCommandBuffer(commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        if (profilingEnabled_ && profilingQueryPool_ != VK_NULL_HANDLE) {
            // Reset this element's two query slots, then write the start timestamp.
            vkCmdResetQueryPool(commandBuffer, profilingQueryPool_, 2 * elementIdx, 2);
            vkCmdWriteTimestamp(commandBuffer,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                profilingQueryPool_, 2 * elementIdx);
        }

        element->_record(commandBuffer, pathId);

        if (profilingEnabled_ && profilingQueryPool_ != VK_NULL_HANDLE) {
            // End timestamp: written after all GPU work in this CB completes.
            vkCmdWriteTimestamp(commandBuffer,
                                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                profilingQueryPool_, 2 * elementIdx + 1);
        }

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void getSubmitInfoForElement(SubmitInfoWrapper& submitInfoWrapper, uint32_t pathId, ComputeGraphElementPtr element, VkCommandBuffer* pCommandBuffer) {

        auto& submitInfo = submitInfoWrapper.submitInfo;
        auto& waitSemaphores = submitInfoWrapper.waitSemaphores;
        auto& waitStages = submitInfoWrapper.waitStages;
        auto& signalSemaphores = submitInfoWrapper.signalSemaphores;

        if (element->renderWaitSemaphores.find(pathId) != element->renderWaitSemaphores.end()) {
            waitSemaphores.push_back(element->renderWaitSemaphores[pathId]);
            waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }

        // for the element, we want to find the semaphores that connect the
        // element to its inputs (given that the element has inputs)
        // for that we have to first get the SemphoreMapMap for the pathId
        // and then find the SemaphoreMap for the element, which
        // contains the semaphore for the connection between the element and its input element
        auto& renderFinishedSemaphores = allRenderFinishedSemaphores[pathId];
        for (auto& input : element->getInputs()) {
            auto& inputElement = input.second;

            auto input_output_map_iter = renderFinishedSemaphores.find(inputElement);
            if (input_output_map_iter != renderFinishedSemaphores.end()) {
                // would be better if it would be a map
                // now find the element in the output of the input element
                auto element_iter = renderFinishedSemaphores[inputElement].find(element);
                if (element_iter != renderFinishedSemaphores[inputElement].end()) {
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
        if (input_output_map_iter != renderFinishedSemaphores.end()) {
            for (auto& outputElement : element->outputs) {
                auto element_iter = renderFinishedSemaphores[element].find(outputElement);
                if (element_iter != renderFinishedSemaphores[element].end()) {
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
        if (element->outputs.size() == 0) {
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
        auto& device = vulkanContext.getDevice();
        auto& config = vulkanContext.getConfig();

        // create the render finished semaphores
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        for (uint32_t i = 0; i < numberPaths; i++) {
            // get the render finished semaphores for this path
            auto& renderFinishedSemaphores = allRenderFinishedSemaphores[i];
            // create mulitple semaphores for each element in the path
            // (one for each output of the element)
            for (auto& element : ordered_elements) {
                for (auto& output_element : element->outputs) {
                    VkSemaphore* finishSemaphore = &renderFinishedSemaphores[element][output_element];
                    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, finishSemaphore) != VK_SUCCESS) {
                        throw std::runtime_error("failed to create render finished semaphore!");
                    }
                }
            }
        }
    }

    void createGraphFinishedSemaphores() {
        auto& device = vulkanContext.getDevice();
        auto& config = vulkanContext.getConfig();

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
        for (auto& element : ordered_elements) {
            element->outputs.clear();
        }

        // now, update the outputs of all elements
        for (auto& element : ordered_elements) {
            for (auto& input : element->getInputs()) {
                auto& inputElement = input.second;
                // Only add if element is not already in outputs
                if (std::find(inputElement->outputs.begin(), inputElement->outputs.end(), element) == inputElement->outputs.end()) {
                    inputElement->outputs.push_back(element);
                }
            }
        }
    }

    void clearOutputs() {
        for (auto& element : ordered_elements) {
            element->outputs.clear();
        }
    }

    typedef std::map<ComputeGraphElementPtr, std::vector<ComputeGraphElementPtr>> EdgeList;

    void fill_edges(EdgeList& edges, EdgeList& incoming, ComputeGraphElementPtr element) {
        for (auto& input : element->getInputs()) {
            // check if input already in graph
            auto it = find(edges[element].begin(), edges[element].end(), input.second);
            if (it == edges[element].end()) {
                // if not, add it
                edges[element].push_back(input.second);
                incoming[input.second].push_back(element);
                std::cout << "edge: " << element->getType() << "(" << element->getName() << ") -> " << input.second->getType() << "(" << input.second->getName() << ")" << std::endl;
                fill_edges(edges, incoming, input.second);
            }
        }
    }

    void computeOrder(ComputeGraphElementPtr element) {
        // Use Kahn's algorithm to find the execution order

        auto& L = ordered_elements;
        L.clear();

        std::queue<ComputeGraphElementPtr> S;
        S.push(element);

        EdgeList edges;
        EdgeList incoming_edges;
        fill_edges(edges, incoming_edges, element);

        while (!S.empty()) {
            // remove a node n from S
            auto n = S.front();
            S.pop();

            L.push_back(n);
            for (auto input : n->getInputs()) {
                // note the convention that N and M are iterators
                auto m = input.second;
                // first check if the input node is still in the graph
                auto M = find(edges[n].begin(), edges[n].end(), m);
                if (M != edges[n].end()) {
                    // if yes, remove edge N->M from the graph
                    edges[n].erase(M);
                    auto N = find(incoming_edges[m].begin(), incoming_edges[m].end(), n);
                    incoming_edges[m].erase(N);

                    // if m has no other incoming edges then
                    // insert m into S
                    if (incoming_edges[m].empty()) {
                        S.push(m);
                    }
                }
            }
        }
        size_t sum_edges = 0;
        for (auto& element : edges) {
            sum_edges += element.second.size();
        }

        size_t sum_incoming_edges = 0;
        for (auto& element : incoming_edges) {
            sum_incoming_edges += element.second.size();
        }

        if (sum_edges > 0 || sum_incoming_edges > 0) {
            throw std::runtime_error("graph has cycles!");
        }

        std::reverse(L.begin(), L.end());
    }
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_HPP