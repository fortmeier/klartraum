// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_GRAPHCHAIN_HPP
#define KLARTRAUM_COMPUTEGRAPH_GRAPHCHAIN_HPP

#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"

namespace klartraum {

/**
 * @brief Compiled graphs submitted together, one after another on the GPU.
 *
 * The links of the chain go to the queue in one vkQueueSubmit: link i + 1
 * waits for a semaphore link i signals, and a fence is signalled when the last
 * link has finished. The CPU does nothing between the links, so data passes
 * from graph to graph on the GPU only.
 *
 * A graph may appear several times (e.g. one denoising step per link) if it was
 * compiled after ComputeGraph::enableSimultaneousUse(). The host-side updates
 * of every graph run once per submit(), before the chain is submitted. The
 * graphs must not wait for semaphores of their own (ComputeGraph::setWaitFor()).
 */
class GraphChain {
public:
    explicit GraphChain(VulkanContext& vulkanContext)
        : vulkanContext(vulkanContext) {}

    GraphChain(const GraphChain&) = delete;
    GraphChain& operator=(const GraphChain&) = delete;

    ~GraphChain() {
        auto& device = vulkanContext.getDevice();
        if (fence != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkDestroyFence(device, fence, nullptr);
        }
        for (VkSemaphore semaphore : semaphores) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
    }

    /// Appends a path of a compiled graph to the chain.
    void add(ComputeGraph& graph, uint32_t pathId = 0) {
        for (const auto& link : links) {
            if (link.first == &graph && link.second == pathId && !graph.allowsSimultaneousUse()) {
                throw std::runtime_error("GraphChain: a graph appears more than once but was not compiled after "
                                         "enableSimultaneousUse()");
            }
        }
        if (graph.hasExternalWaits(pathId)) {
            throw std::runtime_error("GraphChain: the graph waits for semaphores of its own");
        }
        links.emplace_back(&graph, pathId);
    }

    size_t size() const { return links.size(); }

    /// Submits the whole chain to `queue` in one vkQueueSubmit; wait() waits for it.
    void submit(VkQueue queue) {
        if (links.empty()) {
            throw std::runtime_error("GraphChain: nothing to submit");
        }
        wait();
        createSync();
        std::set<std::pair<ComputeGraph*, uint32_t>> prepared;
        for (const auto& link : links) {
            if (prepared.insert(link).second) {
                link.first->prepareSubmit(queue, link.second);
            }
        }
        std::vector<VkSubmitInfo> submits(links.size());
        std::vector<VkCommandBuffer> commandBuffers(links.size());
        for (size_t i = 0; i < links.size(); ++i) {
            commandBuffers[i] = links[i].first->getCommandBuffer(links[i].second);
            VkSubmitInfo& info = submits[i];
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            info.commandBufferCount = 1;
            info.pCommandBuffers = &commandBuffers[i];
            if (i > 0) {
                info.waitSemaphoreCount = 1;
                info.pWaitSemaphores = &semaphores[i - 1];
                info.pWaitDstStageMask = &waitStage;
            }
            if (i + 1 < links.size()) {
                info.signalSemaphoreCount = 1;
                info.pSignalSemaphores = &semaphores[i];
            }
        }
        vkResetFences(vulkanContext.getDevice(), 1, &fence);
        if (vulkanContext.queueSubmit(queue, uint32_t(submits.size()), submits.data(), fence) != VK_SUCCESS) {
            throw std::runtime_error("GraphChain: failed to submit the chain");
        }
        pending = true;
    }

    /// Waits until the last submitted chain has finished.
    void wait() {
        if (!pending) {
            return;
        }
        const VkResult result = vkWaitForFences(vulkanContext.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX);
        pending = false;
        if (result != VK_SUCCESS) {
            throw std::runtime_error(result == VK_ERROR_DEVICE_LOST ? "GPU device lost while executing a graph chain"
                                                                    : "failed waiting for a graph chain");
        }
    }

    void submitAndWait(VkQueue queue) {
        submit(queue);
        wait();
    }

private:
    void createSync() {
        auto& device = vulkanContext.getDevice();
        if (fence == VK_NULL_HANDLE) {
            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
                throw std::runtime_error("GraphChain: failed to create a fence");
            }
        }
        while (semaphores.size() + 1 < links.size()) {
            VkSemaphoreCreateInfo semaphoreInfo{};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkSemaphore semaphore;
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
                throw std::runtime_error("GraphChain: failed to create a semaphore");
            }
            semaphores.push_back(semaphore);
        }
    }

    VulkanContext& vulkanContext;
    std::vector<std::pair<ComputeGraph*, uint32_t>> links;
    // semaphores[i] is signalled by link i and waited for by link i + 1.
    std::vector<VkSemaphore> semaphores;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_GRAPHCHAIN_HPP
