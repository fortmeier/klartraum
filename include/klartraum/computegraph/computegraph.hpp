// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_COMPUTEGRAPH_HPP
#define KLARTRAUM_COMPUTEGRAPH_COMPUTEGRAPH_HPP

#include <algorithm>
#include <cstring>
#include <iterator>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

class SubmitInfoWrapper {
public:
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkSemaphore> signalSemaphores;
    VkSubmitInfo submitInfo{};
    std::vector<VkPipelineStageFlags> waitStages;
};

class ComputeGraph {
public:
    ComputeGraph(VulkanContext& vulkanContext, uint32_t numberPaths)
        : vulkanContext(vulkanContext),
          numberPaths(numberPaths) {
        auto& device = vulkanContext.getDevice();

        pathSubmits.resize(numberPaths);

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

        for (auto& semaphores : graphFinishedSemaphores) {
            vkDestroySemaphore(device, semaphores, nullptr);
        }

        for (auto& buffer : commandBuffers) {
            vkFreeCommandBuffers(device, commandPool, 1, &buffer);
        }
        vkDestroyCommandPool(device, commandPool, nullptr);

        for (VkQueryPool pool : profilingQueryPools)
            vkDestroyQueryPool(device, pool, nullptr);
        if (perfQueryPool != VK_NULL_HANDLE)
            vkDestroyQueryPool(device, perfQueryPool, nullptr);
    }

    void compileFrom(ComputeGraphElementPtr element) {
        auto& device = vulkanContext.getDevice();

        computeOrder(element);

        updateOutputs();

        createGraphFinishedSemaphores();

        for (auto& element : ordered_elements) {
            element->_setup(vulkanContext, numberPaths);
        }

        updatable_elements.clear();
        std::copy_if(ordered_elements.begin(), ordered_elements.end(), std::back_inserter(updatable_elements),
                     [](const ComputeGraphElementPtr& e) { return e->isUpdatable(); });

        commandBuffers.resize(numberPaths);

        // create the command buffers
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        // Timestamp query pools: 2 slots per element (start/end), split into
        // pools of at most kProfilingElementsPerPool elements. MoltenVK backs a
        // timestamp pool with one MTLCounterSampleBuffer, which is limited to
        // 4096 samples; larger pools silently fall back to emulated zeros.
        if (profilingEnabled) {
            profilingTimestampPeriodNs = vulkanContext.getTimestampPeriod();
            profilingAccum.assign(ordered_elements.size(), {0.0, 0ULL});

            uint32_t remaining = (uint32_t)ordered_elements.size();
            while (remaining > 0) {
                uint32_t elements = std::min(remaining, kProfilingElementsPerPool);
                VkQueryPoolCreateInfo qi{};
                qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qi.queryCount = 2u * elements;
                VkQueryPool pool = VK_NULL_HANDLE;
                vkCreateQueryPool(device, &qi, nullptr, &pool);
                profilingQueryPools.push_back(pool);
                remaining -= elements;
            }
        }

        // Performance counter query pool.
        if (perfProfilingEnabled && !perfCounterInfos.empty()) {
            uint32_t n = (uint32_t)ordered_elements.size();
            perfAccum.assign(n * (uint32_t)perfCounterInfos.size(), {0.0, 0ULL});

            if (perfUsingHwCounters) {
                // VK_KHR_performance_query path
                uint32_t qf = vulkanContext.getQueueFamilyIndices().graphicsAndComputeFamily.value();
                VkQueryPoolPerformanceCreateInfoKHR perfCI{};
                perfCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR;
                perfCI.queueFamilyIndex = qf;
                perfCI.counterIndexCount = (uint32_t)perfCounterIndices.size();
                perfCI.pCounterIndices = perfCounterIndices.data();
                VkQueryPoolCreateInfo qi{};
                qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qi.pNext = &perfCI;
                qi.queryType = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;
                qi.queryCount = n;
                vkCreateQueryPool(device, &qi, nullptr, &perfQueryPool);
            } else {
                // Pipeline statistics fallback path: count CS invocations per element.
                VkQueryPoolCreateInfo qi{};
                qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qi.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
                qi.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
                qi.queryCount = n;
                vkCreateQueryPool(device, &qi, nullptr, &perfQueryPool);
            }
        }

        for (uint32_t pathId = 0; pathId < numberPaths; pathId++) {
            recordPath(commandBuffers[pathId], pathId);
            setupPathSubmitInfo(pathId);
        }
    }

    /*
     * Submit the graph to the graphics queue
     *
     * The submit infos will have to be prepared before by calling compile_from
     */
    VkSemaphore submitTo(VkQueue graphicsQueue, uint32_t pathId, VkFence fence = VK_NULL_HANDLE) {
        updateElements(pathId);

        if (vkQueueSubmit(graphicsQueue, 1, &pathSubmits[pathId].submitInfo, fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit the graph elements!");
        }

        return graphFinishedSemaphores[pathId];
    }

    void submitAndWait(VkQueue graphicsQueue, uint32_t pathId) {
        auto& device = vulkanContext.getDevice();

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
            throw std::runtime_error("failed to create fence!");

        if (perfProfilingEnabled && perfUsingHwCounters && perfQueryPool != VK_NULL_HANDLE) {
            // Acquire the profiling lock that serialises performance-counter collection,
            // then chain VkPerformanceQuerySubmitInfoKHR onto the path's VkSubmitInfo so
            // the driver knows this is pass 0 of the perf-query.
            VkAcquireProfilingLockInfoKHR lockInfo{};
            lockInfo.sType = VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR;
            lockInfo.timeout = UINT64_MAX;
            pfn_AcquireLock_(device, &lockInfo);

            updateElements(pathId);
            VkPerformanceQuerySubmitInfoKHR perfSubmit{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR, nullptr,
                                                       /*counterPassIndex=*/0};
            VkSubmitInfo info = pathSubmits[pathId].submitInfo;
            info.pNext = &perfSubmit;
            if (vkQueueSubmit(graphicsQueue, 1, &info, fence) != VK_SUCCESS)
                throw std::runtime_error("failed to submit perf-query command buffers!");

            const VkResult waitResult = vkWaitForFences(device, 1, &fence, true, UINT64_MAX);
            vkDestroyFence(device, fence, nullptr);
            pfn_ReleaseLock_(device);
            if (waitResult != VK_SUCCESS) {
                throw std::runtime_error(waitResult == VK_ERROR_DEVICE_LOST
                                             ? "GPU device lost while executing compute graph"
                                             : "failed waiting for compute graph fence");
            }
        } else {
            auto finishSemaphore = submitTo(graphicsQueue, pathId, fence);
            const VkResult waitResult = vkWaitForFences(device, 1, &fence, true, UINT64_MAX);
            vkDestroyFence(device, fence, nullptr);
            (void)finishSemaphore;
            if (waitResult != VK_SUCCESS) {
                throw std::runtime_error(waitResult == VK_ERROR_DEVICE_LOST
                                             ? "GPU device lost while executing compute graph"
                                             : "failed waiting for compute graph fence");
            }
        }

        // Drain graphFinishedSemaphores[pathId].
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo drainInfo{};
        drainInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        drainInfo.waitSemaphoreCount = 1;
        drainInfo.pWaitSemaphores = &graphFinishedSemaphores[pathId];
        drainInfo.pWaitDstStageMask = &waitStage;
        vkQueueSubmit(graphicsQueue, 1, &drainInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        readAndAccumulateTimestamps();
        readAndAccumulatePerformanceCounters();
    }

private:
    VulkanContext& vulkanContext;
    uint32_t numberPaths;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<ComputeGraphElementPtr> ordered_elements;
    // The elements with a host-side update (isUpdatable()), collected once.
    std::vector<ComputeGraphElementPtr> updatable_elements;

    // Runs the host-side updates for the path about to be submitted.
    void updateElements(uint32_t pathId) {
        for (auto& element : updatable_elements) {
            element->_update(pathId);
        }
    }

    // One submission per path: the path's command buffer, the external
    // semaphores its elements wait for, and the graph-finished signal.
    std::vector<SubmitInfoWrapper> pathSubmits;

    std::vector<VkSemaphore> graphFinishedSemaphores;

    // ---- Timestamp profiling ---------------------------------------------
    bool profilingEnabled = false;
    static constexpr uint32_t kProfilingElementsPerPool = 2048;
    std::vector<VkQueryPool> profilingQueryPools;
    float profilingTimestampPeriodNs = 1.0f;
    // Per ordered_element: {accumulated nanoseconds, sample count}
    std::vector<std::pair<double, uint64_t>> profilingAccum;

    // ---- Performance counter profiling -----------------------------------
    // Tries VK_KHR_performance_query first; falls back to pipeline statistics
    // (VK_QUERY_TYPE_PIPELINE_STATISTICS) which is always available.
    bool perfProfilingEnabled = false;
    bool perfUsingHwCounters = false; // true = VK_KHR_performance_query path
    VkQueryPool perfQueryPool = VK_NULL_HANDLE;
    uint32_t perfPassCount = 0;

    struct PerfCounterInfo {
        std::string name;
        VkPerformanceCounterStorageKHR storage;
    };
    std::vector<uint32_t> perfCounterIndices;
    std::vector<PerfCounterInfo> perfCounterInfos;
    // [elementIdx * numCounters + counterIdx] = {accumulated value, sample count}
    std::vector<std::pair<double, uint64_t>> perfAccum;

    PFN_vkAcquireProfilingLockKHR pfn_AcquireLock_ = nullptr;
    PFN_vkReleaseProfilingLockKHR pfn_ReleaseLock_ = nullptr;

public:
    // Call before compileFrom().
    void enableProfiling() { profilingEnabled = true; }

    // Enable VK_KHR_performance_query counter collection.  Call before compileFrom().
    // nameFilter: sub-strings to match against counter name/description.  Empty = all counters.
    // If a selected counter set requires more than 1 pass the set is trimmed to fit 1 pass.
    void enablePerformanceProfiling(std::vector<std::string> nameFilter = {}) {
        auto& instance = vulkanContext.getInstance();
        auto& device = vulkanContext.getDevice();

        auto pfnEnum = (PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)vkGetInstanceProcAddr(
            instance, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
        auto pfnPasses = (PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR)vkGetInstanceProcAddr(
            instance, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR");
        pfn_AcquireLock_ = (PFN_vkAcquireProfilingLockKHR)vkGetDeviceProcAddr(device, "vkAcquireProfilingLockKHR");
        pfn_ReleaseLock_ = (PFN_vkReleaseProfilingLockKHR)vkGetDeviceProcAddr(device, "vkReleaseProfilingLockKHR");

        if (!pfnEnum || !pfnPasses || !pfn_AcquireLock_ || !pfn_ReleaseLock_) {
            // VK_KHR_performance_query not available — fall back to pipeline statistics.
            if (!vulkanContext.isPipelineStatisticsQuerySupported()) {
                std::cout << "[ComputeGraph] Neither VK_KHR_performance_query nor pipeline"
                             " statistics queries are supported; perf profiling disabled.\n";
                return;
            }
            std::cout << "[ComputeGraph] VK_KHR_performance_query unavailable;"
                         " using pipeline statistics (CS invocations) instead.\n"
                         "              (Enable Windows Developer Mode for hardware SM counters.)\n";
            enablePipelineStatisticsProfiling();
            return;
        }

        uint32_t qf = vulkanContext.getQueueFamilyIndices().graphicsAndComputeFamily.value();

        uint32_t cnt = 0;
        pfnEnum(vulkanContext.physicalDevice, qf, &cnt, nullptr, nullptr);
        if (cnt == 0) {
            std::cerr << "[ComputeGraph] No performance counters found\n";
            return;
        }

        std::vector<VkPerformanceCounterKHR> counters(cnt, {VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR});
        std::vector<VkPerformanceCounterDescriptionKHR> descs(cnt,
                                                              {VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR});
        pfnEnum(vulkanContext.physicalDevice, qf, &cnt, counters.data(), descs.data());

        for (uint32_t i = 0; i < cnt; ++i) {
            std::string name = descs[i].name;
            std::string desc = descs[i].description;
            bool match = nameFilter.empty();
            if (!match) {
                for (auto& f : nameFilter) {
                    if (name.find(f) != std::string::npos || desc.find(f) != std::string::npos) {
                        match = true;
                        break;
                    }
                }
            }
            if (match) {
                perfCounterIndices.push_back(i);
                perfCounterInfos.push_back({name, counters[i].storage});
            }
        }

        if (perfCounterIndices.empty()) {
            std::cerr << "[ComputeGraph] No counters match filter\n";
            return;
        }

        // Trim counter set until it fits in a single pass.
        auto queryPassCount = [&]() {
            VkQueryPoolPerformanceCreateInfoKHR ci{};
            ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR;
            ci.queueFamilyIndex = qf;
            ci.counterIndexCount = (uint32_t)perfCounterIndices.size();
            ci.pCounterIndices = perfCounterIndices.data();
            uint32_t passes = 0;
            pfnPasses(vulkanContext.physicalDevice, &ci, &passes);
            return passes;
        };

        perfPassCount = queryPassCount();
        while (perfPassCount > 1 && perfCounterIndices.size() > 1) {
            perfCounterIndices.pop_back();
            perfCounterInfos.pop_back();
            perfPassCount = queryPassCount();
        }
        if (perfPassCount > 1) {
            std::cerr << "[ComputeGraph] Cannot fit any counter in 1 pass; disabling perf profiling\n";
            perfCounterIndices.clear();
            perfCounterInfos.clear();
            return;
        }

        std::cout << "[ComputeGraph] Hardware performance counters enabled (" << perfCounterIndices.size()
                  << " counter(s), " << perfPassCount << " pass):\n";
        for (auto& ci : perfCounterInfos)
            std::cout << "  " << ci.name << "\n";

        perfProfilingEnabled = true;
        perfUsingHwCounters = true;
    }

    // Returns {elementName, meanTimeMs} for timestamp entries, followed by
    // {elementName " [" counterName "]", meanValue} for performance counter entries.
    std::vector<std::pair<std::string, float>> getProfilingResults() const {
        std::vector<std::pair<std::string, float>> out;

        auto label = [&](size_t i) {
            std::string l = ordered_elements[i]->getName();
            return l.empty() ? std::string(ordered_elements[i]->getType()) : l;
        };

        for (size_t i = 0; i < ordered_elements.size(); ++i) {
            auto& [totalNs, count] = profilingAccum[i];
            float meanMs = (count > 0) ? float(totalNs / double(count)) * 1e-6f : 0.f;
            out.push_back({label(i), meanMs});
        }

        if (perfProfilingEnabled && !perfCounterInfos.empty()) {
            uint32_t nc = (uint32_t)perfCounterInfos.size();
            // Distinguish unit: HW counters keep their own unit label;
            // pipeline-statistics path reports raw invocation counts (not ms).
            std::string unit = perfUsingHwCounters ? "" : "";
            for (size_t i = 0; i < ordered_elements.size(); ++i) {
                for (uint32_t c = 0; c < nc; ++c) {
                    auto& [sum, count] = perfAccum[i * nc + c];
                    float mean = (count > 0) ? float(sum / double(count)) : 0.f;
                    out.push_back({label(i) + " [" + perfCounterInfos[c].name + unit + "]", mean});
                }
            }
        }
        return out;
    }

    void readAndAccumulateTimestamps() {
        if (!profilingEnabled || profilingQueryPools.empty())
            return;
        uint32_t n = (uint32_t)ordered_elements.size();
        std::vector<uint64_t> ts(2u * n, 0ULL);
        for (size_t p = 0; p < profilingQueryPools.size(); ++p) {
            uint32_t first = (uint32_t)p * kProfilingElementsPerPool;
            uint32_t elements = std::min(n - first, kProfilingElementsPerPool);
            VkResult r = vkGetQueryPoolResults(vulkanContext.getDevice(), profilingQueryPools[p], 0, 2u * elements,
                                               sizeof(uint64_t) * 2u * elements, ts.data() + 2u * first,
                                               sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            if (r != VK_SUCCESS && r != VK_NOT_READY)
                return;
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (ts[2 * i + 1] >= ts[2 * i]) {
                profilingAccum[i].first += double(ts[2 * i + 1] - ts[2 * i]) * profilingTimestampPeriodNs;
                profilingAccum[i].second += 1;
            }
        }
    }

    void readAndAccumulatePerformanceCounters() {
        if (!perfProfilingEnabled || perfQueryPool == VK_NULL_HANDLE)
            return;
        uint32_t n = (uint32_t)ordered_elements.size();

        if (perfUsingHwCounters) {
            uint32_t nc = (uint32_t)perfCounterIndices.size();
            size_t stride = sizeof(VkPerformanceCounterResultKHR) * nc;
            std::vector<VkPerformanceCounterResultKHR> raw(n * nc);
            VkResult r = vkGetQueryPoolResults(vulkanContext.getDevice(), perfQueryPool, 0, n, stride * n, raw.data(),
                                               stride, VK_QUERY_RESULT_WAIT_BIT);
            if (r != VK_SUCCESS)
                return;
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t c = 0; c < nc; ++c) {
                    double val = extractCounterValue(raw[i * nc + c], perfCounterInfos[c].storage);
                    auto& acc = perfAccum[i * nc + c];
                    acc.first += val;
                    acc.second += 1;
                }
        } else {
            // Pipeline statistics: one uint64 per element (CS invocations).
            std::vector<uint64_t> raw(n, 0);
            VkResult r =
                vkGetQueryPoolResults(vulkanContext.getDevice(), perfQueryPool, 0, n, sizeof(uint64_t) * n, raw.data(),
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (r != VK_SUCCESS)
                return;
            for (uint32_t i = 0; i < n; ++i) {
                auto& acc = perfAccum[i];
                acc.first += (double)raw[i];
                acc.second += 1;
            }
        }
    }

    // Called when VK_KHR_performance_query is unavailable.  Sets up pipeline statistics
    // queries (CS invocation counts) which are always available in core Vulkan.
    void enablePipelineStatisticsProfiling() {
        // One synthetic "counter": compute shader invocations.
        perfCounterInfos.push_back({"CS invocations", VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR});
        perfProfilingEnabled = true;
        perfUsingHwCounters = false;
        perfPassCount = 1;
        // perfQueryPool is created in compileFrom() once element count is known.
    }

    static double extractCounterValue(const VkPerformanceCounterResultKHR& r, VkPerformanceCounterStorageKHR storage) {
        switch (storage) {
        case VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR:
            return r.int32;
        case VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR:
            return (double)r.int64;
        case VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR:
            return r.uint32;
        case VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR:
            return (double)r.uint64;
        case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR:
            return r.float32;
        case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR:
            return r.float64;
        default:
            return 0.0;
        }
    }

private:
    // Records every element of the path, in topological order, into one
    // command buffer. A full memory barrier separates consecutive elements, so
    // each element sees all writes of the elements before it; this is what
    // orders producers before consumers.
    void recordPath(VkCommandBuffer commandBuffer, uint32_t pathId) {
        vkResetCommandBuffer(commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        for (size_t i = 0; i < ordered_elements.size(); i++) {
            if (i > 0) {
                VkMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            }
            recordElement(commandBuffer, ordered_elements[i], pathId, (uint32_t)i);
        }

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void recordElement(VkCommandBuffer commandBuffer, ComputeGraphElementPtr element, uint32_t pathId,
                       uint32_t elementIdx) {
        VkQueryPool timestampPool = VK_NULL_HANDLE;
        uint32_t timestampQuery = 0;
        if (profilingEnabled && elementIdx / kProfilingElementsPerPool < profilingQueryPools.size()) {
            timestampPool = profilingQueryPools[elementIdx / kProfilingElementsPerPool];
            timestampQuery = 2 * (elementIdx % kProfilingElementsPerPool);
        }

        if (timestampPool != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(commandBuffer, timestampPool, timestampQuery, 2);
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool, timestampQuery);
        }

        if (perfProfilingEnabled && perfQueryPool != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(commandBuffer, perfQueryPool, elementIdx, 1);
            vkCmdBeginQuery(commandBuffer, perfQueryPool, elementIdx, 0);
        }

        element->_record(commandBuffer, pathId);

        if (perfProfilingEnabled && perfQueryPool != VK_NULL_HANDLE)
            vkCmdEndQuery(commandBuffer, perfQueryPool, elementIdx);

        if (timestampPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool, timestampQuery + 1);
        }
    }

    // The path's submission waits for every external semaphore any of its
    // elements was given (e.g. swapchain image acquisition) and signals the
    // graph-finished semaphore once the whole command buffer has executed.
    void setupPathSubmitInfo(uint32_t pathId) {
        auto& wrapper = pathSubmits[pathId];
        wrapper.waitSemaphores.clear();
        wrapper.waitStages.clear();
        wrapper.signalSemaphores.clear();

        for (auto& element : ordered_elements) {
            auto it = element->renderWaitSemaphores.find(pathId);
            if (it == element->renderWaitSemaphores.end())
                continue;
            if (std::find(wrapper.waitSemaphores.begin(), wrapper.waitSemaphores.end(), it->second) ==
                wrapper.waitSemaphores.end()) {
                wrapper.waitSemaphores.push_back(it->second);
                wrapper.waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
        }
        wrapper.signalSemaphores.push_back(graphFinishedSemaphores[pathId]);

        auto& submitInfo = wrapper.submitInfo;
        submitInfo = VkSubmitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[pathId];
        submitInfo.waitSemaphoreCount = (uint32_t)wrapper.waitSemaphores.size();
        submitInfo.pWaitSemaphores = wrapper.waitSemaphores.data();
        submitInfo.pWaitDstStageMask = wrapper.waitStages.data();
        submitInfo.signalSemaphoreCount = (uint32_t)wrapper.signalSemaphores.size();
        submitInfo.pSignalSemaphores = wrapper.signalSemaphores.data();
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
            for (auto& inputElement : getPredecessors(element)) {
                // Only add if element is not already in outputs
                if (std::find(inputElement->outputs.begin(), inputElement->outputs.end(), element) ==
                    inputElement->outputs.end()) {
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

    std::vector<ComputeGraphElementPtr> getPredecessors(const ComputeGraphElementPtr& element) const {
        std::vector<ComputeGraphElementPtr> predecessors;
        for (const auto& input : element->getInputs()) {
            if (std::find(predecessors.begin(), predecessors.end(), input.second) == predecessors.end()) {
                predecessors.push_back(input.second);
            }
        }
        for (const auto& dependency : element->getDependencies()) {
            if (std::find(predecessors.begin(), predecessors.end(), dependency) == predecessors.end()) {
                predecessors.push_back(dependency);
            }
        }
        return predecessors;
    }

    void fill_edges(EdgeList& edges, EdgeList& incoming, ComputeGraphElementPtr element) {
        for (auto& input : getPredecessors(element)) {
            // check if input already in graph
            auto it = find(edges[element].begin(), edges[element].end(), input);
            if (it == edges[element].end()) {
                // if not, add it
                edges[element].push_back(input);
                incoming[input].push_back(element);
                fill_edges(edges, incoming, input);
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
            for (auto input : getPredecessors(n)) {
                // note the convention that N and M are iterators
                auto m = input;
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

#endif // KLARTRAUM_COMPUTEGRAPH_COMPUTEGRAPH_HPP
