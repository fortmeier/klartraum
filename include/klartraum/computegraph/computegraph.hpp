#ifndef KLARTRAUM_COMPUTEGRAPH_HPP
#define KLARTRAUM_COMPUTEGRAPH_HPP

#include <cstring>
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

        if (profilingQueryPool_ != VK_NULL_HANDLE)
            vkDestroyQueryPool(device, profilingQueryPool_, nullptr);
        if (perfQueryPool_ != VK_NULL_HANDLE)
            vkDestroyQueryPool(device, perfQueryPool_, nullptr);
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

        // Timestamp query pool: 2 slots per element (start/end).
        if (profilingEnabled_) {
            profilingTimestampPeriodNs_ = vulkanContext.getTimestampPeriod();
            profilingAccum_.assign(ordered_elements.size(), {0.0, 0ULL});

            VkQueryPoolCreateInfo qi{};
            qi.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = 2u * (uint32_t)ordered_elements.size();
            vkCreateQueryPool(device, &qi, nullptr, &profilingQueryPool_);
        }

        // Performance counter query pool.
        if (perfProfilingEnabled_ && !perfCounterInfos_.empty()) {
            uint32_t n  = (uint32_t)ordered_elements.size();
            perfAccum_.assign(n * (uint32_t)perfCounterInfos_.size(), {0.0, 0ULL});

            if (perfUsingHwCounters_) {
                // VK_KHR_performance_query path
                uint32_t qf = vulkanContext.getQueueFamilyIndices().graphicsAndComputeFamily.value();
                VkQueryPoolPerformanceCreateInfoKHR perfCI{};
                perfCI.sType             = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR;
                perfCI.queueFamilyIndex  = qf;
                perfCI.counterIndexCount = (uint32_t)perfCounterIndices_.size();
                perfCI.pCounterIndices   = perfCounterIndices_.data();
                VkQueryPoolCreateInfo qi{};
                qi.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qi.pNext      = &perfCI;
                qi.queryType  = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;
                qi.queryCount = n;
                vkCreateQueryPool(device, &qi, nullptr, &perfQueryPool_);
            } else {
                // Pipeline statistics fallback path: count CS invocations per element.
                VkQueryPoolCreateInfo qi{};
                qi.sType              = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qi.queryType          = VK_QUERY_TYPE_PIPELINE_STATISTICS;
                qi.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
                qi.queryCount         = n;
                vkCreateQueryPool(device, &qi, nullptr, &perfQueryPool_);
            }
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
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
            throw std::runtime_error("failed to create fence!");

        if (perfProfilingEnabled_ && perfUsingHwCounters_ && perfQueryPool_ != VK_NULL_HANDLE) {
            // Acquire the profiling lock that serialises performance-counter collection,
            // then chain VkPerformanceQuerySubmitInfoKHR onto every VkSubmitInfo so the
            // driver knows this is pass 0 of the perf-query.
            VkAcquireProfilingLockInfoKHR lockInfo{};
            lockInfo.sType   = VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR;
            lockInfo.timeout = UINT64_MAX;
            pfn_AcquireLock_(device, &lockInfo);

            auto& origInfos = all_path_submit_infos[pathId];
            std::vector<VkPerformanceQuerySubmitInfoKHR> perfSubmits(origInfos.size());
            std::vector<VkSubmitInfo> infos = origInfos;
            for (size_t i = 0; i < infos.size(); ++i) {
                perfSubmits[i] = {VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR,
                                  nullptr, /*counterPassIndex=*/0};
                infos[i].pNext = &perfSubmits[i];
            }
            if (vkQueueSubmit(graphicsQueue, (uint32_t)infos.size(), infos.data(), fence) != VK_SUCCESS)
                throw std::runtime_error("failed to submit perf-query command buffers!");

            vkWaitForFences(device, 1, &fence, true, UINT64_MAX);
            vkDestroyFence(device, fence, nullptr);
            pfn_ReleaseLock_(device);
        } else {
            auto finishSemaphore = submitTo(graphicsQueue, pathId, fence);
            vkWaitForFences(device, 1, &fence, true, UINT64_MAX);
            vkDestroyFence(device, fence, nullptr);
            (void)finishSemaphore;
        }

        // Drain graphFinishedSemaphores[pathId].
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo drainInfo{};
        drainInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        drainInfo.waitSemaphoreCount = 1;
        drainInfo.pWaitSemaphores    = &graphFinishedSemaphores[pathId];
        drainInfo.pWaitDstStageMask  = &waitStage;
        vkQueueSubmit(graphicsQueue, 1, &drainInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        readAndAccumulateTimestamps_();
        readAndAccumulatePerformanceCounters_();
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

    // ---- Timestamp profiling ---------------------------------------------
    bool         profilingEnabled_           = false;
    VkQueryPool  profilingQueryPool_         = VK_NULL_HANDLE;
    float        profilingTimestampPeriodNs_ = 1.0f;
    // Per ordered_element: {accumulated nanoseconds, sample count}
    std::vector<std::pair<double, uint64_t>> profilingAccum_;

    // ---- Performance counter profiling -----------------------------------
    // Tries VK_KHR_performance_query first; falls back to pipeline statistics
    // (VK_QUERY_TYPE_PIPELINE_STATISTICS) which is always available.
    bool        perfProfilingEnabled_  = false;
    bool        perfUsingHwCounters_   = false;  // true = VK_KHR_performance_query path
    VkQueryPool perfQueryPool_         = VK_NULL_HANDLE;
    uint32_t    perfPassCount_         = 0;

    struct PerfCounterInfo {
        std::string                    name;
        VkPerformanceCounterStorageKHR storage;
    };
    std::vector<uint32_t>        perfCounterIndices_;
    std::vector<PerfCounterInfo> perfCounterInfos_;
    // [elementIdx * numCounters + counterIdx] = {accumulated value, sample count}
    std::vector<std::pair<double, uint64_t>> perfAccum_;

    PFN_vkAcquireProfilingLockKHR pfn_AcquireLock_ = nullptr;
    PFN_vkReleaseProfilingLockKHR pfn_ReleaseLock_ = nullptr;

public:
    // Call before compileFrom().
    void enableProfiling() { profilingEnabled_ = true; }

    // Enable VK_KHR_performance_query counter collection.  Call before compileFrom().
    // nameFilter: sub-strings to match against counter name/description.  Empty = all counters.
    // If a selected counter set requires more than 1 pass the set is trimmed to fit 1 pass.
    void enablePerformanceProfiling(std::vector<std::string> nameFilter = {}) {
        auto& instance = vulkanContext.getInstance();
        auto& device   = vulkanContext.getDevice();

        auto pfnEnum = (PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)
            vkGetInstanceProcAddr(instance,
                "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
        auto pfnPasses = (PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR)
            vkGetInstanceProcAddr(instance,
                "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR");
        pfn_AcquireLock_ = (PFN_vkAcquireProfilingLockKHR)
            vkGetDeviceProcAddr(device, "vkAcquireProfilingLockKHR");
        pfn_ReleaseLock_ = (PFN_vkReleaseProfilingLockKHR)
            vkGetDeviceProcAddr(device, "vkReleaseProfilingLockKHR");

        if (!pfnEnum || !pfnPasses || !pfn_AcquireLock_ || !pfn_ReleaseLock_) {
            // VK_KHR_performance_query not available — fall back to pipeline statistics.
            std::cout << "[ComputeGraph] VK_KHR_performance_query unavailable;"
                         " using pipeline statistics (CS invocations) instead.\n"
                         "              (Enable Windows Developer Mode for hardware SM counters.)\n";
            enablePipelineStatisticsProfiling_();
            return;
        }

        uint32_t qf = vulkanContext.getQueueFamilyIndices().graphicsAndComputeFamily.value();

        uint32_t cnt = 0;
        pfnEnum(vulkanContext.physicalDevice, qf, &cnt, nullptr, nullptr);
        if (cnt == 0) { std::cerr << "[ComputeGraph] No performance counters found\n"; return; }

        std::vector<VkPerformanceCounterKHR>            counters(cnt,
            {VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR});
        std::vector<VkPerformanceCounterDescriptionKHR> descs(cnt,
            {VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR});
        pfnEnum(vulkanContext.physicalDevice, qf, &cnt, counters.data(), descs.data());

        for (uint32_t i = 0; i < cnt; ++i) {
            std::string name = descs[i].name;
            std::string desc = descs[i].description;
            bool match = nameFilter.empty();
            if (!match) {
                for (auto& f : nameFilter) {
                    if (name.find(f) != std::string::npos ||
                        desc.find(f) != std::string::npos) { match = true; break; }
                }
            }
            if (match) {
                perfCounterIndices_.push_back(i);
                perfCounterInfos_.push_back({name, counters[i].storage});
            }
        }

        if (perfCounterIndices_.empty()) {
            std::cerr << "[ComputeGraph] No counters match filter\n"; return;
        }

        // Trim counter set until it fits in a single pass.
        auto queryPassCount = [&]() {
            VkQueryPoolPerformanceCreateInfoKHR ci{};
            ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR;
            ci.queueFamilyIndex  = qf;
            ci.counterIndexCount = (uint32_t)perfCounterIndices_.size();
            ci.pCounterIndices   = perfCounterIndices_.data();
            uint32_t passes = 0;
            pfnPasses(vulkanContext.physicalDevice, &ci, &passes);
            return passes;
        };

        perfPassCount_ = queryPassCount();
        while (perfPassCount_ > 1 && perfCounterIndices_.size() > 1) {
            perfCounterIndices_.pop_back();
            perfCounterInfos_.pop_back();
            perfPassCount_ = queryPassCount();
        }
        if (perfPassCount_ > 1) {
            std::cerr << "[ComputeGraph] Cannot fit any counter in 1 pass; disabling perf profiling\n";
            perfCounterIndices_.clear(); perfCounterInfos_.clear(); return;
        }

        std::cout << "[ComputeGraph] Hardware performance counters enabled ("
                  << perfCounterIndices_.size() << " counter(s), "
                  << perfPassCount_ << " pass):\n";
        for (auto& ci : perfCounterInfos_)
            std::cout << "  " << ci.name << "\n";

        perfProfilingEnabled_ = true;
        perfUsingHwCounters_  = true;
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
            auto& [totalNs, count] = profilingAccum_[i];
            float meanMs = (count > 0) ? float(totalNs / double(count)) * 1e-6f : 0.f;
            out.push_back({label(i), meanMs});
        }

        if (perfProfilingEnabled_ && !perfCounterInfos_.empty()) {
            uint32_t nc     = (uint32_t)perfCounterInfos_.size();
            // Distinguish unit: HW counters keep their own unit label;
            // pipeline-statistics path reports raw invocation counts (not ms).
            std::string unit = perfUsingHwCounters_ ? "" : "";
            for (size_t i = 0; i < ordered_elements.size(); ++i) {
                for (uint32_t c = 0; c < nc; ++c) {
                    auto& [sum, count] = perfAccum_[i * nc + c];
                    float mean = (count > 0) ? float(sum / double(count)) : 0.f;
                    out.push_back({label(i) + " [" + perfCounterInfos_[c].name + unit + "]", mean});
                }
            }
        }
        return out;
    }

    void readAndAccumulateTimestamps_() {
        if (!profilingEnabled_ || profilingQueryPool_ == VK_NULL_HANDLE) return;
        uint32_t n = (uint32_t)ordered_elements.size();
        std::vector<uint64_t> ts(2u * n, 0ULL);
        VkResult r = vkGetQueryPoolResults(
            vulkanContext.getDevice(), profilingQueryPool_,
            0, 2u * n,
            sizeof(uint64_t) * 2u * n, ts.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (r != VK_SUCCESS && r != VK_NOT_READY) return;
        for (uint32_t i = 0; i < n; ++i) {
            if (ts[2*i+1] >= ts[2*i]) {
                profilingAccum_[i].first  += double(ts[2*i+1] - ts[2*i]) * profilingTimestampPeriodNs_;
                profilingAccum_[i].second += 1;
            }
        }
    }

    void readAndAccumulatePerformanceCounters_() {
        if (!perfProfilingEnabled_ || perfQueryPool_ == VK_NULL_HANDLE) return;
        uint32_t n  = (uint32_t)ordered_elements.size();

        if (perfUsingHwCounters_) {
            uint32_t nc = (uint32_t)perfCounterIndices_.size();
            size_t stride = sizeof(VkPerformanceCounterResultKHR) * nc;
            std::vector<VkPerformanceCounterResultKHR> raw(n * nc);
            VkResult r = vkGetQueryPoolResults(
                vulkanContext.getDevice(), perfQueryPool_,
                0, n, stride * n, raw.data(), stride,
                VK_QUERY_RESULT_WAIT_BIT);
            if (r != VK_SUCCESS) return;
            for (uint32_t i = 0; i < n; ++i)
                for (uint32_t c = 0; c < nc; ++c) {
                    double val = extractCounterValue_(raw[i*nc+c], perfCounterInfos_[c].storage);
                    auto& acc = perfAccum_[i*nc+c];
                    acc.first += val; acc.second += 1;
                }
        } else {
            // Pipeline statistics: one uint64 per element (CS invocations).
            std::vector<uint64_t> raw(n, 0);
            VkResult r = vkGetQueryPoolResults(
                vulkanContext.getDevice(), perfQueryPool_,
                0, n, sizeof(uint64_t) * n, raw.data(), sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (r != VK_SUCCESS) return;
            for (uint32_t i = 0; i < n; ++i) {
                auto& acc = perfAccum_[i];
                acc.first += (double)raw[i]; acc.second += 1;
            }
        }
    }

    // Called when VK_KHR_performance_query is unavailable.  Sets up pipeline statistics
    // queries (CS invocation counts) which are always available in core Vulkan.
    void enablePipelineStatisticsProfiling_() {
        // One synthetic "counter": compute shader invocations.
        perfCounterInfos_.push_back({"CS invocations",
                                      VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR});
        perfProfilingEnabled_ = true;
        perfUsingHwCounters_  = false;
        perfPassCount_        = 1;
        // perfQueryPool_ is created in compileFrom() once element count is known.
    }

    static double extractCounterValue_(const VkPerformanceCounterResultKHR& r,
                                       VkPerformanceCounterStorageKHR storage) {
        switch (storage) {
            case VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR:   return r.int32;
            case VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR:   return (double)r.int64;
            case VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR:  return r.uint32;
            case VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR:  return (double)r.uint64;
            case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR: return r.float32;
            case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR: return r.float64;
            default: return 0.0;
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
            vkCmdResetQueryPool(commandBuffer, profilingQueryPool_, 2 * elementIdx, 2);
            vkCmdWriteTimestamp(commandBuffer,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                profilingQueryPool_, 2 * elementIdx);
        }

        if (perfProfilingEnabled_ && perfQueryPool_ != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(commandBuffer, perfQueryPool_, elementIdx, 1);
            vkCmdBeginQuery(commandBuffer, perfQueryPool_, elementIdx, 0);
        }

        element->_record(commandBuffer, pathId);

        if (perfProfilingEnabled_ && perfQueryPool_ != VK_NULL_HANDLE)
            vkCmdEndQuery(commandBuffer, perfQueryPool_, elementIdx);

        if (profilingEnabled_ && profilingQueryPool_ != VK_NULL_HANDLE) {
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