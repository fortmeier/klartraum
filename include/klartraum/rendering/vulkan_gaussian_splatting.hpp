// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RENDERING_VULKAN_GAUSSIAN_SPLATTING_HPP
#define KLARTRAUM_RENDERING_VULKAN_GAUSSIAN_SPLATTING_HPP

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/rendering/gaussian_splatting_interface.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

class VulkanGaussianSplatting : public GaussianSplattingInterface {
public:
    VulkanGaussianSplatting(VulkanContext& vulkanContext, std::shared_ptr<ImageViewSrc> imageViewSrc,
                            std::shared_ptr<CameraUboType> cameraUBO, GaussianSoABuffers buffers,
                            GsplatConfig config = GsplatConfig{});

    ~VulkanGaussianSplatting() override;

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override;
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths,
                        VkQueue setupQueue = VK_NULL_HANDLE) override;
    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

    virtual const char* getType() const override { return "GaussianSplatting"; }

    std::shared_ptr<DepthBuffer> getDepth() const override { return depth; }

private:
    friend class VulkanGaussianSplattingRaster;
    VulkanGaussianSplatting(VulkanContext& context, std::shared_ptr<ImageViewSrc> target,
                            std::shared_ptr<CameraUboType> camera, GaussianSoABuffers buffers, GsplatConfig config,
                            bool depthOnly);
    void initialize(VulkanContext& vulkanContext, std::shared_ptr<ImageViewSrc> imageViewSrc,
                    std::shared_ptr<CameraUboType> cameraUBO, GsplatConfig config);

    GsplatConfig config;

    VulkanContext* vulkanContext = nullptr;
    uint32_t numberOfPaths = 0;

    // Static SoA input buffers (position, rotation, scale, colour+alpha, SH
    // R/G/B) + splat count, injected by the caller. Held as shared handles so a
    // single upload can be shared across backends/viewports without reloading.
    GaussianSoABuffers buffers;

    // Pipeline stages
    std::shared_ptr<GaussianProjection> project3Dto2D;
    std::shared_ptr<GaussianBinningCount> binCount;
    std::shared_ptr<GeneralComputation<>> binPrefixSum;
    std::shared_ptr<GaussianBinningScatter> binScatter;
    std::shared_ptr<GeneralComputation<>> extractSortKeys;
    std::shared_ptr<RadixSort> sortOp;
    std::shared_ptr<GeneralComputation<>> gatherSorted;
    std::shared_ptr<GeneralComputation<>> computeBounds;
    std::shared_ptr<GaussianSplatting> splat;
    std::shared_ptr<DepthBuffer> depth;
    std::shared_ptr<DepthTransfer> depthRead;
    std::shared_ptr<DepthTransfer> depthWrite;
    bool depthOnly = false;

    // Sort ping-pong value buffers (stored as members for _record pre-fill)
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> sortRadixValA;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> sortRadixValB;
};

} // namespace klartraum

#endif // KLARTRAUM_RENDERING_VULKAN_GAUSSIAN_SPLATTING_HPP
