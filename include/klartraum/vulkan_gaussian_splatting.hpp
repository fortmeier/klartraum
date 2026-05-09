#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/rendergraphelement.hpp"
#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

class VulkanGaussianSplatting : virtual public RenderGraphElement,
                                virtual public ComputeGraphGroup {
public:
    VulkanGaussianSplatting(
        VulkanContext& vulkanContext,
        std::shared_ptr<ImageViewSrc> imageViewSrc,
        std::shared_ptr<CameraUboType> cameraUBO,
        std::string path,
        GsplatConfig config = GsplatConfig{});

    VulkanGaussianSplatting(
        VulkanContext& vulkanContext,
        std::shared_ptr<ImageViewSrc> imageViewSrc,
        std::shared_ptr<CameraUboType> cameraUBO,
        std::vector<Gaussian3D> gaussians,
        GsplatConfig config = GsplatConfig{});

    ~VulkanGaussianSplatting();

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override;
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override;
    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

    virtual const char* getType() const override { return "GaussianSplatting"; }

private:
    void loadSPZModel(std::string path);
    void initialize(VulkanContext& vulkanContext,
                    std::shared_ptr<ImageViewSrc> imageViewSrc,
                    std::shared_ptr<CameraUboType> cameraUBO,
                    GsplatConfig config);

    GsplatConfig config_;

    VulkanContext* vulkanContext = nullptr;
    uint32_t number_of_gaussians = 0;
    uint32_t numberOfPaths       = 0;

    std::vector<Gaussian3D> gaussians3DData;

    // SoA 3D input buffers (single-path: static data)
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec3>>> buf3DPos;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec4>>> buf3DRot;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec3>>> buf3DScale;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec4>>> buf3DColAlpha;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<float>>>     buf3DShR;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<float>>>     buf3DShG;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<float>>>     buf3DShB;

    // Pipeline stages
    std::shared_ptr<GaussianProjection>     project3Dto2D;
    std::shared_ptr<GaussianBinningCount>   binCount;
    std::shared_ptr<GeneralComputation<>>   binPrefixSum;
    std::shared_ptr<GaussianBinningScatter> binScatter;
    std::shared_ptr<GeneralComputation<>>   extractSortKeys;
    std::shared_ptr<RadixSort>              sortOp;
    std::shared_ptr<GeneralComputation<>>   gatherSorted;
    std::shared_ptr<GeneralComputation<>>   computeBounds;
    std::shared_ptr<GaussianSplatting>      splat;

    // Sort ping-pong value buffers (stored as members for _record pre-fill)
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> sortRadixValA;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> sortRadixValB;
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP
