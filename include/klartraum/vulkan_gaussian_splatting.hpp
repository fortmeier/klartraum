#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/drawgraph/buffertransformation.hpp"
#include "klartraum/drawgraph/drawgraphgroup.hpp"
#include "klartraum/drawgraph/imageviewsrc.hpp"
#include "klartraum/drawgraph/rendergraphelement.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

enum class GaussianSplattingRenderingType {
    PointCloud,
    // GaussianSplatting, not implemented yet
};

class VulkanGaussianSplatting : virtual public RenderGraphElement, virtual public DrawGraphGroup {
    /**
     * @brief
     *
     * Gaussian Splatting consists of these steps:
     * 1. project the 3D Gaussian to 2D
     * 2. distribute/bin the 2D Gaussians to 4x4 subtiles
     * 3. sort the 2D Gaussians by depth and tile using radix sort
     * 4. splat the 2D Gaussians to each subtile of the image
     *
     * The current implementation is probably not optimal:
     * - projection and binning could be combined into single step
     * - binning of the 2D gaussians creates a new number of gaussians,
     *   which is limited to 2 * number of initinal 3D gaussians.
     *   (2 is a magic number currently)
     * - all further steps thus start enough workgroups to potentially
     *   process all newly created gaussians and discarding the ones
     *   that are not needed. a dynamic workgroup count
     *   would be more efficient, but it is unclear how to implement this
     */
public:
    VulkanGaussianSplatting(
        VulkanKernel& vulkanKernel,
        std::shared_ptr<ImageViewSrc> imageViewSrc,
        std::shared_ptr<CameraUboType> cameraUBO,
        std::string path);
    ~VulkanGaussianSplatting();

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) override;

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) override;

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

    virtual const char* getType() const override {
        return "GaussianSplatting";
    }

private:
    void loadSPZModel(std::string path);

    VulkanKernel* vulkanKernel = nullptr;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    std::vector<Gaussian3D> gaussians3DData;
    uint32_t number_of_gaussians;

    uint32_t numberOfPaths = 0;

    std::shared_ptr<BufferElementSinglePath<Gaussian3DBuffer>> gaussians3D;

    std::shared_ptr<GaussianProjection> project3Dto2D;
    std::shared_ptr<GaussianSort> sort2DGaussians;
    std::shared_ptr<GaussianBinning> bin;
    std::shared_ptr<GaussianComputeBounds> computeBounds;
    std::shared_ptr<GaussianSplatting> splat;
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP