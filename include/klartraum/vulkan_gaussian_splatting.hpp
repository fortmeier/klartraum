#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <vector>
#include <string>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/drawgraph/rendergraphelement.hpp"

#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

#include "klartraum/drawgraph/drawgraphgroup.hpp"
#include "klartraum/drawgraph/buffertransformation.hpp"
#include "klartraum/drawgraph/imageviewsrc.hpp"

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
 * 2. distribute the 2D Gaussians to 4x4 subtiles
 * 3. sort the 2D Gaussians by depth and tile using radix sort
 * 4. splat the 2D Gaussians to each subtile of the image
 * 
 * Here, this is solved by a drawgraph consisting of
 * 1. shader drawgraphelent (projection shader, 3D gaussians in, 2D gaussians out)
 * 2. sorting shader (2D gaussians in, sorted 2D gaussians out)
 */
public:
    VulkanGaussianSplatting(
        VulkanKernel& vulkanKernel,
        std::shared_ptr<ImageViewSrc> imageViewSrc,
        std::shared_ptr<CameraUboType> cameraUBO,
        std::string path);
    ~VulkanGaussianSplatting();

    virtual void checkInput(DrawGraphElementPtr input, int index = 0);

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths);

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId);

    virtual const char* getType() const
    {
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