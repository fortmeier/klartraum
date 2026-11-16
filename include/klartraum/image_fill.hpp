// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_IMAGE_FILL_HPP
#define KLARTRAUM_IMAGE_FILL_HPP

#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/**
 * @brief Fills the images of the ImageViewSrc at input 0 with a colour that can change per submission of a
 *        recorded graph (setColour()), and stands for them: connect renderers that draw over the image's contents
 *        (the Gaussian splatting backends, a RenderPass that loads them) to it instead of the source.
 *
 * The colour is written into a host-visible buffer, copied into a one-texel image and blitted over the whole image,
 * which is left in VK_IMAGE_LAYOUT_GENERAL. The images must be of the swapchain format with 8 bits per channel.
 */
class ImageFill : public ImageViewForward {
public:
    explicit ImageFill(glm::vec3 colour = glm::vec3(0.0f))
        : colour(colour) {}
    ~ImageFill();

    const char* getType() const override { return "ImageFill"; }
    VkImageLayout getOutputLayout() const override { return VK_IMAGE_LAYOUT_GENERAL; }

    /** @brief The colour (rgb in [0, 1]) of the next submissions. */
    void setColour(glm::vec3 colour);

    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue = VK_NULL_HANDLE) override;
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

private:
    // Writes the colour, in the images' byte order, into every path's buffer.
    void writeColour();

    glm::vec3 colour;
    VulkanContext* vulkanContext = nullptr;
    bool bgra = true; // the byte order of the images' format
    // Per path: the host-visible buffer holding the colour's four bytes and the one-texel image it is copied into.
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceMemory> bufferMemories;
    std::vector<void*> mapped;
    std::vector<VkImage> texels;
    std::vector<VkDeviceMemory> texelMemories;
};

} // namespace klartraum

#endif // KLARTRAUM_IMAGE_FILL_HPP
