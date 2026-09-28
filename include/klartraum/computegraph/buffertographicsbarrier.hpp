#ifndef KLARTRAUM_COMPUTEGRAPH_BUFFERTOGRAPHICSBARRIER_HPP
#define KLARTRAUM_COMPUTEGRAPH_BUFFERTOGRAPHICSBARRIER_HPP

#include <vector>
#include <vulkan/vulkan.h>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/bufferelement.hpp"

namespace klartraum {

// Klartraum's graph only knows how to order compute work after compute work
// (GeneralComputation emits compute->compute VkMemoryBarriers between its own
// dispatches). An indirect draw that consumes compute-written buffers (sorted
// indices, indirect draw args) needs an explicit buffer barrier whose
// destination is the graphics pipeline's indirect-draw and vertex-shader
// stages — this element supplies exactly that, as a graph node placed between
// the producing compute (sub)graph and the consuming RenderPass so Kahn
// ordering and the per-edge semaphores schedule it in between.
class BufferToGraphicsBarrier : public ComputeGraphElement {
public:
    BufferToGraphicsBarrier() {}

    // Buffers written by compute that the render pass will read via indirect
    // draw (VkDrawIndirectCommand) and/or vertex-shader SSBO reads (sorted
    // indices, per-splat attributes).
    void addBuffer(std::shared_ptr<BufferElementInterface> buffer) {
        buffers.push_back(buffer);
    }

    // Include the mesh-shader stage in the destination scope, for backends whose
    // render pass consumes these buffers from a mesh shader (VK_EXT_mesh_shader)
    // rather than the vertex shader. Only call when the extension is enabled.
    void setIncludeMeshShaderStage(bool include) {
        includeMeshShaderStage = include;
    }

    // This node is a pure ordering/sync edge — it accepts any producer as its
    // input purely so the graph schedules that producer before this barrier
    // (and thus before the RenderPass that depends on this barrier).
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override {
    }

    virtual const char* getType() const override {
        return "BufferToGraphicsBarrier";
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        ComputeGraphElement::_record(commandBuffer, pathId);

        std::vector<VkBufferMemoryBarrier> barriers;
        barriers.reserve(buffers.size());
        for (auto& buffer : buffers) {
            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = buffer->getVkBuffer(pathId);
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;
            barriers.push_back(barrier);
        }

        if (barriers.empty()) {
            return;
        }

        VkPipelineStageFlags dstStage =
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
        if (includeMeshShaderStage) {
            dstStage |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT;
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            dstStage,
            0,
            0, nullptr,
            (uint32_t)barriers.size(), barriers.data(),
            0, nullptr
        );
    }

private:
    std::vector<std::shared_ptr<BufferElementInterface>> buffers;
    bool includeMeshShaderStage = false;
};

typedef std::shared_ptr<BufferToGraphicsBarrier> BufferToGraphicsBarrierPtr;

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_BUFFERTOGRAPHICSBARRIER_HPP
