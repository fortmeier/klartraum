// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
#define VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP

#include <array>
#include <memory>

#include <glm/glm.hpp>

#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/vulkan_buffer.hpp"

#include "klartraum/draw_component.hpp" // for CameraUboType, TODO: remove this dependency

namespace klartraum {

// Runtime-tunable knobs for the Gaussian splatting pipeline.
// Different hardware benefits from different settings; adjust before construction.
struct GsplatConfig {
    // Gaussian footprint multiplier used in bin overlap tests (default 2.5 = 2.5 sigma).
    // Smaller → fewer bins touched, faster binning, slight clipping at edges.
    // Larger  → more bins touched, slower binning, better edge quality.
    float spreadMultiplier = 2.5f;

    // Binned buffer capacity = N * maxMod.  Each Gaussian can appear in at most
    // maxMod bins on average before the buffer clips.  2 is safe for most scenes.
    uint32_t maxMod = 2u;

    // Hard cap on radix-sort workgroup count.
    // More WGs = better GPU utilisation for large scenes.
    // Fewer WGs = less overhead for small scenes.
    uint32_t numSortWGsCap = 320u;

    // Splatting tile dimensions (workgroup = tileX × tileY threads).
    // Larger tiles → fewer barrier() pairs per Gaussian, higher shared-memory use.
    // Must divide the per-bin tile count evenly.
    uint32_t splatTileX = 8u;
    uint32_t splatTileY = 8u;

    // Spherical-harmonics degree evaluated by the raster backend's per-splat
    // attribute precompute. 0 = DC only, 3 = full degree-3 (all 15 bands). The
    // model carries degree-3 SH, so 3 reproduces the reference colour; lower
    // degrees skip band loads + evaluation for speed at a colour-fidelity cost.
    int shDegree = 3;

    // Raster backend dist-side opacity cull: splats whose post-activation alpha
    // is below this are dropped before sort/draw. 1/255 is near-bit-preserving
    // (such splats contribute < 1/255 even at their centre). 0 disables it.
    float alphaCullThreshold = 1.0f / 255.0f;

    // Raster backend: use the VK_EXT_mesh_shader draw path instead of the vertex
    // path. Only takes effect when the device supports mesh shaders; otherwise
    // the raster backend falls back to the vertex path. Default off (portable).
    bool useMeshShader = false;
};

// this is a copy of the UnpackedGaussian struct from spz::UnpackedGaussian
struct Gaussian3D {
    std::array<float, 3> position; // x, y, z
    std::array<float, 4> rotation; // x, y, z, w
    std::array<float, 3> scale;    // std::log(scale)
    std::array<float, 3> color;    // rgb sh0 encoding
    float alpha;                   // inverse logistic
    std::array<float, 15> shR;
    std::array<float, 15> shG;
    std::array<float, 15> shB;
};

typedef VulkanBuffer<Gaussian3D> Gaussian3DBuffer;

// A buffer as a consumer connects to it: `element` itself (slot -1), or the
// buffer at input `slot` of `element`, e.g. one a compute element writes. In
// the second case the consumer depends on that element, so it runs first.
struct BufferRef {
    ComputeGraphElementPtr element;
    int slot = -1;

    // The buffer element itself.
    ComputeGraphElementPtr buffer() const { return slot < 0 ? element : element->getInputElement(slot); }

    // Makes the buffer input `index` of `consumer`.
    void connectTo(ComputeGraphElement& consumer, int index) const { consumer.setInput(element, index, slot); }
};

// SoA GPU storage for a 3D Gaussian model: the seven buffers both splatting
// backends read (position vec3, rotation vec4 as x, y, z, w, linear scale
// vec3, colour+alpha vec4, and the three SH streams of 15 floats per Gaussian,
// coefficient-major: sh[b * count + i]) plus the splat count. A plain handle
// bundle with no loading logic: the buffers may be static uploads (e.g.
// GaussianDataStandard) or written every frame by compute elements (e.g.
// GaussianTransform), so a backend can be wired to buffers from any source.
struct GaussianSoABuffers {
    uint32_t count = 0;
    BufferRef pos;
    BufferRef rot;
    BufferRef scale;
    BufferRef colAlpha;
    BufferRef shR;
    BufferRef shG;
    BufferRef shB;

    std::array<const BufferRef*, 7> all() const { return {&pos, &rot, &scale, &colAlpha, &shR, &shG, &shB}; }
};

// Makes `gaussians` inputs `first` .. `first + 6` of `element`.
inline void connectGaussians(ComputeGraphElement& element, const GaussianSoABuffers& gaussians, int first) {
    int index = first;
    for (const BufferRef* ref : gaussians.all()) {
        ref->connectTo(element, index++);
    }
}

// Per-path buffers for `count` Gaussians that `element` writes, made its
// inputs `first` .. `first + 6`; returns them as the element's output.
inline GaussianSoABuffers addGaussianOutputs(VulkanContext& vulkanContext, const ComputeGraphElementPtr& element,
                                             uint32_t count, int first) {
    GaussianSoABuffers output;
    output.count = count;
    int slot = first;
    auto add = [&](auto buffer, const char* name) {
        buffer->setName(name);
        element->setInput(buffer, slot);
        return BufferRef{element, slot++};
    };
    output.pos = add(std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vulkanContext, count), "Pos3D");
    output.rot = add(std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, count), "Rot3D");
    output.scale = add(std::make_shared<BufferElement<VulkanBuffer<glm::vec3>>>(vulkanContext, count), "Scale3D");
    output.colAlpha = add(std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, count), "ColAlpha3D");
    output.shR = add(std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, 15 * count), "ShR");
    output.shG = add(std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, 15 * count), "ShG");
    output.shB = add(std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, 15 * count), "ShB");
    return output;
}

struct ProjectionPushConstants {
    uint32_t numElements;
    uint32_t gridSize;
    float screenWidth;
    float screenHeight;
};

typedef GeneralComputation<ProjectionPushConstants> GaussianProjection;

struct BinningCountPushConstants {
    uint32_t numElements;
    uint32_t gridSize;
    float screenWidth;
    float screenHeight;
    float spreadMultiplier; // Gaussian footprint radius multiplier
};
typedef GeneralComputation<BinningCountPushConstants> GaussianBinningCount;

struct BinningScatterPushConstants {
    uint32_t numElements;
    uint32_t gridSize;
    float screenWidth;
    float screenHeight;
    uint32_t maxOutput;
    float spreadMultiplier; // Gaussian footprint radius multiplier
};
typedef GeneralComputation<BinningScatterPushConstants> GaussianBinningScatter;

struct SplatPushConstants {
    uint32_t numElements;
    uint32_t gridSize;
    uint32_t gridX;
    uint32_t gridY;
    float screenWidth;
    float screenHeight;
};

struct DistPushConstants {
    uint32_t numSplats;
    float frustumDilation;       // dilates the cull frustum so near-edge splat footprints survive
    float alphaThreshold = 0.0f; // drop splats with post-activation alpha below this (0 = off)
};
typedef GeneralComputation<DistPushConstants> GaussianDist;

// Per-splat 2D attribute precompute (raster backend, gsplat_raster_project.comp).
// All-scalar (4-byte) members so std430 packing matches the GLSL push block.
struct RasterProjectPushConstants {
    uint32_t numSplats;
    float screenWidth;
    float screenHeight;
    float splatScale;
    int32_t shDegree;
};
typedef GeneralComputation<RasterProjectPushConstants> GaussianRasterProject;

struct SortPushConstants {
    uint32_t pass;
    uint32_t numElements;
    uint32_t numBins;
    // 1 = read the active element count from the sort's count buffer (binding 6)
    // instead of numElements, so a fixed-dispatch sort processes only a
    // GPU-determined visible count. Default 0 keeps the static-count behaviour.
    uint32_t useCountBuffer = 0;
};
typedef GeneralComputation<SortPushConstants> RadixSort;

typedef GeneralComputation<SplatPushConstants> GaussianSplatting;

// Push-constant-less compute (e.g. the raster mesh-dispatch-args fill).
typedef GeneralComputation<> MeshArgsFill;

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
