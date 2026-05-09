#ifndef VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
#define VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP

#include <array>

#include <glm/glm.hpp>

#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"

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
};

// this is a copy of the UnpackedGaussian struct from spz::UnpackedGaussian
struct Gaussian3D {
    std::array<float, 3> position;  // x, y, z
    std::array<float, 4> rotation;  // x, y, z, w
    std::array<float, 3> scale;     // std::log(scale)
    std::array<float, 3> color;     // rgb sh0 encoding
    float alpha;                    // inverse logistic
    std::array<float, 15> shR;
    std::array<float, 15> shG;
    std::array<float, 15> shB;
};

typedef VulkanBuffer<Gaussian3D> Gaussian3DBuffer;

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
    float    screenWidth;
    float    screenHeight;
    float    spreadMultiplier;  // Gaussian footprint radius multiplier
};
typedef GeneralComputation<BinningCountPushConstants> GaussianBinningCount;

struct BinningScatterPushConstants {
    uint32_t numElements;
    uint32_t gridSize;
    float    screenWidth;
    float    screenHeight;
    uint32_t maxOutput;
    float    spreadMultiplier;  // Gaussian footprint radius multiplier
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

struct SortPushConstants {
  uint32_t pass;
  uint32_t numElements;
  uint32_t numBins;
};
typedef GeneralComputation<SortPushConstants> RadixSort;

typedef GeneralComputation<SplatPushConstants> GaussianSplatting;

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
