#ifndef VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
#define VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP

#include <array>

#include <glm/glm.hpp>

#include "klartraum/drawgraph/buffertransformation.hpp"
#include "klartraum/drawgraph/generalcomputation.hpp"

#include "klartraum/draw_component.hpp" // for CameraUboType, TODO: remove this dependency

namespace klartraum {

struct Gaussian2D {
    glm::vec2 position;
    float z;
    uint32_t binMask;
    glm::mat2 covariance;
    glm::vec3 color;
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
typedef VulkanBuffer<Gaussian2D> Gaussian2DBuffer;

struct ProjectionPushConstants {
  uint32_t numElements;
  uint32_t gridSize;
  float screenWidth;
  float screenHeight;
};

typedef BufferTransformation<Gaussian3DBuffer, Gaussian2DBuffer, CameraUboType, ProjectionPushConstants> GaussianProjection;

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
typedef BufferTransformation<Gaussian2DBuffer, Gaussian2DBuffer, void, SortPushConstants> GaussianSort;

typedef GeneralComputation<ProjectionPushConstants> GaussianBinning;
typedef GeneralComputation<ProjectionPushConstants> GaussianComputeBounds;
typedef GeneralComputation<SplatPushConstants> GaussianSplatting;

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
