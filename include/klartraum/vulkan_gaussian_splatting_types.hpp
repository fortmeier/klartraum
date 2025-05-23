#ifndef VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
#define VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP

#include <array>

#include <glm/glm.hpp>

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

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
