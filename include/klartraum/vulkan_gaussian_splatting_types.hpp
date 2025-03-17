#ifndef VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
#define VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP

#include <glm/glm.hpp>

namespace klartraum {

struct Gaussian2D {
    glm::vec2 position;
    glm::mat2 covariance;
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_TYPES_HPP
