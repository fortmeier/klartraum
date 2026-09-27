#ifndef KLARTRAUM_GAUSSIAN_DATA_STANDARD_HPP
#define KLARTRAUM_GAUSSIAN_DATA_STANDARD_HPP

#include <string>
#include <vector>

#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

// Loads and unpacks an SPZ file into Gaussians: linear scales, opacities in
// [0, 1], rotations as x, y, z, w quaternions. With `flipY` the scene is
// mirrored across the Y axis (e.g. for Nerfstudio exports). Throws if the file
// cannot be read.
std::vector<Gaussian3D> loadGaussiansSpz(const std::string& path, bool flipY = false);

// Owns a 3D Gaussian model's GPU-side SoA storage. It takes the Gaussian3D array
// (from an SPZ file or a caller-supplied vector), converts it AoS -> SoA, and
// uploads the static single-path storage buffers the pipelines read; the
// CPU-side array is not kept. The result is exposed as a GaussianSoABuffers handle bundle via buffers();
// consumers (e.g. the splatting backends) take that bundle, not this class, so the
// model's loading/ownership is decoupled from how the buffers get wired.
class GaussianDataStandard {
public:
    // Load + unpack an SPZ file, then upload the SoA buffers.
    GaussianDataStandard(VulkanContext& vulkanContext, const std::string& path, bool flipY = false);
    // Take an already-built AoS vector (moved in), then upload the SoA buffers.
    GaussianDataStandard(VulkanContext& vulkanContext, std::vector<Gaussian3D> gaussians);

    uint32_t count() const { return buffers_.count; }

    // The uploaded SoA buffers, ready to be handed to a backend constructor.
    const GaussianSoABuffers& buffers() const { return buffers_; }

private:
    void uploadSoA(VulkanContext& vulkanContext, const std::vector<Gaussian3D>& gaussians);

    GaussianSoABuffers buffers_;
};

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_DATA_STANDARD_HPP
