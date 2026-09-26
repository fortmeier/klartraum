#ifndef KLARTRAUM_GAUSSIAN_DATA_STANDARD_HPP
#define KLARTRAUM_GAUSSIAN_DATA_STANDARD_HPP

#include <string>
#include <vector>

#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

// Owns the CPU-side 3D Gaussian model and its GPU-side SoA storage. It loads/holds
// the Gaussian3D array (from an SPZ file or a caller-supplied vector), converts it
// AoS -> SoA, and uploads the static single-path storage buffers the pipelines
// read. The result is exposed as a GaussianSoABuffers handle bundle via buffers();
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
    void loadSPZModel(const std::string& path, bool flipY);
    void uploadSoA(VulkanContext& vulkanContext);

    std::vector<Gaussian3D> gaussians3DData;
    GaussianSoABuffers buffers_;
};

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_DATA_STANDARD_HPP
