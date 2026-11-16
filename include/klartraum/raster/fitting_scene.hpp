// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_FITTING_SCENE_HPP
#define KLARTRAUM_RASTER_FITTING_SCENE_HPP

#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

namespace klartraum::raster {

/**
 * @brief Masked photographs of an object with their cameras, prepared for MeshTraining by
 * scripts/mesh_fitting/prepare_data.py.
 *
 * On disk, a directory with targets.bin (float32, views x height x width x 4: premultiplied rgb, mask), cameras.bin
 * (float32, views x 16: Vulkan clip-from-world matrices, column-major) and scene.txt ("key values..." lines: width,
 * height, views, train, validation, centre, radius).
 */
struct FittingScene {
    uint32_t width = 0, height = 0;   ///< image size in pixels
    std::vector<float> targets;       ///< four floats per pixel and view, views one after another
    std::vector<glm::mat4> cameras;   ///< clip from world, one per view
    std::vector<uint32_t> train;      ///< indices of the training views
    std::vector<uint32_t> validation; ///< indices of the held-out views
    glm::vec3 centre{0.0f};           ///< centre of a sphere enclosing the object
    float radius = 1.0f;              ///< radius of that sphere

    /** @brief The targets of the views @p indices, one after another (for MeshTraining::setTargets()). */
    std::vector<float> targetsOf(const std::vector<uint32_t>& indices) const;

    /** @brief The cameras of the views @p indices. */
    std::vector<glm::mat4> camerasOf(const std::vector<uint32_t>& indices) const;
};

/** @brief Reads a scene directory; throws std::runtime_error when a file is missing or sizes disagree. */
FittingScene loadFittingScene(const std::filesystem::path& directory);

/** @brief Writes a scene directory in the format loadFittingScene() reads. */
void writeFittingScene(const FittingScene& scene, const std::filesystem::path& directory);

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_FITTING_SCENE_HPP
