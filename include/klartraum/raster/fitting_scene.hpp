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
 * height, views, train, validation, centre, radius, and optionally points and spacing with points.bin: float32,
 * points x 3).
 */
struct FittingScene {
    uint32_t width = 0, height = 0;   ///< image size in pixels
    std::vector<float> targets;       ///< four floats per pixel and view, views one after another
    std::vector<glm::mat4> cameras;   ///< clip from world, one per view
    std::vector<uint32_t> train;      ///< indices of the training views
    std::vector<uint32_t> validation; ///< indices of the held-out views
    glm::vec3 centre{0.0f};           ///< centre of a sphere enclosing the object
    float radius = 1.0f;              ///< radius of that sphere
    std::vector<glm::vec3> points;    ///< surface points of the carved visual hull (points.bin), if any
    float spacing = 0.0f;             ///< distance between neighbouring points

    /** @brief The targets of the views @p indices, one after another (for MeshTraining::setTargets()). */
    std::vector<float> targetsOf(const std::vector<uint32_t>& indices) const;

    /** @brief The cameras of the views @p indices. */
    std::vector<glm::mat4> camerasOf(const std::vector<uint32_t>& indices) const;

    /**
     * @brief Maps the unit sphere to the enclosing sphere with the scene's up on +y.
     *
     * The object rests on the ground, so the part of the enclosing sphere that no camera sees over the horizon faces
     * down; up is opposite its mean direction (or, if the cameras see all around, the mean direction to them). Fits
     * work in the unit sphere (cameras times this matrix), so step sizes and regularizer weights apply at any scene
     * scale, and the object stands upright.
     */
    glm::mat4 sceneFromUnitSphere() const;
};

/** @brief The position of the camera of a clip-from-world matrix (where clip x, y and w are zero). */
glm::vec3 cameraPosition(const glm::mat4& clipFromWorld);

/**
 * @brief A view for progress frames of a fit in the unit sphere of a scene (FittingScene::sceneFromUnitSphere(),
 * +y up): a photo turned upright in quarter turns and a camera looking horizontally at the object's side the photo
 * shows, from a little above.
 */
struct DisplayView {
    glm::mat4 camera;         ///< clip from unit-sphere coordinates, square image
    uint32_t side = 0;        ///< width and height of the square image
    std::vector<float> photo; ///< the photo, upright and centred on a transparent square, four floats per pixel
};

/** @brief The display view of the scene's photo @p view. */
DisplayView displayView(const FittingScene& scene, uint32_t view);

/** @brief Reads a scene directory; throws std::runtime_error when a file is missing or sizes disagree. */
FittingScene loadFittingScene(const std::filesystem::path& directory);

/** @brief Writes a scene directory in the format loadFittingScene() reads. */
void writeFittingScene(const FittingScene& scene, const std::filesystem::path& directory);

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_FITTING_SCENE_HPP
