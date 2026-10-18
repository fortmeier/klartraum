// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GAUSSIAN_MORPH_HPP
#define KLARTRAUM_GAUSSIAN_MORPH_HPP

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// Turns Gaussians laid out on the z = 0 plane (x right, y up, seen from +z),
/// such as text from sampleTextGaussians(), so they face the camera of the view
/// matrix `view`: x along the camera's right, y along its up, the plane
/// perpendicular to its viewing direction. The plane's origin moves to `centre`.
void placeFacingCamera(std::vector<Gaussian3D>& gaussians, const glm::mat4& view, const glm::vec3& centre);

/// Pairs two equally large sets of Gaussians by where they appear for the
/// camera of the view matrix `view`, so that neighbours stay neighbours: both
/// sets are sorted from top to bottom into horizontal bands of equal size, and
/// the members of a band from left to right. Returns, for each Gaussian of
/// `scene`, the index of its partner in `text`; every index occurs once.
std::vector<uint32_t> matchByScreenPosition(const std::vector<Gaussian3D>& text, const std::vector<Gaussian3D>& scene,
                                            const glm::mat4& view);

/// How the Gaussians travel from the start to the end set (GaussianMorph).
enum class MorphVariant : int {
    Straight = 0, ///< on straight lines
    Swirl = 1,    ///< on spirals around MorphParams::axis
    Scatter = 2,  ///< out into a cloud around the scene, then into place
    Wave = 3,     ///< straight, but each starts after its delay, e.g. letter by letter
    // The following first burst out and come to rest at explodeShare of the
    // morph, then fly into place.
    Explode = 4,      ///< burst away from MorphParams::centre, then straight
    ExplodeSwirl = 5, ///< burst away from the centre, then on spirals around the axis
    Shatter = 6,      ///< burst towards MorphParams::direction (e.g. the camera), then straight
    SwirlScatter = 7, ///< the scatter path, turned around the axis while flying
    SwirlWave = 8,    ///< spirals, each starting after its delay as in the wave
};

/// The values a GaussianMorph reads every run.
struct MorphParams {
    static constexpr uint32_t kSize = 17;

    float progress = 0.0f; ///< 0: start set, 1: end set
    MorphVariant variant = MorphVariant::Straight;
    float swirlTurns = 1.0f;          ///< extra turns around the axis, rounded to whole turns so swirls end in place
    float scatterRadius = 0.3f;       ///< how far Gaussians fly out, world units (Scatter)
    float waveSpread = 0.6f;          ///< part of the morph over which the starts are spread (Wave), in [0, 1)
    glm::vec3 centre{0.0f};           ///< a point on the swirl axis, and where the explosions burst from
    glm::vec3 axis{0.0f, 1.0f, 0.0f}; ///< swirl axis (unit length)
    uint32_t seed = 1;                ///< random directions of the scatter and the explosions
    float explodeRadius = 0.3f;       ///< how far Gaussians burst out, world units (Explode, ExplodeSwirl, Shatter)
    float explodeShare = 0.35f;       ///< part of the morph the burst takes, in (0, 1)
    glm::vec3 direction{0.0f, 0.0f, 1.0f}; ///< where the shatter flies (unit length), e.g. towards the camera

    /// The layout the shader reads: progress, variant, swirlTurns,
    /// scatterRadius, waveSpread, centre (3), axis (3), seed, explodeRadius,
    /// explodeShare, direction (3).
    std::vector<float> values() const;
};

/// Text Gaussians ready to morph into a scene.
struct TextMorphStart {
    std::vector<Gaussian3D> gaussians; ///< text Gaussian i starts scene Gaussian i
    std::vector<float> delays;         ///< per Gaussian: the text position from left (0) to right (1)
};

/// Places `text` (on the z = 0 plane, as sampleTextGaussians() makes it, as
/// many as `scene` has) facing the camera of `view` at `centre`, matches it to
/// `scene` by screen position and orders it like `scene`.
TextMorphStart prepareTextMorph(std::vector<Gaussian3D> text, const std::vector<Gaussian3D>& scene,
                                const glm::mat4& view, const glm::vec3& centre);

/// Timing of a shot that holds a title, morphs it into a scene and turns the
/// camera around the scene: the camera starts with the morph, speeds up evenly
/// until the morph ends and then turns at a constant rate, `turns` times in
/// all by the end of the shot.
struct MorphTimeline {
    float fps = 30.0f;
    float titleSeconds = 2.0f;
    float morphSeconds = 1.5f;
    float spinSeconds = 5.0f;
    float turns = 1.0f;

    uint32_t frameCount() const;
    /// Number of frames that show the still title.
    uint32_t titleFrames() const;
    /// Morph progress (0 to 1) at frame `frame`.
    float progressAt(uint32_t frame) const;
    /// Camera angle around the scene, in radians since the morph started.
    double angleAt(uint32_t frame) const;
    /// The angle the camera turns by the end of the shot, at time
    /// titleSeconds + morphSeconds + spinSeconds.
    double totalAngle() const;
};

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_MORPH_HPP
