// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_CLOSEUP_INPAINTING_HPP
#define KLARTRAUM_SD15_CLOSEUP_INPAINTING_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/camera.hpp"

namespace klartraum {

/**
 * @brief Inpainting a character into a wide view through a square close-up, on the CPU.
 *
 * A person is small in a wide view, too small for SD1.5. A crop camera therefore zooms onto a window around the
 * character that is square in pixels (CloseUpWindow); the close-up is rendered, inpainted in latent space and pasted
 * back: first into the close-up scene through the feathered body mask (pasteGenerated()), then into the full view
 * inside the feathered character mask (compositeCloseUp()). Images are BGRA bytes.
 *
 * These functions are the reference for the GPU elements in gpu_inpainting.hpp.
 */

/// RGB of a BGRA image as a [3, height, width] tensor with values from `low` to `high`.
std::vector<float> bgraToTensor(const std::vector<uint8_t>& bgra, size_t pixels, float low, float high);

/// The red channel of a BGRA image.
std::vector<uint8_t> redChannel(const std::vector<uint8_t>& bgra);

/// A feathered mask (non-zero: set): its coverage of the 5 x 5 box around each pixel, from 0 to 1.
std::vector<float> featherMask(const std::vector<uint8_t>& mask, int width, int height);

/// Bilinear sample of a BGRA channel at continuous pixel coordinates (pixel centres at integers).
float sampleChannel(const std::vector<uint8_t>& bgra, int width, int height, glm::vec2 position, size_t channel);

/**
 * @brief The latent mask of the close-up: 1 where the body is (latentMask() of `bodyMask`), else half the latent
 * mask of `marginMask`, the body with a margin, so the margin ring stays close to the scene.
 *
 * Both masks are single-channel (non-zero: set), `width` x `height` pixels.
 */
std::vector<float> closeUpLatentMask(const std::vector<uint8_t>& marginMask, const std::vector<uint8_t>& bodyMask,
                                     uint32_t width, uint32_t height, uint32_t factor = 8, uint32_t dilation = 1);

/// Like cropProjection(), with separate half extents for x and y, so a window square in pixels fits a wide view.
glm::mat4 cropProjection(const glm::mat4& proj, glm::vec2 centre, glm::vec2 halfExtent);

/**
 * @brief The window of the close-up in the NDC of the full view, following the character.
 *
 * The window is square in pixels, with room around the body. Its centre follows the character exactly; its size,
 * which swinging limbs change, is smoothed from frame to frame so the close-up does not pump, but never falls below
 * 90 % of the body's.
 */
struct CloseUpWindow {
    glm::vec2 centre{0.0f};
    glm::vec2 halfExtent{1.0f};
    float halfY = 1.0f;
    bool initialized = false;

    /// Moves the window to `joints` (in the character's space) seen by `full`; `aspect` is the full view's.
    void update(const std::vector<glm::vec4>& joints, const CameraMVP& full, const glm::mat4& characterToScene,
                float aspect);

    /// `full` with its projection cropped to the window.
    CameraMVP sceneCamera(const CameraMVP& full) const;
    /// The scene camera with the character's placement as model.
    CameraMVP skeletonCamera(const CameraMVP& full, const glm::mat4& characterToScene) const;
};

/// A generated close-up pasted into the close-up scene.
struct PastedCloseUp {
    std::vector<uint8_t> image; ///< BGRA: the scene with the generated person
    std::vector<uint8_t> alpha; ///< BGRA with the feathered body mask (0 .. 255) in byte 0
};

/**
 * @brief Pastes `generated` ([3, size, size], RGB in [-1, 1]) into `scene` through the feathered `bodyMask`
 * (single-channel).
 */
PastedCloseUp pasteGenerated(const std::vector<uint8_t>& scene, const std::vector<float>& generated,
                             const std::vector<uint8_t>& bodyMask, int size);

/**
 * @brief The full view `scene` (width x height) with the close-up pasted inside the feathered character mask
 * (BGRA, red channel set), the close-up covering `window`.
 */
std::vector<uint8_t> compositeCloseUp(const std::vector<uint8_t>& scene, const std::vector<uint8_t>& characterMask,
                                      const PastedCloseUp& closeUp, int size, int width, int height,
                                      const CloseUpWindow& window);

} // namespace klartraum

#endif // KLARTRAUM_SD15_CLOSEUP_INPAINTING_HPP
