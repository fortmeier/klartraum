// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_FIELD_STYLE_HPP
#define KLARTRAUM_FIELD_STYLE_HPP

#include <array>
#include <memory>
#include <string>
#include <utility>

#include <glm/glm.hpp>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace klartraum {

/**
 * @brief The distances from @p eye that the box [@p low, @p high] spans (at least 0.05): the window FieldStyle's depth
 * G-buffer is drawn with.
 */
std::pair<float, float> fieldDepthWindow(glm::vec3 eye, glm::vec3 low, glm::vec3 high);

/**
 * @brief Restyles a rendering per pixel with a style anchored in the world (scripts/neural/train_watercolor_field.py)
 *        and blends the result over it.
 *
 * Every pixel is styled from what the renderer knows about it: its colour, class and distance from the camera (the
 * BonsaiRenderer's colour, classes, coverage and depth G-buffers of the same view: the coverage a depth rendering
 * over black whose window lies in front of everything, the depth one over black between the near and far of
 * setFrame()), the features of two grids in world space at the surface point it shows, the paper (a tile repeated
 * across the frame, two pixels per texel) at the pixel where it shows none, and the frame's conditions.
 * `style_field_input.comp` packs these into the network's 41 input planes, the ONNX network (the edge convolutions and
 * the per-pixel MLP) runs at the frame's size, and `style_transfer_output.comp` blends its output over `color` into
 * `output` by setBlend(). The grids and the paper come from the field file. The images are the frame's size and in
 * TRANSFER_SRC_OPTIMAL before (`producer`, if any, last writes them in the same graph; without, the images are filled
 * before the graph is submitted, e.g. copied from another graph's renderings) and after; compile a graph from
 * getOutput().
 */
class FieldStyle {
public:
    /** @brief Loads the network from @p modelPath (.onnx) and the grids and paper from @p fieldPath (.bin). */
    FieldStyle(VulkanContext& vulkanContext, const std::string& modelPath, const std::string& fieldPath,
               std::shared_ptr<OffscreenTarget> color, std::shared_ptr<OffscreenTarget> classes,
               std::shared_ptr<OffscreenTarget> coverage, std::shared_ptr<OffscreenTarget> depth,
               std::shared_ptr<OffscreenTarget> output, ComputeGraphElementPtr producer);

    ComputeGraphElementPtr getOutput() const { return done; }
    /// The network's input planes, e.g. to check them.
    std::shared_ptr<TensorElement<float>> getInput() const { return tensor; }
    /// The network, e.g. to read its output tensor ("output").
    std::shared_ptr<OnnxNetwork> getNetwork() const { return network; }
    /// How much of the restyled image the next frames show: 0 the rendering, 1 the style's image.
    void setBlend(float blend) { blendValues->set(0, glm::vec4(blend, 0.0f, 0.0f, 0.0f)); }

    /**
     * @brief Sets the next frames' camera (clip from world, as the renderer's), its eye, the distances the depth
     * G-buffer maps from 0 to 1, and the conditions the style was trained with (train_watercolor_field.py:
     * conditions()).
     */
    void setFrame(const glm::mat4& camera, glm::vec3 eye, float near, float far,
                  const std::array<float, 7>& conditions);

    /** @brief fieldDepthWindow() of the style's box: the depth window to draw the G-buffer with. */
    std::pair<float, float> depthWindow(glm::vec3 eye) const { return fieldDepthWindow(eye, low, high); }

    struct Push {
        uint32_t width, height;
        uint32_t coarseX, coarseY, coarseZ;
        uint32_t fineX, fineY, fineZ;
        uint32_t paperWidth, paperHeight;
        uint32_t fineOffset, paperOffset;
        float lowX, lowY, lowZ;
        float highX, highY, highZ;
    };

private:
    // The box the grids cover.
    glm::vec3 low{0.0f}, high{0.0f};
    // The grids and the paper, as the field file holds them after its header.
    std::shared_ptr<TensorElement<float>> fieldData;
    // The network's input planes, written by `input`.
    std::shared_ptr<TensorElement<float>> tensor;
    std::shared_ptr<GeneralComputation<Push>> input;
    std::shared_ptr<OnnxNetwork> network;
    // The frame (see style_field_input.comp): inverse camera, eye, depth window, conditions.
    std::shared_ptr<HostValues<glm::vec4>> frameValues;
    // Blend factor (x) of the output pass.
    std::shared_ptr<HostValues<glm::vec4>> blendValues;
    ComputeGraphElementPtr done;
};

} // namespace klartraum

#endif // KLARTRAUM_FIELD_STYLE_HPP
