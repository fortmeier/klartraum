// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_STYLE_TRANSFER_HPP
#define KLARTRAUM_STYLE_TRANSFER_HPP

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace klartraum {

/**
 * @brief Restyles a rendering with an image-to-image network (scripts/neural/train_watercolor.py) and blends the
 *        result over it.
 *
 * `style_transfer_input.comp` makes the network's input at its size, the frame's or half of it: the colour (and the
 * depth), averaged over 2 x 2 pixels at half the size, and, for a network with 11 input channels, the classes of the
 * G-buffer (the class whose colour, tree::treeClassColor(), is nearest to pixel (x, y) or, at half the size,
 * (2x + 1, 2y + 1)) one-hot, with `classes` and `depth` the
 * BonsaiRenderer's G-buffers of the same view; the ONNX network runs; `style_transfer_output.comp` resizes its output
 * bilinearly to the frame and blends it over `color` into `output` by setBlend(). The images are the frame's size
 * and in TRANSFER_SRC_OPTIMAL before (`producer` last writes them) and after; compile a graph from getOutput().
 */
class StyleTransfer {
public:
    StyleTransfer(VulkanContext& vulkanContext, const std::string& modelPath, std::shared_ptr<OffscreenTarget> color,
                  std::shared_ptr<OffscreenTarget> output, ComputeGraphElementPtr producer,
                  std::shared_ptr<OffscreenTarget> classes = nullptr, std::shared_ptr<OffscreenTarget> depth = nullptr);

    ComputeGraphElementPtr getOutput() const { return done; }
    /// The network, e.g. to read its output tensor ("output").
    std::shared_ptr<OnnxNetwork> getNetwork() const { return network; }
    /// Whether the network reads the classes and depth G-buffers (11 input channels).
    bool usesGBuffers() const { return gbuffers; }
    /// How much of the restyled image the next frames show: 0 the rendering, 1 the network's image.
    void setBlend(float blend) { parameters->set(0, glm::vec4(blend, 0.0f, 0.0f, 0.0f)); }

    struct Push {
        uint32_t frameWidth, frameHeight; ///< the frame's size
        uint32_t width, height;           ///< the network's size
        uint32_t channels;                ///< the network's input channels
        uint32_t scale;                   ///< the frame's size over the network's, 1 or 2
        uint32_t pad[2];
    };

private:
    // The network's input tensor, written by `input`.
    std::shared_ptr<TensorElement<float>> tensor;
    std::shared_ptr<GeneralComputation<Push>> input;
    std::shared_ptr<OnnxNetwork> network;
    // Blend factor (x) of the output pass.
    std::shared_ptr<HostValues<glm::vec4>> parameters;
    std::shared_ptr<GeneralComputation<Push>> toImage;
    ComputeGraphElementPtr done;
    // Whether the network reads the G-buffers.
    bool gbuffers = false;
};

} // namespace klartraum

#endif // KLARTRAUM_STYLE_TRANSFER_HPP
