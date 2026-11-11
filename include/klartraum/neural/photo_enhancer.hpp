// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_NEURAL_PHOTO_ENHANCER_HPP
#define KLARTRAUM_NEURAL_PHOTO_ENHANCER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {
class OnnxNetwork;
}

namespace klartraum::neural {

/// The G-buffers of one frame as the renderers write them, read back as BGRA bytes at the render size.
struct GBufferFrame {
    uint32_t width = 0;  ///< render width in pixels
    uint32_t height = 0; ///< render height in pixels
    std::vector<uint8_t> colour;
    std::vector<uint8_t> classes; ///< class colours (tree::treeClassColor())
    std::vector<uint8_t> depth16; ///< 16-bit depth from 0 to depthFar
    std::vector<uint8_t> u;       ///< coordinate u, blue the object id's high byte
    std::vector<uint8_t> v;       ///< coordinate v, blue the object id's low byte
    std::vector<uint8_t> normals; ///< (n + 1) / 2 of the world-space normal, optional
    float depthFar = 1.0f;        ///< the far end of the depth window of depth16
};

/// The photo enhancer's sample: its size and how its inverse depth is stretched.
struct PhotoEnhancerSample {
    uint32_t width = 832;
    uint32_t height = 464;
    float depthHigh = 0.98f; ///< the quantile of the covered pixels' inverse depth that becomes 1 (1: the nearest)
};

/// The generator's G-buffer channels and the texture's inputs of one frame, at the sample size.
struct PhotoEnhancerInputs {
    std::vector<float> channels;    ///< [11][height][width]: RGB, stretched inverse depth, 7 one-hot classes
    std::vector<float> coordinates; ///< [3][height][width]: u, v, object id in [0, 1]
    std::vector<float> classes;     ///< [height][width]: class indices as floats
    std::vector<float> normals;     ///< [3][height][width] in [-1, 1] (0 on the background), if the frame has them
};

/**
 * @brief The photo enhancer's inputs of a frame, as scripts/neural/prepare_photo_pairs.py and
 * add_surface_coordinates.py make them from the frames of a flight: the frame scaled to the sample width (PIL's
 * resampling) and cropped to its height around the middle; RGB bicubically (in 8-bit steps), the inverse depth of the
 * covered pixels stretched between its 2nd percentile and `depthHigh` and resampled bilinearly, classes (the nearest
 * class colour), surface coordinates and normals sampled nearest, so that they never blend between surfaces.
 */
PhotoEnhancerInputs photoEnhancerInputs(const GBufferFrame& frame, const PhotoEnhancerSample& sample);

/**
 * @brief A photo enhancer exported by scripts/neural/train_photo_enhancer.py, run on the GPU: the learnable surface
 * texture (hashTexture(), from `<stem>_texture.txt` and `_texture_f32.bin`) writes its features after the G-buffer
 * channels, and the generator (`<stem>.onnx`) turns them into the image. A generator conditioned on the season
 * (`_season_f32.bin` next to it) or on an appearance code (`_appearance_input_f32.bin`, the training images' codes in
 * `_appearance_codes_f32.bin`) gets those inputs as well.
 */
class PhotoEnhancer {
public:
    PhotoEnhancer(VulkanContext& vulkanContext, const std::string& stem, uint32_t width, uint32_t height);
    ~PhotoEnhancer();

    bool hasSeason() const { return season != nullptr; }
    /// The season S in [0, 4) (0 winter, 1 spring, 2 summer, 3 autumn) for a season-conditioned generator.
    void setSeason(float value);

    /// The size of the appearance code (0: the generator has none).
    uint32_t appearanceSize() const { return codeSize; }
    /// The codes of the training images, [images][appearanceSize()], and their sample ids.
    const std::vector<float>& trainingCodes() const { return codes; }
    const std::vector<std::string>& trainingIds() const { return codeIds; }
    /// The appearance code the generator gets (default: the training codes' mean).
    void setAppearance(const std::vector<float>& code);

    /// Runs texture and generator on `inputs`; the image as [3][height][width] in [0, 1].
    std::vector<float> run(const PhotoEnhancerInputs& inputs);

private:
    VulkanContext& vulkanContext;
    uint32_t width, height;
    /// The generator's input [1][11 + texture features][height][width]; the texture writes after the G-buffers.
    std::shared_ptr<TensorElement<float>> input;
    std::shared_ptr<TensorElement<float>> coordinates;
    std::shared_ptr<TensorElement<float>> classes;
    std::shared_ptr<TensorElement<float>> table;
    std::shared_ptr<TensorElement<float>> season;     ///< the season's code (sin, cos), or none
    std::shared_ptr<TensorElement<float>> appearance; ///< the appearance code, or none
    std::shared_ptr<OnnxNetwork> network;
    std::shared_ptr<TensorElement<float>> output;
    /// Texture and network as two graphs, submitted in this order.
    std::unique_ptr<ComputeGraph> textureGraph;
    std::unique_ptr<ComputeGraph> networkGraph;
    uint32_t codeSize = 0;
    std::vector<float> codes;
    std::vector<std::string> codeIds;
};

} // namespace klartraum::neural

#endif // KLARTRAUM_NEURAL_PHOTO_ENHANCER_HPP
