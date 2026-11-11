// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_NEURAL_PHOTO_ENHANCER_TRAINING_HPP
#define KLARTRAUM_NEURAL_PHOTO_ENHANCER_TRAINING_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/layers/training.hpp"
#include "klartraum/neural/hash_texture.hpp"
#include "klartraum/neural/vgg_loss.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::neural {

/// The photo enhancer that PhotoEnhancerTraining trains, and how.
struct PhotoEnhancerTrainingConfig {
    uint32_t batch = 16; ///< crops per step
    uint32_t size = 128; ///< crop width and height, a multiple of 2^(levels - 1)
    /// The U-Net's channels per level (3 levels: the local generator of train_photo_enhancer.py --levels 3).
    std::vector<uint32_t> widths{32, 64, 128};
    /// The surface texture's grids; width, height, batch and outputOffset are set by the training.
    HashTextureShape texture;
    uint32_t appearances = 0;    ///< training images with an appearance code (0: none, no FiLM)
    uint32_t appearanceSize = 8; ///< numbers per code
    uint32_t filmHidden = 64;    ///< width of the FiLM MLP
    float appearanceDecay = 1e-2f;
    float l1 = 0.5f;                                 ///< weight of the L1 loss
    float colour = 5.0f;                             ///< weight of the L1 loss of 8 x 8 block means
    float perceptual = 0.0f;                         ///< weight of the VGG perceptual loss (VggLoss; 0: none)
    std::string vggDirectory = "build/models/vgg16"; ///< scripts/neural/export_vgg16.py's weights
    /// Weight of the adversarial loss of a 2-scale conditional PatchGAN (0: no discriminator); its step comes first
    /// in every training step, on DiffAugment-ed pairs (no R1 penalty).
    float adversarial = 0.0f;
    float featureMatching = 2.0f; ///< weight of the discriminator's feature matching
    training::AdamParameters discriminatorAdam{2e-4f, 0.5f, 0.999f};
    uint32_t seed = 1; ///< of the augmentations' random numbers
    float clip = 1.0f; ///< global norm the generator's gradients are clipped to
    training::AdamParameters networkAdam{1e-4f, 0.5f, 0.999f};
    training::AdamParameters textureAdam{1e-2f, 0.9f, 0.99f};
    training::AdamParameters codesAdam{5e-3f, 0.9f, 0.99f};
    float emaRate = 1e-3f;
    uint32_t emaWarmup = 1001; ///< steps during which the average copies the parameters
    /// Only the forward pass (no loss, gradients or optimizer), e.g. to render with trained parameters; step() then
    /// computes image() from the batch's G-buffers, coordinates, classes and rows.
    bool forwardOnly = false;
    bool profile = false; ///< time every element of a step on the GPU (profile())
};

/**
 * @brief Trains the photo enhancer of scripts/neural/train_photo_enhancer.py (--texture --levels L --norm channel
 * --appearance D) with its photometric losses, entirely in Klartraum: the learnable surface texture (hashTexture()),
 * the U-Net generator (3 x 3 convolutions, ChannelNorm, ELU, bilinear upsampling, skips), FiLM by per-image appearance
 * codes, the L1 and block-mean L1 losses with the codes' L2 penalty, the backward pass of all of it, gradient clipping,
 * Adam in three parameter groups (network, texture, codes) and an exponential moving average of every parameter.
 *
 * One step is one pre-recorded graph, submitted unchanged: upload a batch with setBatch(), then step(). Parameters are
 * named as in the PyTorch state dict ("generator.down.0.0.weight", "codes.weight"; the texture's grids together as
 * "texture.table" [grids][levels][table][features]), so checkpoints can move between both.
 */
class PhotoEnhancerTraining {
public:
    PhotoEnhancerTraining(VulkanContext& vulkanContext, const PhotoEnhancerTrainingConfig& config);
    ~PhotoEnhancerTraining();

    const PhotoEnhancerTrainingConfig& getConfig() const { return config; }

    /**
     * @brief The next step's batch: the G-buffer channels [batch][11][size][size], the surface coordinates
     * [batch][3][size][size], class indices [batch][size][size], target photos [batch][3][size][size] in [0, 1] and
     * the crops' appearance code rows [batch] (ignored without codes).
     */
    void setBatch(const std::vector<float>& channels, const std::vector<float>& coordinates,
                  const std::vector<float>& classes, const std::vector<float>& targets, const std::vector<float>& rows);
    /// One training step on the batch.
    void step();
    /// With PhotoEnhancerTrainingConfig::profile: the mean GPU time of a step per element kind, in milliseconds.
    std::vector<std::pair<std::string, double>> profile() const;
    /// The photometric loss of the last step (without the codes' penalty and the perceptual loss).
    float loss() const;
    /// The perceptual loss of the last step (0 without).
    float perceptualLoss() const;
    /// The discriminator's loss, the adversarial loss and the feature matching of the last step (0 without).
    float discriminatorLoss() const;
    float adversarialLoss() const;
    float featureMatchingLoss() const;
    /// The next step's augmentation numbers, [2 batch][6] (real pairs, then fake pairs; see training::augmentPair()),
    /// instead of random ones.
    void setAugmentation(const std::vector<float>& numbers);
    /// The generator's images of the last step, [batch][3][size][size].
    std::vector<float> image() const;
    /// The generator's input of the last step, [batch][11 G-buffer channels + texture features][size][size].
    std::vector<float> generatorInput() const;

    /// The parameters' names, in the order of the PyTorch state dict (texture first, codes last).
    std::vector<std::string> parameterNames() const;
    std::shared_ptr<TensorElement<float>> parameter(const std::string& name) const;
    std::shared_ptr<TensorElement<float>> gradient(const std::string& name) const;
    /// The exponential moving average of a parameter.
    std::shared_ptr<TensorElement<float>> average(const std::string& name) const;
    /// Sets a parameter and its average, e.g. from a checkpoint.
    void setParameter(const std::string& name, const std::vector<float>& values);

    /**
     * @brief Writes every parameter (or its average) to `<directory>/<name>.bin` (float32), their names and shapes, one
     * per line, to `<directory>/parameters.txt` and the texture's shape to `<directory>/texture.txt`.
     */
    void save(const std::string& directory, bool averages = true) const;
    /// Reads parameters written by save() (into parameter and average); every parameter must be there.
    void load(const std::string& directory);
    /// The mean of the training images' appearance codes (of the averages), e.g. for a new view.
    std::vector<float> meanCode() const;
    /// Forward only: renders every crop of the batch with this appearance code (whatever its rows).
    void setAppearance(const std::vector<float>& code);

private:
    struct Parameter;
    struct Block;
    struct Critic;

    std::shared_ptr<TensorElement<float>> tensor(const std::vector<uint32_t>& shape);
    Parameter& addParameter(const std::string& name, const std::vector<uint32_t>& shape, int group);
    const Parameter& find(const std::string& name) const;
    void add(ComputeGraphElementPtr element);
    void forwardBlock(Block& block);
    void backwardBlock(Block& block);
    void buildDiscriminator();
    void forwardCritic(Critic& critic, std::shared_ptr<TensorElement<float>> pairs);
    /// The critic's backward pass from its logits' gradients: its parameters' gradients (withWeights), its input's
    /// gradient into `inputGradient` (if given), `featureGradient(scale, layer)` called once each layer's output
    /// gradient is written (to add to it).
    void backwardCritic(Critic& critic, bool withWeights, std::shared_ptr<TensorElement<float>> inputGradient,
                        const std::function<void(Critic&, size_t, size_t)>& featureGradient);
    std::vector<float> randomAugmentation();

    VulkanContext& vulkanContext;
    PhotoEnhancerTrainingConfig config;
    std::vector<Parameter> parameters;
    std::map<std::string, size_t> parameterIndex;
    std::vector<Block> blocks; ///< the down blocks, then the up blocks
    /// The step's elements in submission order; each depends on the previous one.
    std::vector<ComputeGraphElementPtr> sequence;
    std::unique_ptr<ComputeGraph> graph;

    std::shared_ptr<TensorElement<float>> channels, coordinates, classes, targets, rows;
    std::shared_ptr<TensorElement<float>> input; ///< the generator's input: G-buffer channels, texture features
    std::shared_ptr<TensorElement<float>> logits, output, outputGradient, logitGradient;
    std::shared_ptr<TensorElement<float>> lossShares, lossTotal, normTotal, scalesDummy;
    /// FiLM: the picked codes, the MLP's pre-activations and activations, and their gradients.
    std::shared_ptr<TensorElement<float>> code, hidden1, hidden1Pre, hidden2, hidden2Pre;
    std::shared_ptr<TensorElement<float>> codeGradient, hidden1Gradient, hidden1PreGradient, hidden2Gradient,
        hidden2PreGradient;
    std::shared_ptr<TensorElement<uint32_t>> stepCounter;
    std::unique_ptr<VggLoss> vgg; ///< the perceptual loss, or none
    /// The discriminator: its passes (the D step's, then the G step's on real and on fake pairs) and their tensors.
    std::vector<std::unique_ptr<Critic>> critics;
    std::shared_ptr<TensorElement<float>> augmentation;
    std::shared_ptr<TensorElement<uint32_t>> discriminatorCounter; ///< the discriminator's Adam steps
    std::vector<std::shared_ptr<TensorElement<float>>> discriminatorTotals, adversarialTotals, matchingTotals;
    bool augmentationGiven = false;
    std::mt19937 random;
};

} // namespace klartraum::neural

#endif // KLARTRAUM_NEURAL_PHOTO_ENHANCER_TRAINING_HPP
