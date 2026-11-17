// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_NEURAL_PHOTO_ENHANCER_TRAINING_HPP
#define KLARTRAUM_NEURAL_PHOTO_ENHANCER_TRAINING_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/layers/training.hpp"
#include "klartraum/neural/hash_texture.hpp"
#include "klartraum/neural/vgg_loss.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::neural {

/// The latent-conditioned MLP texture (latent_renderer.MlpTexture): a per-pixel MLP of 1 x 1 convolutions with ELU
/// from the positional encoding of the surface coordinates and the latent G-buffer to texture features.
struct TextureMlp {
    uint32_t frequencies = 0; ///< octaves of the positional encoding per coordinate (0: no MLP texture)
    uint32_t width = 32;      ///< channels of the hidden layers
    uint32_t outputs = 8;     ///< texture features
    uint32_t hidden = 2;      ///< hidden layers
    /// Residual blocks after the first (latent_renderer.ResidualMlpTexture): each a texture MLP of its own octaves and
    /// hidden width (pairs of frequencies and width, parameters "generator.texture_residual.<j>."), its features added
    /// to the previous blocks', so that the finer blocks correct what the coarser ones leave (none: one MLP).
    std::vector<std::pair<uint32_t, uint32_t>> residual;

    /// The MLP texture of a saved model (the "mlp F W K H" and "mlp-residual F1 W1 F2 W2 ..." lines of its
    /// texture.txt), frequencies 0 without one.
    static TextureMlp fromFile(const std::string& path);
};

/// The code-mixed feature texture (latent_renderer.TextureBasis, a neural texture as in Deferred Neural Rendering mixed
/// by the latent): per face of the object and per resolution a base layer and one layer per latent number of
/// `channels` features, sampled bilinearly at (u, v); the pixel's features are the base plus each layer times its
/// latent, summed over the resolutions (parameters "texture.basis.<level>", trained with textureAdam).
struct TextureBasis {
    uint32_t channels = 0;             ///< features (0: no feature texture)
    std::vector<uint32_t> resolutions; ///< texels per side of each level
    uint32_t faces = 6;                ///< faces of the object (the third coordinate's id x faces)
    /// Weights of the regularisation (training::textureBasisRegularize()) of the coarsest level: the squared texels'
    /// and the squared differences of neighbouring texels'; level k's are these times resolutions[k] / resolutions[0],
    /// so that finer levels, whose texels fewer pixels see, are held back more.
    float decay = 0.0f;
    float smoothness = 0.0f;

    bool enabled() const { return channels > 0 && !resolutions.empty(); }
    /// The feature texture of a saved model (the "basis C F R1 R2 ..." and "basis-regularize D S" lines of its
    /// texture.txt), none without one.
    static TextureBasis fromFile(const std::string& path);
};

/// The latent size of a saved latent model: the "latent D" line of its texture.txt, else half its encoder head's
/// outputs (older models), else 0.
uint32_t savedLatentSize(const std::string& directory);
/// Whether a saved latent model's generator takes the latents (PhotoEnhancerTrainingConfig::latentToDecoder): false
/// where its texture.txt says "decoder-latent 0".
bool savedLatentToDecoder(const std::string& directory);

/// The photo enhancer that PhotoEnhancerTraining trains, and how.
struct PhotoEnhancerTrainingConfig {
    uint32_t batch = 16; ///< crops per step
    uint32_t size = 128; ///< crop width and height, a multiple of 2^(levels - 1)
    /// The U-Net's channels per level (3 levels: the local generator of train_photo_enhancer.py --levels 3).
    std::vector<uint32_t> widths{32, 64, 128};
    /// The surface texture's grids; width, height, batch and outputOffset are set by the training.
    HashTextureShape texture;
    /// The MLP texture instead of the grids (needs latents and a texture of no levels; its classes are the textured
    /// ones).
    TextureMlp textureMlp;
    /// The code-mixed feature texture instead (needs latents, a texture of no levels whose classes are the textured
    /// ones and no MLP texture).
    TextureBasis textureBasis;
    /// Forward only: render with the texture's features zeroed (its parameters still load), to measure how much the
    /// generator relies on them.
    bool textureOff = false;
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
    /// Whether the discriminator sees the G-buffer channels with the image (a conditional PatchGAN) or the image only.
    bool conditionalDiscriminator = true;
    /// Scales of the PatchGAN, each at half the previous resolution (2: full and half).
    uint32_t criticScales = 2;
    /// The conditional discriminator's G-buffer channels: the first this many (0: all), e.g. 1 for inverse depth.
    uint32_t criticGBufferChannels = 0;
    /// DiffAugment in the generator's step too: the generated pairs augmented (with its own random numbers) before the
    /// discriminator judges them, as in its own step.
    bool augmentGeneratorStep = false;
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
    /// Geometry channels the batch uploads (setBatch()); 11 for the photo enhancer's RGB, depth and one-hot classes,
    /// e.g. 4 for inverse depth and normals.
    uint32_t gbufferChannels = 11;
    /// The latent-conditioned renderer (scripts/neural/latent_renderer.py): numbers per object of a latent G-buffer
    /// that joins the generator's input after the texture features, from a VAE encoder of the target photo (0: none).
    uint32_t latentSize = 0;
    uint32_t latentObjects = 1; ///< objects 1..latentObjects of the object map (setObjects())
    float klWeight = 1e-3f;     ///< weight of the latents' KL divergence (summed over objects, mean over batch)
    /// Nats of KL per latent number that cost nothing (free bits), so the encoder can use the latents freely.
    float freeBits = 0.0f;
    /// Weight of the temporal consistency loss (latent_renderer.flow_warp_loss()): a second, nearby view of every
    /// sample (setSecondView()) rendered with the same latents, the first render warped onto it by the views' flow
    /// as the target, the mean squared difference on the pixels both views see (0: none).
    float temporal = 0.0f;
    std::vector<uint32_t> encoderWidths{32, 64, 128}; ///< stride-2 layers down to an eighth
    /// Forward only: the latents from the encoder of the batch's photos (their means), else given (setLatents()).
    bool encodeLatents = false;
    /// Learned latents instead of the encoder (an auto-decoder): a table "latents.weight" of this many rows, each
    /// the latentObjects x latentSize latents of one appearance (e.g. one crate), the batch's rows chosen by
    /// setBatch(); trained with codesAdam, no KL (0: none).
    uint32_t latentCodes = 0;
    float latentCodeDecay = 1e-2f; ///< weight of the latent table's L2 penalty (the batch's mean squared row norm)
    /// Whether the latent G-buffer also enters the generator; false: only the texture sees the latents and the
    /// generator's latent channels stay zero, so every difference between latents goes through the texture.
    bool latentToDecoder = true;
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
     * the crops' rows [batch] of the appearance codes or of the latent table (ignored without either).
     */
    void setBatch(const std::vector<float>& channels, const std::vector<float>& coordinates,
                  const std::vector<float>& classes, const std::vector<float>& targets, const std::vector<float>& rows);
    /// One training step on the batch.
    void step();
    /// With PhotoEnhancerTrainingConfig::profile: the mean GPU time of a step per element kind, in milliseconds.
    std::vector<std::pair<std::string, double>> profile() const;
    /// With PhotoEnhancerTrainingConfig::profile: the mean GPU time of every element of a step in execution order,
    /// in milliseconds, and (gpuSpan()) of the whole step including the barriers between elements.
    std::vector<std::pair<std::string, double>> profileElements() const;
    double gpuSpan() const;
    /// Starts the profiling means anew, e.g. after warm-up steps.
    void resetProfile();
    /// The photometric loss of the last step (without the codes' penalty and the perceptual loss).
    float loss() const;
    /// The perceptual loss of the last step (0 without).
    float perceptualLoss() const;
    /// The discriminator's loss, the adversarial loss and the feature matching of the last step (0 without).
    float discriminatorLoss() const;
    float adversarialLoss() const;
    float featureMatchingLoss() const;
    /// The objects of the batch's pixels, [batch][size][size] (indices as floats; 0: none), for the latents.
    void setObjects(const std::vector<float>& objects);
    /// Forward only without encoder: every object's latent, [batch][latentObjects][latentSize].
    void setLatents(const std::vector<float>& latents);
    /// The next step's VAE noise, [batch][latentObjects][latentSize], instead of random numbers.
    void setLatentNoise(const std::vector<float>& noise);
    /// The latents of the last step, [batch][latentObjects][latentSize], and their KL divergence (unweighted).
    std::vector<float> latents() const;
    float klLoss() const;
    /// The second views of the batch for the temporal loss: G-buffer channels, coordinates, classes and objects as in
    /// setBatch() and setObjects(), the flow onto the first views ([batch][2][size][size]: each pixel's position in
    /// the first view, in pixels with centres at i + 0.5) and where the first view sees it ([batch][size][size]).
    void setSecondView(const std::vector<float>& channels, const std::vector<float>& coordinates,
                       const std::vector<float>& classes, const std::vector<float>& objects,
                       const std::vector<float>& flow, const std::vector<float>& valid);
    /// The temporal loss's mean squared difference of the last step (unweighted).
    float temporalLoss() const;
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
    /// The U-Net and its output layer on `input` with these blocks' activations: the last up block's output, the
    /// logits and the image.
    std::tuple<std::shared_ptr<TensorElement<float>>, std::shared_ptr<TensorElement<float>>,
               std::shared_ptr<TensorElement<float>>>
    forwardGenerator(std::vector<Block>& blocks, std::shared_ptr<TensorElement<float>> input);
    /// The backward pass of forwardGenerator() from the image's gradient, down to blocks[0].dIn; the parameters'
    /// gradients go to gradientOf().
    void backwardGenerator(std::vector<Block>& blocks, std::shared_ptr<TensorElement<float>> x,
                           std::shared_ptr<TensorElement<float>> logits,
                           std::shared_ptr<TensorElement<float>> outputGradient,
                           std::shared_ptr<TensorElement<float>> logitGradient);
    /// Where the backward pass writes a parameter's gradient: its gradient, or a tensor of gradientRedirect.
    std::shared_ptr<TensorElement<float>> gradientOf(const std::string& name) const;
    struct EncoderLayer;
    /// The encoder's layers on `image` and its head into `statistics` ([batch][2 latentSize][size / 8][size / 8]).
    std::vector<EncoderLayer> forwardEncoder(std::shared_ptr<TensorElement<float>> image,
                                             std::shared_ptr<TensorElement<float>> statistics);
    /// The input gradient of forwardEncoder() from its statistics' gradient, without the parameters' gradients.
    std::shared_ptr<TensorElement<float>>
    backwardEncoderInput(std::vector<EncoderLayer>& layers, std::shared_ptr<TensorElement<float>> statisticsGradient);
    /// One pass of the MLP texture: its encoding, input (encoding and latent G-buffer), layers and output.
    struct MlpPass {
        std::shared_ptr<TensorElement<float>> encoding, in;
        std::vector<std::shared_ptr<TensorElement<float>>> pre, activated; ///< each layer's output, before / after ELU
        std::vector<MlpPass> residual; ///< the MLP texture's residual blocks (TextureMlp::residual)
    };
    /// A texture MLP (parameters prefix + "0.", "2.", ...): the positional encoding of these coordinates and classes
    /// and the latent G-buffer (generatorInput's channels from latentOffset) through hidden ELU layers of `width` to
    /// `outputs` values (pass.pre.back()).
    MlpPass forwardMlp(const std::string& prefix, uint32_t frequencies, uint32_t width, uint32_t hidden,
                       uint32_t outputs, std::shared_ptr<TensorElement<float>> passCoordinates,
                       std::shared_ptr<TensorElement<float>> passClasses,
                       std::shared_ptr<TensorElement<float>> generatorInput, uint32_t latentOffset);
    /// The backward pass of forwardMlp() from its outputs' gradient: the parameters' gradients (to gradientOf()) and
    /// its input's gradient, whose last latentSize channels are the latent G-buffer's.
    std::shared_ptr<TensorElement<float>> backwardMlp(MlpPass& pass, const std::string& prefix, uint32_t frequencies,
                                                      uint32_t width, uint32_t hidden, uint32_t outputs,
                                                      std::shared_ptr<TensorElement<float>> outputGradient);
    /// The MLP texture on these coordinates and classes and the latent G-buffer already in `generatorInput`, its
    /// features written into `generatorInput`'s texture channels.
    MlpPass forwardTextureMlp(std::shared_ptr<TensorElement<float>> passCoordinates,
                              std::shared_ptr<TensorElement<float>> passClasses,
                              std::shared_ptr<TensorElement<float>> generatorInput);
    /// The backward pass of forwardTextureMlp() from the generator input's gradient: the MLP's parameter gradients
    /// (to gradientOf()) and its input's gradient, whose last latentSize channels are the latent G-buffer's.
    std::shared_ptr<TensorElement<float>>
    backwardTextureMlp(MlpPass& pass, std::shared_ptr<TensorElement<float>> generatorInputGradient);
    /// The texture's feature channels of the generator input.
    uint32_t textureFeatures() const;
    void buildTemporal();
    /// The backward pass of a second generator pass (its blocks, output layer input, logits and image gradient):
    /// the generator's gradients added to the first pass's, the texture's for these coordinates and classes too.
    void backwardSecondPass(std::vector<Block>& passBlocks, std::shared_ptr<TensorElement<float>> x,
                            std::shared_ptr<TensorElement<float>> passLogits,
                            std::shared_ptr<TensorElement<float>> passOutputGradient,
                            std::shared_ptr<TensorElement<float>> passCoordinates,
                            std::shared_ptr<TensorElement<float>> passClasses, MlpPass* passMlp);
    void forwardBlock(Block& block);
    void backwardBlock(Block& block);
    void buildDiscriminator();
    void buildEncoder();
    /// The latents' gradient from the latent G-buffer's (generator input and texture), into the latent table or
    /// through the VAE and the encoder.
    void buildLatentBackward();
    /// Zeroes the latent channels of a generator input once the texture has read them (latentToDecoder false).
    void hideLatents(std::shared_ptr<TensorElement<float>> generatorInput);
    /// The feature texture's size for one level (training::TextureBasisShape) on the generator input.
    training::TextureBasisShape basisShape(size_t level, bool accumulate) const;
    /// The feature texture's features of these coordinates, classes and latent images into the generator input.
    void forwardTextureBasis(std::shared_ptr<TensorElement<float>> passCoordinates,
                             std::shared_ptr<TensorElement<float>> passClasses,
                             std::shared_ptr<TensorElement<float>> passLatents,
                             std::shared_ptr<TensorElement<float>> generatorInput);
    /// Its backward pass from the generator input's gradient: the tables' gradients added (zeroed before the main
    /// pass), the latents' gradient into `latentGradient` ([batch][latentSize][size][size]).
    void backwardTextureBasis(std::shared_ptr<TensorElement<float>> passCoordinates,
                              std::shared_ptr<TensorElement<float>> passClasses,
                              std::shared_ptr<TensorElement<float>> passLatents,
                              std::shared_ptr<TensorElement<float>> generatorInputGradient,
                              std::shared_ptr<TensorElement<float>> latentGradient);
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
    /// Parameter gradients written elsewhere while the graph is built (a second generator pass), by name.
    std::map<std::string, std::shared_ptr<TensorElement<float>>> gradientRedirect;
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
    /// The latent renderer: the object map, the encoder's layers (convolution, ChannelNorm with statistics, ELU) and
    /// head, the per-object statistics, noise, latents and KL shares, and their gradients.
    std::shared_ptr<TensorElement<float>> objects, encoderStatistics, pooled, counts, latentNoise, latentValues,
        klShares, klTotal, latentGBufferGradient, latentGradient, pooledGradient, statisticsGradient, latentSums;
    /// Zeros of the latent G-buffer's shape that hideLatents() copies over the generator input's latent channels.
    std::shared_ptr<TensorElement<float>> latentZeros;
    /// The feature texture's inputs and outputs: the latent G-buffer on its own ([batch][latentSize][size][size], of
    /// the main and the temporal pass) and the main pass's latent gradient from the texture.
    std::shared_ptr<TensorElement<float>> latentImage, latentImage2, basisLatentGradient;
    struct EncoderLayer {
        uint32_t inputChannels, outputChannels, inputSize;
        std::shared_ptr<TensorElement<float>> in, a, n, stats, e, da, dn, de;
    };
    std::vector<EncoderLayer> encoderLayers;
    bool latentNoiseGiven = false;
    /// The temporal loss's second generator pass, the second views' inputs, their flow and mask, and its loss.
    std::vector<Block> temporalBlocks;
    /// The MLP texture's passes (main, temporal) and the main pass's input gradient.
    MlpPass mainMlp, temporalMlp;
    std::shared_ptr<TensorElement<float>> mlpInputGradient;
    std::shared_ptr<TensorElement<float>> channels2, coordinates2, classes2, objects2, flow, flowValid, temporalTotal;
    /// The discriminator: its passes (the D step's, then the G step's on real and on fake pairs) and their tensors.
    std::vector<std::unique_ptr<Critic>> critics;
    std::shared_ptr<TensorElement<float>> augmentation;
    /// The generator step's augmentation numbers.
    std::shared_ptr<TensorElement<float>> generatorAugmentation;
    std::shared_ptr<TensorElement<uint32_t>> discriminatorCounter; ///< the discriminator's Adam steps
    std::vector<std::shared_ptr<TensorElement<float>>> discriminatorTotals, adversarialTotals, matchingTotals;
    bool augmentationGiven = false;
    std::mt19937 random;
};

} // namespace klartraum::neural

#endif // KLARTRAUM_NEURAL_PHOTO_ENHANCER_TRAINING_HPP
