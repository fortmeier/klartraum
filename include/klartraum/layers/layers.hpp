// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_LAYERS_LAYERS_HPP
#define KLARTRAUM_LAYERS_LAYERS_HPP

#include <array>
#include <cstdint>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_context.hpp"

/**
 * @brief Neural-network layers as compute-graph elements.
 *
 * Each factory returns one element that runs a compute shader from
 * `shaders/onnx/` on tensors. The factories take plain shapes and attributes;
 * OnnxNetwork builds its operations with them, and applications can place the
 * same layers directly in a compute graph.
 *
 * Slot convention: connect the layer's input tensors with
 * `setInput(tensor, slot)` at slots 0, 1, ... in the order the factory lists
 * them, then its output tensors at the following slots. Shapes are in ONNX
 * order (outermost first, e.g. NCHW) and must match the tensors connected;
 * ranks above four are not supported unless noted. Every layer writes all of
 * its output elements.
 */
namespace klartraum::layers {

/** @brief A tensor shape, outermost dimension first. */
using Shape = std::vector<uint32_t>;

/** @brief Element types of tensors that layers accept. */
enum class ElementType { Float32, Int64 };

/** @brief Number of elements of @p shape; throws if it exceeds 2^32 - 1. */
uint32_t elementCount(const Shape& shape);

/**
 * @brief The shape @p lhs and @p rhs broadcast to (NumPy/ONNX rules).
 * @throws std::runtime_error If the shapes cannot be broadcast.
 */
Shape broadcastShape(const Shape& lhs, const Shape& rhs);

/** @brief Elementwise operations with two inputs. */
enum class BinaryOp { Add, Sub, Mul, Div };

/**
 * @brief `output = lhs op rhs`, broadcasting both inputs to @p output.
 *
 * Slots: 0 lhs, 1 rhs, 2 output (float tensors).
 */
ComputeGraphElementPtr binary(VulkanContext& vulkanContext, BinaryOp op, const Shape& lhs, const Shape& rhs,
                              const Shape& output);

/** @brief Elementwise operations with one input. */
enum class UnaryOp { Sigmoid, Cos, Sin, Sqrt, Erf, CastInt64ToFloat };

/**
 * @brief `output = op(input)` for every element; input and output have @p shape.
 *
 * Slots: 0 input, 1 output. The input is an int64 tensor for
 * UnaryOp::CastInt64ToFloat and a float tensor otherwise.
 */
ComputeGraphElementPtr unary(VulkanContext& vulkanContext, UnaryOp op, const Shape& shape);

/**
 * @brief `output = max(input, 0)`; input and output have @p shape.
 *
 * Slots: 0 input, 1 output.
 */
ComputeGraphElementPtr relu(VulkanContext& vulkanContext, const Shape& shape);

/** @brief Attributes of a 2D (transposed) convolution. */
struct ConvAttributes {
    std::array<uint32_t, 2> kernelShape{1, 1};
    std::array<uint32_t, 2> strides{1, 1};
    std::array<uint32_t, 4> pads{0, 0, 0, 0}; ///< top, left, bottom, right
    std::array<uint32_t, 2> dilations{1, 1};
    uint32_t group = 1;
};

struct Conv3dAttributes {
    std::array<uint32_t, 3> kernelShape{1, 1, 1};
    std::array<uint32_t, 3> strides{1, 1, 1};
    std::array<uint32_t, 6> pads{0, 0, 0, 0, 0, 0}; ///< depth, height, width begins, then their ends
    std::array<uint32_t, 3> dilations{1, 1, 1};
    uint32_t group = 1;
};

/**
 * @brief 3D convolution of an NCDHW input with OIDHW weights.
 *
 * Asymmetric pads allow causal temporal convolutions. Slots: 0 input,
 * 1 weights, 2 bias (O), 3 output of shape @p output.
 */
ComputeGraphElementPtr conv3d(VulkanContext& vulkanContext, const Conv3dAttributes& attributes, const Shape& input,
                              const Shape& weights, const Shape& output);

/**
 * @brief 2D convolution of an NCHW input with OIHW weights.
 *
 * Slots: 0 input, 1 weights, 2 bias, 3 output of shape @p output. The shader
 * always reads a bias; @p bias is its shape (O), or empty for a bias of one
 * element.
 */
ComputeGraphElementPtr conv(VulkanContext& vulkanContext, const ConvAttributes& attributes, const Shape& input,
                            const Shape& weights, const Shape& bias, const Shape& output);

/**
 * @brief 2D transposed convolution of an NCHW input with IOHW weights.
 *
 * The output is N x O x ((H - 1) * stride - 2 * pad + kernel) x ... .
 * Slots: 0 input, 1 weights, 2 bias, 3 output; see conv() for @p bias.
 */
ComputeGraphElementPtr convTranspose(VulkanContext& vulkanContext, const ConvAttributes& attributes, const Shape& input,
                                     const Shape& weights, const Shape& bias);

/**
 * @brief Normalizes every channel of an N x C x ... input over its spatial
 * dimensions, then scales and shifts it per channel.
 *
 * Slots: 0 input, 1 scale (C), 2 bias (C), 3 output (the input's shape).
 */
ComputeGraphElementPtr instanceNormalization(VulkanContext& vulkanContext, const Shape& input, float epsilon = 1e-5f);

/**
 * @brief Normalizes over the last axis, then scales and shifts.
 *
 * Slots: 0 input, 1 scale, 2 bias (both the last axis' size), 3 output.
 */
ComputeGraphElementPtr layerNormalization(VulkanContext& vulkanContext, const Shape& input, float epsilon = 1e-5f);

/**
 * @brief Batched matrix product `output = lhs x rhs` over the last two axes;
 * a batch of one broadcasts.
 *
 * Slots: 0 lhs, 1 rhs, 2 output.
 */
ComputeGraphElementPtr matMul(VulkanContext& vulkanContext, const Shape& lhs, const Shape& rhs, const Shape& output);

/**
 * @brief `output = input x weightsᵀ + bias` for a rows x K input and
 * columns x K weights.
 *
 * Slots: 0 input, 1 weights, 2 bias (columns), 3 output (rows x columns).
 */
ComputeGraphElementPtr gemm(VulkanContext& vulkanContext, const Shape& input, const Shape& weights);

/**
 * @brief `softmax(query x keyᵀ) x value` for rank-four B x H x queries x depth
 * tensors, as one kernel.
 *
 * Slots: 0 query, 1 key (B x H x depth x keys), 2 value, 3 output.
 */
ComputeGraphElementPtr fusedAttention(VulkanContext& vulkanContext, const Shape& query, const Shape& key,
                                      const Shape& value, const Shape& output);

/**
 * @brief Softmax over the last axis.
 *
 * Slots: 0 input, 1 output (the input's shape).
 */
ComputeGraphElementPtr softmax(VulkanContext& vulkanContext, const Shape& shape);

/**
 * @brief Mean over the contiguous axes [@p firstAxis, @p firstAxis + @p axisCount).
 *
 * Slots: 0 input, 1 output (the input's shape with the reduced axes removed or kept as 1).
 */
ComputeGraphElementPtr reduceMean(VulkanContext& vulkanContext, const Shape& input, uint32_t firstAxis,
                                  uint32_t axisCount);

/**
 * @brief Joins two tensors along @p axis.
 *
 * Slots: 0 lhs, 1 rhs, 2 output.
 */
ComputeGraphElementPtr concat(VulkanContext& vulkanContext, const Shape& lhs, const Shape& rhs, uint32_t axis,
                              const Shape& output);

/**
 * @brief Splits a tensor along @p axis into three outputs, the first two with
 * @p firstSize and @p secondSize elements on that axis.
 *
 * Slots: 0 input, 1 split sizes (int64, bound but not read), 2-4 outputs.
 */
ComputeGraphElementPtr split3(VulkanContext& vulkanContext, const Shape& input, uint32_t axis, uint32_t firstSize,
                              uint32_t secondSize);

/**
 * @brief Broadcasts @p input to @p output.
 *
 * Slots: 0 input, 1 target shape (int64, bound but not read), 2 output;
 * input and output are of @p type.
 */
ComputeGraphElementPtr expand(VulkanContext& vulkanContext, ElementType type, const Shape& input, const Shape& output);

/**
 * @brief Picks entries of a float tensor along @p axis by int64 indices.
 *
 * Slots: 0 data, 1 indices, 2 output.
 */
ComputeGraphElementPtr gather(VulkanContext& vulkanContext, const Shape& data, const Shape& indices, uint32_t axis,
                              const Shape& output);

/**
 * @brief Nearest-neighbour resize of an NCHW tensor to @p output's height and
 * width.
 *
 * Slots: 0 input, 1 scales (bound but not read), 2 output.
 */
ComputeGraphElementPtr resizeNearest(VulkanContext& vulkanContext, const Shape& input, const Shape& output);

/**
 * @brief Permutes the axes of @p input (rank up to eight).
 *
 * Slots: 0 input, 1 output.
 */
ComputeGraphElementPtr transpose(VulkanContext& vulkanContext, const Shape& input, const Shape& output,
                                 const std::vector<uint32_t>& permutation);

/** @brief Which unused parameter tensors a slice binds after its input (as ONNX Slice does). */
enum class SliceParameters { StartsEnds = 2, StartsEndsAxes = 3, StartsEndsAxesSteps = 4 };

/**
 * @brief Copies the part of @p input that starts at @p start on the one axis
 * where @p input and @p output differ.
 *
 * Slots: 0 input, then the parameter tensors @p parameters names (bound but
 * not read), then the output; input and output are of @p type.
 */
ComputeGraphElementPtr slice(VulkanContext& vulkanContext, ElementType type, const Shape& input, const Shape& output,
                             uint32_t start, SliceParameters parameters);

} // namespace klartraum::layers

#endif // KLARTRAUM_LAYERS_LAYERS_HPP
