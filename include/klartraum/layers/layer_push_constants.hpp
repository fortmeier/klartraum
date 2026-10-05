// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_LAYERS_LAYER_PUSH_CONSTANTS_HPP
#define KLARTRAUM_LAYERS_LAYER_PUSH_CONSTANTS_HPP

#include <cstdint>

namespace klartraum {

struct TensorOpPushConstants {
    uint32_t dimInput[4];
    uint32_t dimOutput[4];
};

struct UnaryPushConstants {
    uint32_t elementCount;
};

struct BinaryBroadcastPushConstants {
    uint32_t elementCount;
    uint32_t rank;
    uint32_t lhsDims[4];
    uint32_t rhsDims[4];
    uint32_t outputDims[4];
};

struct InstanceNormalizationPushConstants {
    uint32_t batch;
    uint32_t channels;
    uint32_t spatialSize;
    float epsilon;
};

struct MatMulPushConstants {
    uint32_t batchCount;
    uint32_t rows;
    uint32_t columns;
    uint32_t reduction;
    uint32_t lhsBatchCount;
    uint32_t rhsBatchCount;
};

struct FusedAttentionPushConstants {
    uint32_t batchCount;
    uint32_t queryCount;
    uint32_t keyCount;
    uint32_t queryDepth;
    uint32_t valueDepth;
};

struct FusedAttentionBiasPushConstants {
    uint32_t batchCount; ///< B x H
    uint32_t heads;      ///< H
    uint32_t queryCount;
    uint32_t keyCount;
    uint32_t biasStrideBatch; ///< bias elements between batches (0: broadcast)
    uint32_t biasStrideHead;  ///< bias elements between heads (0: broadcast)
    uint32_t biasStrideQuery; ///< bias elements between queries (0: broadcast)
};

struct RmsNormalizationPushConstants {
    uint32_t rows;
    uint32_t width;
};

struct GemmPushConstants {
    uint32_t rows;
    uint32_t columns;
    uint32_t reduction;
};

struct LayerNormalizationPushConstants {
    uint32_t outerCount;
    uint32_t axisSize;
    float epsilon;
};

struct ConcatPushConstants {
    uint32_t outerCount;
    uint32_t lhsAxisSize;
    uint32_t rhsAxisSize;
    uint32_t innerSize;
    uint32_t elementCount;
};

struct ExpandPushConstants {
    uint32_t elementCount;
    uint32_t rank;
    uint32_t inputDims[4];
    uint32_t outputDims[4];
};

struct GatherPushConstants {
    uint32_t outerCount;
    uint32_t axisSize;
    uint32_t innerSize;
    uint32_t indexCount;
    uint32_t outputCount;
};

struct SoftmaxPushConstants {
    uint32_t outerCount;
    uint32_t axisSize;
};

struct ReducePushConstants {
    uint32_t outerCount;
    uint32_t axisSize;
    uint32_t innerSize;
};

struct SplitPushConstants {
    uint32_t outerCount;
    uint32_t axisSize;
    uint32_t innerSize;
    uint32_t split0;
    uint32_t split1;
};

struct ResizePushConstants {
    uint32_t batch;
    uint32_t channels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t outputHeight;
    uint32_t outputWidth;
};

struct SlicePushConstants {
    uint32_t rank;
    uint32_t axis;
    uint32_t start;
    uint32_t elementCount;
    uint32_t inputDims[4];
    uint32_t outputDims[4];
};

struct ConvPushConstants {
    // operation attributes
    uint32_t dilations[2];
    uint32_t groups[1];
    uint32_t kernel_shape[2];
    uint32_t pads[4];
    uint32_t strides[2];

    // tensor input sizes
    uint32_t dimInput[4];
    uint32_t dimWeights[4];
    uint32_t dimBias[1];
    uint32_t dimOutput[4];
};

struct Conv3dPushConstants {
    uint32_t inputChannels, inputDepth, inputHeight, inputWidth;
    uint32_t outputChannels, outputDepth, outputHeight, outputWidth;
    uint32_t kernelDepth, kernelHeight, kernelWidth;
    uint32_t strideDepth, strideHeight, strideWidth;
    uint32_t padDepth, padHeight, padWidth;
};

// struct ReshapePushConstants {
//     uint32_t dimInput[4];
//     uint32_t dimShape[6];
// };

struct TransposePushConstants {
    uint32_t dims; // number of dims, max. 8 in this implementation
    uint32_t dimInput[8];
    uint32_t dimOutput[8];
    uint32_t dimPerm[8];
};

struct ConvTransposePushConstants {
    uint32_t dimInput[4];   // [batch, input_channels, input_height, input_width]
    uint32_t dimWeights[4]; // [input_channels, output_channels, kernel_height, kernel_width]
    uint32_t dimOutput[4];  // [batch, output_channels, output_height, output_width]
    uint32_t dimBias[4];    // [output_channels] (only first element used)
    uint32_t kernel_shape[8];
    uint32_t strides[8];
    uint32_t pads[8];
    uint32_t dilations[8];
    uint32_t groups[8];
    // uint32_t output_padding[8];
};

} // namespace klartraum

#endif // KLARTRAUM_LAYERS_LAYER_PUSH_CONSTANTS_HPP
