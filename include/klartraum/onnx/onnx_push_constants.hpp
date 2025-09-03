#ifndef KLARTRAUM_ONNX_PUSH_CONSTANTS_HPP
#define KLARTRAUM_ONNX_PUSH_CONSTANTS_HPP

#include <cstdint>

namespace klartraum {

struct TensorOpPushConstants {
    uint32_t dimInput[4];
    uint32_t dimOutput[4];
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
    uint32_t dimInput[4];    // [batch, input_channels, input_height, input_width]
    uint32_t dimWeights[4];  // [input_channels, output_channels, kernel_height, kernel_width]
    uint32_t dimOutput[4];   // [batch, output_channels, output_height, output_width]
    uint32_t dimBias[4];     // [output_channels] (only first element used)
    uint32_t kernel_shape[8];
    uint32_t strides[8];
    uint32_t pads[8];
    uint32_t dilations[8];
    uint32_t groups[8];
    uint32_t output_padding[8];
};

} // namespace klartraum

#endif // KLARTRAUM_ONNX_PUSH_CONSTANTS_HPP
