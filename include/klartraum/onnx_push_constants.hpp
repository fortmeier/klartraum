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
    uint32_t dimInput[4];
    // this might work for this example, but probably will not in the future
    uint32_t dimPerm[6];
};

} // namespace klartraum

#endif // KLARTRAUM_ONNX_PUSH_CONSTANTS_HPP
