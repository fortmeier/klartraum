#include "gaussian_splatting_types.glsl"

#extension GL_EXT_debug_printf : enable

#extension GL_EXT_scalar_block_layout : enable

layout(scalar, binding = 0) buffer InputBuffer {
    Gaussian2D gaussians[];
} inputBuffer;

layout(scalar, binding = 2) buffer OutputBuffer {
    Gaussian2D gaussians[];
} outputBuffer;

layout(scalar, binding = 3) buffer CountBuffer {
    uint counts[];
} countBuffer;

layout(scalar, binding = 4) buffer OffsetBuffer {
    uint offsets[];
} offsetBuffer;

layout(scalar, binding = 5) buffer InputBuffer2 {
    uint numberTotalGaussians;
} inputBuffer2;

layout(scalar, binding = 6) buffer InputBuffer3 {
    uint histogram[];
} inputBuffer3;

layout(push_constant) uniform PushConstants {
    uint pass;
    uint numElements;
    uint numBins;

} pushConstants;

bool debug = false;

// Helper function to get the bin at a specific position
// this uses 16 bins (= 4 bits of the float value for each pass)
uint getBin(uint value, uint binMask, uint pass) {
    // Invert the sign bit for correct float sorting
    value ^= 0x80000000u;

    // depending on the pass, we need to switch between the depth value and the binMask
    uint switched = pass >= (32/4) ? binMask : value;
    // for pass >= 32, we would need to either subtract 32 or
    // use the modulo operator
    pass = pass % (32/4);
    // shift the value to the right by 4 * pass bits
    // and mask it to get the last 4 bits
    return (switched >> (pass * 4)) & 0xF;
}
