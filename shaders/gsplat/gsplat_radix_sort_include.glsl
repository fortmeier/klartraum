#include "gsplat_types.glsl"

#extension GL_EXT_debug_printf : enable

#extension GL_EXT_scalar_block_layout : enable

layout(scalar, binding = 0) buffer BufferA {
    Gaussian2D gaussiansA[];
};

layout(scalar, binding = 2) buffer BufferB {
    Gaussian2D gaussiansB[];
};

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

layout(scalar, binding = 7) buffer InputBufferIndexA {
    uint inputBufferIndexA[];
};

layout(scalar, binding = 8) buffer InputBufferIndexB {
    uint inputBufferIndexB[];
};


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


// idea: depending on the pass, read/write either from the buffers directly, (pass 0 and 11)
// or alternatively from index buffers
// this way, the other shaders do not have to change

Gaussian2D getInputGaussian(uint idx) {
    if (pushConstants.pass == 0) {
        inputBufferIndexA[idx] = idx;
    }

    if (pushConstants.pass % 2 == 0) {
        return gaussiansA[inputBufferIndexA[idx]];
    } else {
        return gaussiansA[inputBufferIndexB[idx]];
    }
}

void setOutputGaussian(uint idxNew, uint idxOld) {
    if (pushConstants.pass % 2 == 0) {
        inputBufferIndexB[idxNew] = inputBufferIndexA[idxOld];
    } else {
        inputBufferIndexA[idxNew] = inputBufferIndexB[idxOld];
    }

    if (pushConstants.pass == 11) {
        // for the last pass, we need to write the final output to the output buffer
        // this is done by writing to the B buffer
        // (one might expect to write again to the A buffer, since 11 % 2 == 1,
        // but that would require to read and write from the same buffer in the last pass,
        // which will give wrong results)
        gaussiansB[idxNew] = gaussiansA[inputBufferIndexA[idxNew]];
    }
}