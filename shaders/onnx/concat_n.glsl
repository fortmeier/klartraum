// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Concatenation of INPUTS (3..8) tensors along one axis; included by concat<N>.comp.
// One invocation per output element: it finds the input whose axis range holds
// its axis index and copies the element.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform PushConstants {
    uint outerCount;
    uint innerSize;
    uint elementCount;
    uint totalAxis;
    uint axisSize[8];
} pc;
layout(set = 0, binding = 0) restrict readonly buffer Input0 { float input0[]; };
layout(set = 0, binding = 1) restrict readonly buffer Input1 { float input1[]; };
layout(set = 0, binding = 2) restrict readonly buffer Input2 { float input2[]; };
#if INPUTS > 3
layout(set = 0, binding = 3) restrict readonly buffer Input3 { float input3[]; };
#endif
#if INPUTS > 4
layout(set = 0, binding = 4) restrict readonly buffer Input4 { float input4[]; };
#endif
#if INPUTS > 5
layout(set = 0, binding = 5) restrict readonly buffer Input5 { float input5[]; };
#endif
#if INPUTS > 6
layout(set = 0, binding = 6) restrict readonly buffer Input6 { float input6[]; };
#endif
#if INPUTS > 7
layout(set = 0, binding = 7) restrict readonly buffer Input7 { float input7[]; };
#endif
layout(set = 0, binding = INPUTS) restrict writeonly buffer OutputBuffer { float outputData[]; };

#define TAKE(k, data) \
    if (axisIndex < start + pc.axisSize[k]) { \
        outputData[index] = data[(outer * pc.axisSize[k] + axisIndex - start) * pc.innerSize + innerIndex]; \
        return; \
    } \
    start += pc.axisSize[k];

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= pc.elementCount) return;
    const uint block = pc.totalAxis * pc.innerSize;
    const uint outer = index / block;
    const uint within = index % block;
    const uint axisIndex = within / pc.innerSize;
    const uint innerIndex = within % pc.innerSize;
    uint start = 0;
    TAKE(0, input0)
    TAKE(1, input1)
    TAKE(2, input2)
#if INPUTS > 3
    TAKE(3, input3)
#endif
#if INPUTS > 4
    TAKE(4, input4)
#endif
#if INPUTS > 5
    TAKE(5, input5)
#endif
#if INPUTS > 6
    TAKE(6, input6)
#endif
#if INPUTS > 7
    TAKE(7, input7)
#endif
}
