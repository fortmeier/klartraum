// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

struct PushConstants {
    uint outerCount;
    uint axisSize;
    uint innerSize;
    uint split0;
    uint split1;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
// The split-size tensor is bound for ONNX slot compatibility. Static split
// sizes are carried in push constants, so its 64-bit payload is not read.
[[vk::binding(1, 0)]] StructuredBuffer<uint> sizes;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> output0;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output1;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> output2;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    uint total = pc.outerCount * pc.axisSize * pc.innerSize;
    if (index >= total) return;
    uint inner = index % pc.innerSize;
    uint axisAndOuter = index / pc.innerSize;
    uint axis = axisAndOuter % pc.axisSize;
    uint outer = axisAndOuter / pc.axisSize;
    if (axis < pc.split0) {
        output0[(outer * pc.split0 + axis) * pc.innerSize + inner] = inputData[index];
    } else if (axis < pc.split0 + pc.split1) {
        uint localAxis = axis - pc.split0;
        output1[(outer * pc.split1 + localAxis) * pc.innerSize + inner] = inputData[index];
    } else {
        uint split2 = pc.axisSize - pc.split0 - pc.split1;
        uint localAxis = axis - pc.split0 - pc.split1;
        output2[(outer * split2 + localAxis) * pc.innerSize + inner] = inputData[index];
    }
}
