// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of a bilinear upsampling by two with half-pixel centres (layers::resizeLinear without
// alignCorners, PyTorch's align_corners=False): every input pixel gathers the output gradients times the weights with
// which the outputs near it sampled it, the same clamped weights as the forward shader. One invocation per input
// pixel.

struct PushConstants {
    uint planes; // batch x channels
    uint inputHeight;
    uint inputWidth;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> outputGradient; // [planes][2 inputHeight][2 inputWidth]
[[vk::binding(1, 0)]] RWStructuredBuffer<float> inputGradient; // [planes][inputHeight][inputWidth]

// The weight with which output coordinate o samples input coordinate i along an axis of `size` inputs.
float weightOf(uint o, uint i, uint size) {
    const float s = clamp((float(o) + 0.5) * 0.5 - 0.5, 0.0, float(size - 1u));
    const uint i0 = uint(s);
    const uint i1 = min(i0 + 1u, size - 1u);
    const float f = s - float(i0);
    float w = 0.0;
    if (i0 == i) w += 1.0 - f;
    if (i1 == i) w += f;
    return w;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.planes * pc.inputHeight * pc.inputWidth) return;
    const uint x = id.x % pc.inputWidth;
    const uint y = (id.x / pc.inputWidth) % pc.inputHeight;
    const uint plane = id.x / (pc.inputWidth * pc.inputHeight);
    const uint outputHeight = 2u * pc.inputHeight, outputWidth = 2u * pc.inputWidth;
    float sum = 0.0;
    // Outputs 2i - 2 .. 2i + 2 are the only ones that sample input i.
    for (int oy = int(2u * y) - 2; oy <= int(2u * y) + 2; ++oy) {
        if (oy < 0 || oy >= int(outputHeight)) continue;
        const float wy = weightOf(uint(oy), y, pc.inputHeight);
        if (wy == 0.0) continue;
        for (int ox = int(2u * x) - 2; ox <= int(2u * x) + 2; ++ox) {
            if (ox < 0 || ox >= int(outputWidth)) continue;
            const float wx = weightOf(uint(ox), x, pc.inputWidth);
            if (wx == 0.0) continue;
            sum += wy * wx * outputGradient[(plane * outputHeight + uint(oy)) * outputWidth + uint(ox)];
        }
    }
    inputGradient[id.x] = sum;
}
