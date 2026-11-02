// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Bilinear Resize of an NCHW tensor with half-pixel centres (ONNX half_pixel / pytorch_half_pixel, PyTorch's
// align_corners=False): output pixel x samples the input at (x + 0.5) * inputWidth / outputWidth - 0.5, clamped
// to the border pixels.

struct PushConstants {
    uint batch;
    uint channels;
    uint inputHeight;
    uint inputWidth;
    uint outputHeight;
    uint outputWidth;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> scales;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> outputData;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint index = globalId.x;
    uint total = pc.batch * pc.channels * pc.outputHeight * pc.outputWidth;
    if (index >= total) return;
    uint x = index % pc.outputWidth;
    uint value = index / pc.outputWidth;
    uint y = value % pc.outputHeight;
    uint plane = value / pc.outputHeight;
    float sx = clamp((float(x) + 0.5) * float(pc.inputWidth) / float(pc.outputWidth) - 0.5, 0.0,
                     float(pc.inputWidth - 1));
    float sy = clamp((float(y) + 0.5) * float(pc.inputHeight) / float(pc.outputHeight) - 0.5, 0.0,
                     float(pc.inputHeight - 1));
    uint x0 = uint(sx);
    uint y0 = uint(sy);
    uint x1 = min(x0 + 1, pc.inputWidth - 1);
    uint y1 = min(y0 + 1, pc.inputHeight - 1);
    float fx = sx - float(x0);
    float fy = sy - float(y0);
    uint base = plane * pc.inputHeight * pc.inputWidth;
    float top = lerp(inputData[base + y0 * pc.inputWidth + x0], inputData[base + y0 * pc.inputWidth + x1], fx);
    float bottom = lerp(inputData[base + y1 * pc.inputWidth + x0], inputData[base + y1 * pc.inputWidth + x1], fx);
    outputData[index] = lerp(top, bottom, fy);
}
