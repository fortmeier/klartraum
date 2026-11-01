// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// NCDHW 3D convolution as an implicit matrix product: output channels x
// output positions, reduced over input channels x kernel taps. The input
// columns are gathered on the fly, so any kernel, stride, and asymmetric
// (e.g. causal) padding works without an im2col buffer. A workgroup computes
// a 64 x 128 tile (channels x positions); every invocation accumulates 4 x 8
// outputs (positions 4 tx .. and 64 + 4 tx ..) from shared tiles read as float4s.
//
// The gather avoids per-element integer division: an invocation always stages
// the same output position (its column of the input tile), decomposed once,
// and the input channel / kernel tap of each reduction step of a tile is
// decomposed once per workgroup into a shared table.

struct PushConstants {
    uint inputChannels;
    uint inputDepth;
    uint inputHeight;
    uint inputWidth;
    uint outputChannels;
    uint outputDepth;
    uint outputHeight;
    uint outputWidth;
    uint kernelDepth;
    uint kernelHeight;
    uint kernelWidth;
    uint strideDepth;
    uint strideHeight;
    uint strideWidth;
    uint padDepth; // leading pads; trailing pads follow from the output size
    uint padHeight;
    uint padWidth;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;

static const uint TILE_M = 64;
static const uint TILE_N = 128;
static const uint TILE_K = 16;

// The weight tile: [k][channel / 4].
groupshared float4 weightTile[TILE_K][TILE_M / 4];
// The gathered input tile: [k][position / 4].
groupshared float4 inputTile[TILE_K][TILE_N / 4];
// Per reduction step of the current tile: offset of (input channel, kd, kh, kw)
// relative to the input origin; -1 marks steps past the reduction.
groupshared int tapOffset[TILE_K];
// Per reduction step of the current tile: the kernel tap (kd, kh, kw).
groupshared int3 tapKernel[TILE_K];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint localIndex: SV_GroupIndex, uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint tx = localId.x;
    const uint ty = localId.y;
    const uint channelBase = groupId.y * TILE_M;
    const uint positionBase = groupId.x * TILE_N;
    const uint batch = groupId.z;

    const uint kernelPlane = pc.kernelHeight * pc.kernelWidth;
    const uint kernelVolume = pc.kernelDepth * kernelPlane;
    const uint reduction = pc.inputChannels * kernelVolume;
    const uint outputPlane = pc.outputHeight * pc.outputWidth;
    const uint positions = pc.outputDepth * outputPlane;
    const int inputPlane = int(pc.inputHeight * pc.inputWidth);
    const int inputVolume = int(pc.inputDepth) * inputPlane;
    const uint inputBatchOffset = batch * pc.inputChannels * uint(inputVolume);

    // The input-tile column this invocation stages: steps localIndex / 128 + 2 r, position localIndex % 128.
    const uint stageColumn = localIndex % TILE_N;
    const uint stageStep = localIndex / TILE_N; // 0..1
    const uint stagePosition = positionBase + stageColumn;
    const bool stageValid = stagePosition < positions;
    const uint od = stagePosition / outputPlane;
    const uint oh = (stagePosition / pc.outputWidth) % pc.outputHeight;
    const uint ow = stagePosition % pc.outputWidth;
    const int3 origin = int3(od * pc.strideDepth, oh * pc.strideHeight, ow * pc.strideWidth) -
                        int3(pc.padDepth, pc.padHeight, pc.padWidth);
    const int3 inputExtent = int3(pc.inputDepth, pc.inputHeight, pc.inputWidth);
    const int originOffset = origin.x * inputPlane + origin.y * int(pc.inputWidth) + origin.z;

    float4 accumulator[4][2];
    for (uint i = 0; i < 4; ++i) {
        accumulator[i][0] = float4(0.0);
        accumulator[i][1] = float4(0.0);
    }

    for (uint kBase = 0; kBase < reduction; kBase += TILE_K) {
        if (localIndex < TILE_K) {
            const uint tap = kBase + localIndex;
            if (tap < reduction) {
                const uint inputChannel = tap / kernelVolume;
                const uint kernelOffset = tap - inputChannel * kernelVolume;
                const int kd = int(kernelOffset / kernelPlane);
                const int kh = int((kernelOffset / pc.kernelWidth) % pc.kernelHeight);
                const int kw = int(kernelOffset % pc.kernelWidth);
                tapKernel[localIndex] = int3(kd, kh, kw);
                tapOffset[localIndex] = int(inputChannel) * inputVolume + kd * inputPlane + kh * int(pc.inputWidth) + kw;
            } else {
                tapOffset[localIndex] = -1;
            }
        }
        // Weights are [outputChannels][reduction]: neighbouring invocations read neighbouring taps.
        for (uint element = localIndex; element < TILE_M * TILE_K; element += 256) {
            const uint m = element / TILE_K;
            const uint k = element % TILE_K;
            const uint channel = channelBase + m;
            const uint tap = kBase + k;
            weightTile[k][m / 4][m % 4] =
                channel < pc.outputChannels && tap < reduction ? weights[channel * reduction + tap] : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint k = stageStep; k < TILE_K; k += 2) {
            float value = 0.0;
            const int offset = tapOffset[k];
            if (stageValid && offset >= 0) {
                const int3 at = origin + tapKernel[k];
                if (all(at >= int3(0)) && all(at < inputExtent)) {
                    value = inputData[inputBatchOffset + uint(originOffset + offset)];
                }
            }
            inputTile[k][stageColumn / 4][stageColumn % 4] = value;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < TILE_K; ++k) {
            const float4 a = weightTile[k][ty];
            const float4 b0 = inputTile[k][tx];
            const float4 b1 = inputTile[k][16 + tx];
            for (uint i = 0; i < 4; ++i) {
                accumulator[i][0] += a[i] * b0;
                accumulator[i][1] += a[i] * b1;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint i = 0; i < 4; ++i) {
        const uint channel = channelBase + ty * 4 + i;
        if (channel >= pc.outputChannels) continue;
        const float channelBias = bias[channel];
        const uint outputOffset = (batch * pc.outputChannels + channel) * positions;
        for (uint part = 0; part < 2; ++part) {
            for (uint j = 0; j < 4; ++j) {
                const uint position = positionBase + part * 64 + tx * 4 + j;
                if (position < positions) outputData[outputOffset + position] = accumulator[i][part][j] + channelBias;
            }
        }
    }
}
