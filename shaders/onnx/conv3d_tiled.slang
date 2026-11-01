// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// NCDHW 3D convolution as an implicit matrix product: output channels x
// output columns (batch x output positions), reduced over input channels x
// kernel taps. The input columns are gathered on the fly, so any kernel,
// stride, and asymmetric (e.g. causal) padding works without an im2col buffer.
// A workgroup of 256 invocations computes a TILE_M x TILE_N tile (channels x
// columns); every invocation accumulates 4 x 8 outputs (channels 4 ty .. 4 ty + 3,
// columns 4 tx .. and TILE_N / 2 + 4 tx ..) from shared tiles read as float4s.
// Folding the batch into the columns keeps small images from leaving most of a
// tile empty; the 256 x 32 tile serves layers with few output positions.
//
// With split-K, workgroup z takes its own range of the reduction and stores
// its unbiased partial sums; conv3d_reduce.comp.slang then adds the splits in
// order and the bias. A deep reduction over few positions thus still fills
// the GPU, and the result does not depend on scheduling.
//
// The gather avoids per-element integer division: an invocation always stages
// the same output column (its column of the input tile), decomposed once,
// and the input channel / kernel tap of each reduction step of a tile is
// decomposed once per workgroup into a shared table.
module conv3d_tiled;

public struct Conv3dPushConstants {
    public uint inputChannels;
    public uint inputDepth;
    public uint inputHeight;
    public uint inputWidth;
    public uint outputChannels;
    public uint outputDepth;
    public uint outputHeight;
    public uint outputWidth;
    public uint kernelDepth;
    public uint kernelHeight;
    public uint kernelWidth;
    public uint strideDepth;
    public uint strideHeight;
    public uint strideWidth;
    public uint padDepth; // leading pads; trailing pads follow from the output size
    public uint padHeight;
    public uint padWidth;
    public uint batchCount;
    public uint splitCount;    // workgroups along z, each reducing its own range
    public uint stepsPerSplit; // reduction steps of TILE_K per split
    public uint tileM;         // the tile of the convolution shader, for the reduce pass
    public uint tileN;
};
[[vk::push_constant]] public ConstantBuffer<Conv3dPushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> inputData;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outputData;
// With split-K: the partial sums, [split][outputChannels][columns].
[[vk::binding(4, 0)]] RWStructuredBuffer<float> partialSums;

public static const uint TILE_K = 16;

// The shared tiles of a convolution shader, which owns them: weights as
// [k][channel / 4] and gathered inputs as [k][column / 4].
public interface IConvTiles {
    static void setWeight(uint k, uint channel, float value);
    static float4 weight4(uint k, uint channelGroup);
    static void setInput(uint k, uint column, float value);
    static float4 input4(uint k, uint columnGroup);
}

// Per reduction step of the current tile: offset of (input channel, kd, kh, kw)
// relative to the input origin; -1 marks steps past the reduction range.
groupshared int tapOffset[TILE_K];
// Per reduction step of the current tile: the kernel tap (kd, kh, kw).
groupshared int3 tapKernel[TILE_K];

// Runs in a 256-wide workgroup.
public void conv3d<let TILE_M : uint, let TILE_N : uint, let SPLIT : bool, Tiles : IConvTiles>(uint localIndex,
                                                                                              uint3 groupId) {
    const uint TX = TILE_N / 8; // invocations across a tile's columns
    const uint tx = localIndex % TX;
    const uint ty = localIndex / TX;
    const uint channelBase = groupId.y * TILE_M;
    const uint columnBase = groupId.x * TILE_N;
    const uint split = groupId.z;

    const uint kernelPlane = pc.kernelHeight * pc.kernelWidth;
    const uint kernelVolume = pc.kernelDepth * kernelPlane;
    const uint reduction = pc.inputChannels * kernelVolume;
    const uint outputPlane = pc.outputHeight * pc.outputWidth;
    const uint positions = pc.outputDepth * outputPlane;
    const uint columns = pc.batchCount * positions;
    const int inputPlane = int(pc.inputHeight * pc.inputWidth);
    const int inputVolume = int(pc.inputDepth) * inputPlane;
    const uint kBegin = SPLIT ? split * pc.stepsPerSplit * TILE_K : 0;
    const uint kEnd = SPLIT ? min(reduction, kBegin + pc.stepsPerSplit * TILE_K) : reduction;

    // The input-tile column this invocation stages: steps localIndex / TILE_N + STAGE_STEPS r.
    const uint STAGE_STEPS = 256 / TILE_N;
    const uint stageColumn = localIndex % TILE_N;
    const uint stageStep = localIndex / TILE_N;
    const uint column = columnBase + stageColumn;
    const bool stageValid = column < columns;
    const uint stageBatch = stageValid ? column / positions : 0;
    const uint stagePosition = column - stageBatch * positions;
    const uint od = stagePosition / outputPlane;
    const uint oh = (stagePosition / pc.outputWidth) % pc.outputHeight;
    const uint ow = stagePosition % pc.outputWidth;
    const int3 origin = int3(od * pc.strideDepth, oh * pc.strideHeight, ow * pc.strideWidth) -
                        int3(pc.padDepth, pc.padHeight, pc.padWidth);
    const int3 inputExtent = int3(pc.inputDepth, pc.inputHeight, pc.inputWidth);
    const int originOffset = origin.x * inputPlane + origin.y * int(pc.inputWidth) + origin.z;
    const uint inputBatchOffset = stageBatch * pc.inputChannels * uint(inputVolume);

    float4 accumulator[4][2];
    for (uint i = 0; i < 4; ++i) {
        accumulator[i][0] = float4(0.0);
        accumulator[i][1] = float4(0.0);
    }

    for (uint kBase = kBegin; kBase < kEnd; kBase += TILE_K) {
        if (localIndex < TILE_K) {
            const uint tap = kBase + localIndex;
            if (tap < kEnd) {
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
            Tiles.setWeight(k, m, channel < pc.outputChannels && tap < kEnd ? weights[channel * reduction + tap] : 0.0);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint k = stageStep; k < TILE_K; k += STAGE_STEPS) {
            float value = 0.0;
            const int offset = tapOffset[k];
            if (stageValid && offset >= 0) {
                const int3 at = origin + tapKernel[k];
                if (all(at >= int3(0)) && all(at < inputExtent)) {
                    value = inputData[inputBatchOffset + uint(originOffset + offset)];
                }
            }
            Tiles.setInput(k, stageColumn, value);
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < TILE_K; ++k) {
            const float4 a = Tiles.weight4(k, ty);
            const float4 b0 = Tiles.input4(k, tx);
            const float4 b1 = Tiles.input4(k, TX + tx);
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
        const float channelBias = SPLIT ? 0.0 : bias[channel];
        for (uint part = 0; part < 2; ++part) {
            for (uint j = 0; j < 4; ++j) {
                const uint outputColumn = columnBase + part * (TILE_N / 2) + tx * 4 + j;
                if (outputColumn >= columns) continue;
                if (SPLIT) {
                    partialSums[(split * pc.outputChannels + channel) * columns + outputColumn] = accumulator[i][part][j];
                } else {
                    const uint batch = outputColumn / positions;
                    const uint position = outputColumn - batch * positions;
                    outputData[(batch * pc.outputChannels + channel) * positions + position] =
                        accumulator[i][part][j] + channelBias;
                }
            }
        }
    }
}

// The reduce pass of split-K, on the convolution's grid: workgroup z sums the
// splits of its share of the tile's elements, in split order, and adds the bias.
public void conv3dReduce(uint localIndex, uint3 groupId) {
    const uint positions = pc.outputDepth * pc.outputHeight * pc.outputWidth;
    const uint columns = pc.batchCount * positions;
    const uint tileElements = pc.tileM * pc.tileN;
    const uint share = (tileElements + pc.splitCount - 1) / pc.splitCount;
    const uint end = min(tileElements, (groupId.z + 1) * share);
    for (uint element = groupId.z * share + localIndex; element < end; element += 256) {
        const uint channel = groupId.y * pc.tileM + element / pc.tileN;
        const uint column = groupId.x * pc.tileN + element % pc.tileN;
        if (channel >= pc.outputChannels || column >= columns) continue;
        float sum = 0.0;
        for (uint s = 0; s < pc.splitCount; ++s) sum += partialSums[(s * pc.outputChannels + channel) * columns + column];
        const uint batch = column / positions;
        const uint position = column - batch * positions;
        outputData[(batch * pc.outputChannels + channel) * positions + position] = sum + bias[channel];
    }
}

// Direct 3D convolution for at most SMALL_CHANNELS output channels (e.g. the
// RGB output of a VAE), where a 64-channel tile would be mostly empty. One
// invocation per output column keeps all channels in registers and reads each
// input value once for all of them; the weights are the same for all
// invocations, so their loads are broadcast.
public static const uint SMALL_CHANNELS = 8;

public void conv3dSmall(uint column) {
    const uint outputPlane = pc.outputHeight * pc.outputWidth;
    const uint positions = pc.outputDepth * outputPlane;
    if (column >= pc.batchCount * positions) return;
    const uint batch = column / positions;
    const uint position = column - batch * positions;
    const int3 origin = int3(position / outputPlane * pc.strideDepth,
                             (position / pc.outputWidth) % pc.outputHeight * pc.strideHeight,
                             position % pc.outputWidth * pc.strideWidth) -
                        int3(pc.padDepth, pc.padHeight, pc.padWidth);
    const uint inputPlane = pc.inputHeight * pc.inputWidth;
    const uint inputVolume = pc.inputDepth * inputPlane;
    const uint kernelVolume = pc.kernelDepth * pc.kernelHeight * pc.kernelWidth;
    const uint reduction = pc.inputChannels * kernelVolume;

    float sums[SMALL_CHANNELS];
    for (uint m = 0; m < SMALL_CHANNELS; ++m) sums[m] = 0.0;
    for (uint kd = 0; kd < pc.kernelDepth; ++kd) {
        const int z = origin.x + int(kd);
        if (z < 0 || z >= int(pc.inputDepth)) continue;
        for (uint kh = 0; kh < pc.kernelHeight; ++kh) {
            const int y = origin.y + int(kh);
            if (y < 0 || y >= int(pc.inputHeight)) continue;
            for (uint kw = 0; kw < pc.kernelWidth; ++kw) {
                const int x = origin.z + int(kw);
                if (x < 0 || x >= int(pc.inputWidth)) continue;
                const uint tap = (kd * pc.kernelHeight + kh) * pc.kernelWidth + kw;
                uint inputIndex = batch * pc.inputChannels * inputVolume + uint(z) * inputPlane +
                                  uint(y) * pc.inputWidth + uint(x);
                for (uint c = 0; c < pc.inputChannels; ++c, inputIndex += inputVolume) {
                    const float value = inputData[inputIndex];
                    const uint weightIndex = c * kernelVolume + tap;
                    for (uint m = 0; m < SMALL_CHANNELS; ++m) {
                        if (m < pc.outputChannels) sums[m] += value * weights[m * reduction + weightIndex];
                    }
                }
            }
        }
    }
    for (uint m = 0; m < SMALL_CHANNELS; ++m) {
        if (m < pc.outputChannels) outputData[(batch * pc.outputChannels + m) * positions + position] = sums[m] + bias[m];
    }
}
