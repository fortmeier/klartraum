// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Body of the binned splatting shaders (gsplat_binned_splatting*.comp.slang,
// gsplat_binned_depth_only*.comp.slang). The variant's outputDepth enables
// configurable accumulated-opacity or nearest-opaque-splat depth, sceneDepth
// tests/updates an attachment snapshot, and depthOnly omits RGB output.
// Exported depth is +inf where no selected surface exists. Resources a variant
// does not use are not part of its module.
module gsplat_binned_splatting;

__exported import gsplat_types;
__exported import scalar_buffer;

// Tile dimensions — tunable for different hardware.
// Larger tiles: fewer barrier pairs per Gaussian, but more registers per thread.
// Default 8×8 = 64 threads balances occupancy and batch throughput.
public static const uint TILE_X = 8;
public static const uint TILE_Y = 8;
static const uint WORKGROUP_SIZE = TILE_X * TILE_Y;

// The outputs of a binned splatting shader.
public interface ISplatVariant {
    static const bool outputDepth;
    static const bool depthOnly;
    static const bool sceneDepth;
}

// Sorted Gaussian2D SoA inputs
[[vk::binding(0, 0)]] ScalarBuffer<float2> sortedPos2D;
[[vk::binding(1, 0)]] ScalarBuffer<float4> sortedCovInv;     // (m00,m10,m01,m11)
[[vk::binding(2, 0)]] ScalarBuffer<float4> sortedColorAlpha; // (r,g,b,alpha)

// One uint: the number of sorted Gaussians.
[[vk::binding(3, 0)]] ScalarBuffer<uint> numberTotalGaussians;
[[vk::binding(4, 0)]] ScalarBuffer<StartAndEnd> startAndEnd;

// Unless depthOnly.
[[vk::binding(5, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outputImage;

// With outputDepth.
[[vk::binding(6, 0)]] ScalarBuffer<float> sortedZ;
[[vk::binding(7, 0)]] RWScalarBuffer<float> depthOutput;

// With sceneDepth.
[[vk::binding(8, 0)]] RWScalarBuffer<float> sceneDepth;

struct PushConstants {
    uint numElements;
    uint gridSize;
    uint binIndexX;
    uint binIndexY;
    float screenWidth;
    float screenHeight;
    uint depthMode;
    uint depthFlags;
    float depthThreshold;
    float depthAlphaCull;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pushConstants;

void selectDepth(float a, float transmittance, float z, inout float depth) {
    if (pushConstants.depthMode == 2u) {
        if (a >= pushConstants.depthThreshold) depth = min(depth, z);
    } else {
        float threshold = 1.0 - pushConstants.depthThreshold;
        if (transmittance > threshold && transmittance * (1.0 - a) <= threshold) depth = z;
    }
}

// Shared memory batch — one slot per thread in the workgroup.
struct GaussBatch {
    float2 pos2D;
    float4 covInv;
    float4 colorAlpha;
};
// The batch of Gaussians the workgroup currently shades.
groupshared GaussBatch sharedGauss[WORKGROUP_SIZE];
// The depths of the batch, with outputDepth. Kept apart from GaussBatch so the
// RGB batch stride stays unchanged: adding z to its float4-aligned struct
// pads every entry and can increase shared-memory bank conflicts during loading.
groupshared float sharedZ[WORKGROUP_SIZE];

// Number of threads that no longer need further batches, double-buffered by
// batch parity: one slot is counted and read while the other is reset.
groupshared uint sharedDoneCount[2];

// Shades the pixel of invocation `globalId` in the bin of the push constants.
// Runs in a TILE_X x TILE_Y workgroup; `localIndex` is the invocation's index in it.
public void binnedSplatting<Variant : ISplatVariant>(uint3 globalId, uint localIndex) {
    // The bin covers the pixels [binBegin, binEnd), i.e. exactly the pixels
    // inside the rectangle [k * cell, (k + 1) * cell] that the binning passes
    // test Gaussians against, so every pixel is shaded with its own bin's list.
    const float2 cell = float2(pushConstants.screenWidth, pushConstants.screenHeight) / float(pushConstants.gridSize);
    const uint2 binIndex = uint2(pushConstants.binIndexX, pushConstants.binIndexY);
    const int2 binBegin = int2(ceil(float2(binIndex) * cell));
    const int2 binEnd = int2(ceil(float2(binIndex + 1u) * cell));

    const int2 pixelCoord = int2(globalId.xy) + binBegin;
    // Threads outside the bin still load their share of every batch into
    // shared memory (the other threads read those slots); they only skip
    // shading and the image write.
    const bool insideBin = all(pixelCoord < binEnd);
    uint pixelIndex = uint(pixelCoord.y) * uint(pushConstants.screenWidth) + uint(pixelCoord.x);
    float sceneLimit = 1.0;
    if (Variant.sceneDepth) sceneLimit = insideBin ? sceneDepth[pixelIndex] : 1.0;
    const float2 pixelCoordF = float2(pixelCoord);

    const StartAndEnd range = startAndEnd[pushConstants.binIndexY * pushConstants.gridSize + pushConstants.binIndexX];
    const uint start = range.start;
    const uint end = range.end;
    const uint stepsize = WORKGROUP_SIZE;
    const uint preend = start + ((end - start) / stepsize) * stepsize;

    float4 finalColor = float4(0.0, 0.0, 0.0, 1.0);
    float accum_opacity = 1.0;
    float colorTransmittance = 1.0;
    float depth = asfloat(0x7F800000u); // +inf until a surface is selected

    // A pixel stops accumulating once it is (nearly) opaque, but keeps loading
    // its slot of each batch until the whole workgroup is done: the loop and
    // its barriers must stay in uniform control flow.
    bool accumulating = insideBin;
    bool colorAccumulating = insideBin;
    if (localIndex == 0u) {
        sharedDoneCount[0] = 0u;
        sharedDoneCount[1] = 0u;
    }

    uint batch = 0u;
    for (uint i = start; i < preend; i += stepsize, batch++) {
        uint li = i + localIndex;
        sharedGauss[localIndex].pos2D = sortedPos2D[li];
        sharedGauss[localIndex].covInv = sortedCovInv[li];
        sharedGauss[localIndex].colorAlpha = sortedColorAlpha[li];
        if (Variant.outputDepth) sharedZ[localIndex] = sortedZ[li];
        GroupMemoryBarrierWithGroupSync();

        // The other slot was last read before the barrier above and is next
        // counted after the next batch's first barrier.
        if (localIndex == 0u) {
            sharedDoneCount[(batch + 1u) & 1u] = 0u;
        }

        for (uint j = 0u; accumulating && j < stepsize; j++) {
            float2 mu = sharedGauss[j].pos2D;
            float4 ci = sharedGauss[j].covInv;
            float4 rgba = sharedGauss[j].colorAlpha;
            if (Variant.depthOnly && rgba.a < pushConstants.depthAlphaCull) continue;

            // Inline Gaussian: exp(-0.5 * diff^T * covInv * diff)
            // Avoids matrix construction + matrix-vector multiply.
            // covInv is symmetric: ci.y == ci.z, so cross term = 2*ci.y*dx*dy.
            float2 d = pixelCoordF - mu;
            float q = ci.x * d.x * d.x + 2.0 * ci.y * d.x * d.y + ci.w * d.y * d.y;
            float a = exp(-0.5 * q) * rgba.a;

            if (Variant.sceneDepth || Variant.depthOnly) {
                float z = sharedZ[j];
                if (Variant.sceneDepth) {
                    if (z < 0.0 || z > 1.0 || ((pushConstants.depthFlags & 1u) != 0u && z > sceneLimit)) continue;
                } else {
                    if (z < 0.0 || z > 1.0) continue;
                }
            }

            if (colorAccumulating) finalColor += float4(rgba.rgb * (a * accum_opacity), 0.0);
            if (Variant.outputDepth) selectDepth(a, accum_opacity, sharedZ[j], depth);
            accum_opacity *= (1.0 - a);
            if (colorAccumulating) colorTransmittance = accum_opacity;
        }
        if (accum_opacity <= 1.0 - 0.9999) colorAccumulating = false;
        accumulating = colorAccumulating;
        if (Variant.outputDepth) {
            accumulating = insideBin && (colorAccumulating || ((pushConstants.depthFlags & 6u) != 0u && isinf(depth)));
        }
        if (!accumulating) InterlockedAdd(sharedDoneCount[batch & 1u], 1u);
        GroupMemoryBarrierWithGroupSync();

        if (sharedDoneCount[batch & 1u] == WORKGROUP_SIZE) break;
    }

    // Tail loop for the remainder (< stepsize elements)
    for (uint i = preend; accumulating && i < end; i++) {
        float2 mu = sortedPos2D[i];
        float4 ci = sortedCovInv[i];
        float4 rgba = sortedColorAlpha[i];
        if (Variant.depthOnly && rgba.a < pushConstants.depthAlphaCull) continue;

        float2 d = pixelCoordF - mu;
        float q = ci.x * d.x * d.x + 2.0 * ci.y * d.x * d.y + ci.w * d.y * d.y;
        float a = exp(-0.5 * q) * rgba.a;

        if (Variant.sceneDepth || Variant.depthOnly) {
            float z = sortedZ[i];
            if (Variant.sceneDepth) {
                if (z < 0.0 || z > 1.0 || ((pushConstants.depthFlags & 1u) != 0u && z > sceneLimit)) continue;
            } else {
                if (z < 0.0 || z > 1.0) continue;
            }
        }

        if (colorAccumulating) finalColor += float4(rgba.rgb * (a * accum_opacity), 0.0);
        if (Variant.outputDepth) selectDepth(a, accum_opacity, sortedZ[i], depth);
        accum_opacity *= (1.0 - a);
        if (colorAccumulating) colorTransmittance = accum_opacity;

        if (accum_opacity <= 1.0 - 0.9999) colorAccumulating = false;
        if (Variant.outputDepth) {
            if (!colorAccumulating && ((pushConstants.depthFlags & 6u) == 0u || !isinf(depth))) break;
        } else {
            if (!colorAccumulating) break;
        }
    }

    // finalColor.rgb holds the premultiplied "over" radiance Σ cᵢ·aᵢ·Tᵢ of the
    // splats; accum_opacity is the remaining transmittance T. The splats are
    // composited over what the image holds: rgb + T·background, and the
    // coverage 1 - T·(1 - background alpha).
    if (insideBin) {
        if (!Variant.depthOnly) {
            float4 background = outputImage.Load(pixelCoord);
            float4 outputColor =
                float4(clamp(finalColor.rgb + colorTransmittance * background.rgb, float3(0.0), float3(1.0)),
                       clamp(1.0 - colorTransmittance * (1.0 - background.a), 0.0, 1.0));
            outputImage.Store(pixelCoord, outputColor);
        }
        if (Variant.outputDepth) {
            if ((pushConstants.depthFlags & 4u) != 0u) depthOutput[pixelIndex] = depth;
        }
        if (Variant.sceneDepth) {
            if ((pushConstants.depthFlags & 2u) != 0u && !isinf(depth)) sceneDepth[pixelIndex] = depth;
        }
    }
}
