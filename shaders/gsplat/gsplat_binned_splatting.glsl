// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Body of gsplat_binned_splatting.comp and gsplat_binned_splatting_depth.comp.
// OUTPUT_DEPTH enables configurable accumulated-opacity or nearest-opaque-splat
// depth. SCENE_DEPTH tests/updates an attachment snapshot, and DEPTH_ONLY omits
// RGB output. Exported depth is +inf where no selected surface exists.

#include "gsplat_types.glsl"

// Tile dimensions — tunable for different hardware.
// Larger tiles: fewer barrier() pairs per Gaussian, but more registers per thread.
// Default 8×8 = 64 threads balances occupancy and batch throughput.
#ifndef TILE_X
#define TILE_X 8
#endif
#ifndef TILE_Y
#define TILE_Y 8
#endif
#define WORKGROUP_SIZE (TILE_X * TILE_Y)
layout(local_size_x = TILE_X, local_size_y = TILE_Y) in;

// Sorted Gaussian2D SoA inputs
layout(scalar, set = 0, binding = 0) readonly buffer SortedPos2D      { vec2  sortedPos2D[];      };
layout(scalar, set = 0, binding = 1) readonly buffer SortedCovInv     { vec4  sortedCovInv[];     }; // (m00,m10,m01,m11)
layout(scalar, set = 0, binding = 2) readonly buffer SortedColorAlpha { vec4  sortedColorAlpha[]; }; // (r,g,b,alpha)

layout(scalar, set = 0, binding = 3) readonly buffer CountBuf  { uint numberTotalGaussians; };
layout(scalar, set = 0, binding = 4) readonly buffer BoundsBuf { StartAndEnd startAndEnd[];  };

#ifndef DEPTH_ONLY
layout(binding = 5, rgba8) uniform image2D outputImage;
#endif

#ifdef OUTPUT_DEPTH
layout(scalar, set = 0, binding = 6) readonly buffer SortedZ { float sortedZ[]; };
layout(scalar, set = 0, binding = 7) writeonly buffer DepthOutput { float depthOutput[]; };
#endif

#ifdef SCENE_DEPTH
layout(scalar, set = 0, binding = 8) buffer SceneDepth { float sceneDepth[]; };
#endif

layout(push_constant) uniform PushConstants {
    uint  numElements;
    uint  gridSize;
    uint  binIndexX;
    uint  binIndexY;
    float screenWidth;
    float screenHeight;
    uint depthMode;
    uint depthFlags;
    float depthThreshold;
    float depthAlphaCull;
} pushConstants;

#ifdef OUTPUT_DEPTH
void selectDepth(float a, float transmittance, float z, inout float depth) {
    if (pushConstants.depthMode == 2u) {
        if (a >= pushConstants.depthThreshold) depth = min(depth, z);
    } else {
        float threshold = 1.0 - pushConstants.depthThreshold;
        if (transmittance > threshold && transmittance * (1.0 - a) <= threshold) depth = z;
    }
}
#endif

// Shared memory batch — one slot per thread in the workgroup.
struct GaussBatch {
    vec2 pos2D;
    vec4 covInv;
    vec4 colorAlpha;
};
shared GaussBatch sharedGauss[WORKGROUP_SIZE];
#ifdef OUTPUT_DEPTH
// Keep the RGB batch stride unchanged: adding z to its vec4-aligned struct
// pads every entry and can increase shared-memory bank conflicts during loading.
shared float sharedZ[WORKGROUP_SIZE];
#endif

// Number of threads that no longer need further batches, double-buffered by
// batch parity: one slot is counted and read while the other is reset.
shared uint sharedDoneCount[2];

void main() {
    // The bin covers the pixels [binBegin, binEnd), i.e. exactly the pixels
    // inside the rectangle [k * cell, (k + 1) * cell] that the binning passes
    // test Gaussians against, so every pixel is shaded with its own bin's list.
    const vec2 cell = vec2(pushConstants.screenWidth, pushConstants.screenHeight)
                    / float(pushConstants.gridSize);
    const uvec2 binIndex = uvec2(pushConstants.binIndexX, pushConstants.binIndexY);
    const ivec2 binBegin = ivec2(ceil(vec2(binIndex) * cell));
    const ivec2 binEnd   = ivec2(ceil(vec2(binIndex + 1u) * cell));

    const ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy) + binBegin;
    // Threads outside the bin still load their share of every batch into
    // shared memory (the other threads read those slots); they only skip
    // shading and the image write.
    const bool insideBin = all(lessThan(pixelCoord, binEnd));
    uint pixelIndex = uint(pixelCoord.y) * uint(pushConstants.screenWidth) + uint(pixelCoord.x);
#ifdef SCENE_DEPTH
    float sceneLimit = insideBin ? sceneDepth[pixelIndex] : 1.0;
#endif
    const vec2 pixelCoordF = vec2(pixelCoord);

    const StartAndEnd range = startAndEnd[
        pushConstants.binIndexY * pushConstants.gridSize + pushConstants.binIndexX];
    const uint start    = range.start;
    const uint end      = range.end;
    const uint stepsize = WORKGROUP_SIZE;
    const uint preend   = start + ((end - start) / stepsize) * stepsize;

    vec4  finalColor    = vec4(0.0, 0.0, 0.0, 1.0);
    float accum_opacity = 1.0;
    float colorTransmittance = 1.0;
#ifdef OUTPUT_DEPTH
    float depth = uintBitsToFloat(0x7F800000u); // +inf until a surface is selected
#endif

    // A pixel stops accumulating once it is (nearly) opaque, but keeps loading
    // its slot of each batch until the whole workgroup is done: the loop and
    // its barriers must stay in uniform control flow.
    bool accumulating = insideBin;
    bool colorAccumulating = insideBin;
    if (gl_LocalInvocationIndex == 0u) {
        sharedDoneCount[0] = 0u;
        sharedDoneCount[1] = 0u;
    }

    uint batch = 0u;
    for (uint i = start; i < preend; i += stepsize, batch++) {
        uint li = i + gl_LocalInvocationIndex;
        sharedGauss[gl_LocalInvocationIndex].pos2D      = sortedPos2D[li];
        sharedGauss[gl_LocalInvocationIndex].covInv     = sortedCovInv[li];
        sharedGauss[gl_LocalInvocationIndex].colorAlpha = sortedColorAlpha[li];
#ifdef OUTPUT_DEPTH
        sharedZ[gl_LocalInvocationIndex] = sortedZ[li];
#endif
        barrier();

        // The other slot was last read before the barrier above and is next
        // counted after the next batch's first barrier.
        if (gl_LocalInvocationIndex == 0u) {
            sharedDoneCount[(batch + 1u) & 1u] = 0u;
        }

        for (uint j = 0u; accumulating && j < stepsize; j++) {
            vec2 mu   = sharedGauss[j].pos2D;
            vec4 ci   = sharedGauss[j].covInv;
            vec4 rgba = sharedGauss[j].colorAlpha;
#ifdef DEPTH_ONLY
            if (rgba.a < pushConstants.depthAlphaCull) continue;
#endif

            // Inline Gaussian: exp(-0.5 * diff^T * covInv * diff)
            // Avoids mat2 construction + matrix-vector multiply.
            // covInv is symmetric: ci.y == ci.z, so cross term = 2*ci.y*dx*dy.
            vec2  d  = pixelCoordF - mu;
            float q  = ci.x * d.x * d.x + 2.0 * ci.y * d.x * d.y + ci.w * d.y * d.y;
            float a  = exp(-0.5 * q) * rgba.a;

#if defined(SCENE_DEPTH) || defined(DEPTH_ONLY)
            float z = sharedZ[j];
#ifdef SCENE_DEPTH
            if (z < 0.0 || z > 1.0 || ((pushConstants.depthFlags & 1u) != 0u && z > sceneLimit)) continue;
#else
            if (z < 0.0 || z > 1.0) continue;
#endif
#endif

            if (colorAccumulating) finalColor += vec4(rgba.rgb * (a * accum_opacity), 0.0);
#ifdef OUTPUT_DEPTH
            selectDepth(a, accum_opacity, sharedZ[j], depth);
#endif
            accum_opacity *= (1.0 - a);
            if (colorAccumulating) colorTransmittance = accum_opacity;
        }
        if (accum_opacity <= 1.0 - 0.9999) colorAccumulating = false;
        accumulating = colorAccumulating;
#ifdef OUTPUT_DEPTH
        accumulating = insideBin && (colorAccumulating || ((pushConstants.depthFlags & 6u) != 0u && isinf(depth)));
#endif
        if (!accumulating) atomicAdd(sharedDoneCount[batch & 1u], 1u);
        barrier();

        if (sharedDoneCount[batch & 1u] == uint(WORKGROUP_SIZE)) break;
    }

    // Tail loop for the remainder (< stepsize elements)
    for (uint i = preend; accumulating && i < end; i++) {
        vec2 mu   = sortedPos2D[i];
        vec4 ci   = sortedCovInv[i];
        vec4 rgba = sortedColorAlpha[i];
#ifdef DEPTH_ONLY
        if (rgba.a < pushConstants.depthAlphaCull) continue;
#endif

        vec2  d = pixelCoordF - mu;
        float q = ci.x * d.x * d.x + 2.0 * ci.y * d.x * d.y + ci.w * d.y * d.y;
        float a = exp(-0.5 * q) * rgba.a;

#if defined(SCENE_DEPTH) || defined(DEPTH_ONLY)
        float z = sortedZ[i];
#ifdef SCENE_DEPTH
        if (z < 0.0 || z > 1.0 || ((pushConstants.depthFlags & 1u) != 0u && z > sceneLimit)) continue;
#else
        if (z < 0.0 || z > 1.0) continue;
#endif
#endif

        if (colorAccumulating) finalColor += vec4(rgba.rgb * (a * accum_opacity), 0.0);
#ifdef OUTPUT_DEPTH
        selectDepth(a, accum_opacity, sortedZ[i], depth);
#endif
        accum_opacity *= (1.0 - a);
        if (colorAccumulating) colorTransmittance = accum_opacity;

        if (accum_opacity <= 1.0 - 0.9999) colorAccumulating = false;
#ifdef OUTPUT_DEPTH
        if (!colorAccumulating && ((pushConstants.depthFlags & 6u) == 0u || !isinf(depth))) break;
#else
        if (!colorAccumulating) break;
#endif
    }

    // finalColor.rgb holds the premultiplied "over" radiance Σ cᵢ·aᵢ·Tᵢ of the
    // splats; accum_opacity is the remaining transmittance T. The splats are
    // composited over what the image holds: rgb + T·background, and the
    // coverage 1 - T·(1 - background alpha).
    if (insideBin) {
#ifndef DEPTH_ONLY
        vec4 background = imageLoad(outputImage, pixelCoord);
        vec4 outputColor = vec4(clamp(finalColor.rgb + colorTransmittance * background.rgb, vec3(0.0), vec3(1.0)),
                                clamp(1.0 - colorTransmittance * (1.0 - background.a), 0.0, 1.0));
        imageStore(outputImage, pixelCoord, outputColor);
#endif
#ifdef OUTPUT_DEPTH
        if ((pushConstants.depthFlags & 4u) != 0u) depthOutput[pixelIndex] = depth;
#endif
#ifdef SCENE_DEPTH
        if ((pushConstants.depthFlags & 2u) != 0u && !isinf(depth)) sceneDepth[pixelIndex] = depth;
#endif
    }
}
