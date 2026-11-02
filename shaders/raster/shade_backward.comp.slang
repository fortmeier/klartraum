// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The masked colour loss of the shaded visibility buffer and its gradients (see differentiable_shading.slang):
//   loss = sum over pixels covered by the mesh and inside the target mask of |colour - target|^2 / (width x height).
// Each pixel recomputes its colour, then bwd_diff of the bilinear lookup and of the texture-coordinate function
// carries the colour gradient back to the texture coordinate and the triangle's vertex positions. The texels get
// their bilinear weights times the colour gradient. Gradients are added atomically into buffers the graph clears
// before this pass.
import differentiable_shading;
import subgroup;

[[vk::binding(7, 0)]] StructuredBuffer<float4> targets;            // rgb, mask; per view and pixel
[[vk::binding(8, 0)]] RWStructuredBuffer<float> vertexGradients;   // four per vertex (xyz used)
[[vk::binding(9, 0)]] RWStructuredBuffer<float> textureGradients;  // four per texel (rgb used)
[[vk::binding(10, 0)]] RWStructuredBuffer<float> loss;             // one element

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    float pixelLoss = 0.0;
    PixelSurface s;
    const uint pixelCount = pc.width * pc.height;
    if (globalId.x < pc.width && globalId.y < pc.height && pixelSurface(globalId.x, globalId.y, s)) {
        const float4 target = targets[viewIndex[0] * pixelCount + globalId.y * pc.width + globalId.x];
        if (target.a >= 0.5) {
            const float3 difference = shade(s) - target.rgb;
            pixelLoss = dot(difference, difference) / float(pixelCount);
            backpropagateColor(s, 2.0 * difference / float(pixelCount), vertexGradients, textureGradients);
        }
    }
    const float groupLoss = WaveActiveSum(pixelLoss);
    if (WaveIsFirstLane()) InterlockedAdd(loss[0], groupLoss);
}
