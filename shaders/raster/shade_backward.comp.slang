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
            const float2 size = float2(pc.width, pc.height);
            const float2 uv = textureCoordinate(s.corner[0], s.corner[1], s.corner[2], s.cornerUv[0], s.cornerUv[1],
                                                s.cornerUv[2], s.camera, s.pixel, size);
            const BilinearTaps taps = bilinearTaps(uv);
            float3 t[4];
            for (uint k = 0; k < 4; ++k) t[k] = texels[taps.index[k]].rgb;
            const float3 difference = bilinear(taps.fraction, t[0], t[1], t[2], t[3]) - target.rgb;
            pixelLoss = dot(difference, difference) / float(pixelCount);
            const float3 colorGradient = 2.0 * difference / float(pixelCount);

            const float4 weights = bilinearWeights(taps.fraction);
            for (uint k = 0; k < 4; ++k) {
                for (uint c = 0; c < 3; ++c) InterlockedAdd(textureGradients[4 * taps.index[k] + c], weights[k] * colorGradient[c]);
            }

            var fraction = diffPair(taps.fraction);
            bwd_diff(bilinear)(fraction, t[0], t[1], t[2], t[3], colorGradient);
            // fraction = uv x texture size - 0.5 - floor(...): the floor is constant almost everywhere.
            const float2 uvGradient = fraction.d * float2(pc.textureWidth, pc.textureHeight);

            var p0 = diffPair(s.corner[0]);
            var p1 = diffPair(s.corner[1]);
            var p2 = diffPair(s.corner[2]);
            var uv0 = diffPair(s.cornerUv[0]);
            var uv1 = diffPair(s.cornerUv[1]);
            var uv2 = diffPair(s.cornerUv[2]);
            bwd_diff(textureCoordinate)(p0, p1, p2, uv0, uv1, uv2, s.camera, s.pixel, size, uvGradient);
            const float3 cornerGradients[3] = { p0.d, p1.d, p2.d };
            for (uint corner = 0; corner < 3; ++corner) {
                for (uint c = 0; c < 3; ++c) InterlockedAdd(vertexGradients[4 * s.vertex[corner] + c], cornerGradients[corner][c]);
            }
        }
    }
    const float groupLoss = WaveActiveSum(pixelLoss);
    if (WaveIsFirstLane()) InterlockedAdd(loss[0], groupLoss);
}
