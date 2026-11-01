// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Draws one image onto another, keeping the target where it is not drawn on.
// The source is sampled bilinearly; pixel centres are mapped onto each other
// and reads are clamped to the edge.

struct PushConstants {
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
    uint mode; // 0: replace, 1: over (the source's alpha blends it over the target)
    uint fit;  // 0: stretch, 1: fit (whole source, aspect kept), 2: fill (covers the target, aspect kept)
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> sourceImage;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> targetImage;

float4 load(int2 p) {
    return sourceImage.Load(clamp(p, int2(0), int2(pc.srcWidth - 1, pc.srcHeight - 1)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    int2 pixel = int2(globalId.xy);
    if (pixel.x >= int(pc.dstWidth) || pixel.y >= int(pc.dstHeight)) {
        return;
    }

    // The rectangle of the target the source is drawn into.
    float2 src = float2(pc.srcWidth, pc.srcHeight);
    float2 dst = float2(pc.dstWidth, pc.dstHeight);
    float2 size = dst;
    if (pc.fit != 0u) {
        float scale = pc.fit == 1u ? min(dst.x / src.x, dst.y / src.y) : max(dst.x / src.x, dst.y / src.y);
        size = src * scale;
    }
    float2 offset = (dst - size) * 0.5;
    float2 uv = (float2(pixel) + 0.5 - offset) / size;
    if (any(uv < float2(0.0)) || any(uv >= float2(1.0))) {
        return; // outside the drawn rectangle: the target stays
    }

    float2 p = uv * src - 0.5;
    int2 p0 = int2(floor(p));
    float2 f = p - float2(p0);
    float4 top = lerp(load(p0), load(p0 + int2(1, 0)), f.x);
    float4 bottom = lerp(load(p0 + int2(0, 1)), load(p0 + int2(1, 1)), f.x);
    float4 color = lerp(top, bottom, f.y);

    if (pc.mode == 1u) {
        float4 below = targetImage.Load(pixel);
        color = float4(color.rgb * color.a + below.rgb * (1.0 - color.a), color.a + below.a * (1.0 - color.a));
    }
    targetImage.Store(pixel, color);
}
