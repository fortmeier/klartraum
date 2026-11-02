// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Differentiable shading of a visibility buffer (shade.comp.slang, shade_backward.comp.slang). The visibility pass
// decides which triangle covers a pixel; everything else is recomputed here from that triangle as differentiable
// functions: the projection of its vertices, perspective-correct barycentric coordinates at the pixel centre, the
// interpolated texture coordinate and a bilinear texture lookup. Slang's bwd_diff turns these into the gradients of
// a pixel's colour with respect to the triangle's vertex positions and the four texels it reads.
//
// Conventions: Vulkan clip space (y down, depth 0 to 1); pixel (x, y) has its centre at (x + 0.5, y + 0.5); the
// texture is textureWidth x textureHeight texels of four floats (rgb used), texel (i, j) centred at
// ((i + 0.5) / width, (j + 0.5) / height), wrapping in u and clamped in v.
module differentiable_shading;

public struct ShadingPushConstants {
    public uint width;
    public uint height;
    public uint textureWidth;
    public uint textureHeight;
};
[[vk::push_constant]] public ConstantBuffer<ShadingPushConstants> pc;

[[vk::binding(0, 0)]] public StructuredBuffer<float4> positions;  // xyz per vertex
[[vk::binding(1, 0)]] public StructuredBuffer<uint> triangles;    // three vertex indices per triangle
[[vk::binding(2, 0)]] public StructuredBuffer<float2> cornerUvs;  // three per triangle
[[vk::binding(3, 0)]] public StructuredBuffer<float4> texels;     // rgb, row-major
[[vk::binding(4, 0)]] public StructuredBuffer<float4x4> cameras;  // clip from world, per view
[[vk::binding(5, 0)]] public StructuredBuffer<uint> viewIndex;    // the drawn view
[[vk::binding(6, 0)]] public StructuredBuffer<uint> visibility;   // triangle + 1 per pixel, 0: background

// The texture coordinate at the pixel centre `pixel` of the triangle with vertices p0, p1, p2 and corner
// coordinates uv0, uv1, uv2 under `camera`: screen-space barycentrics of the projected corners, corrected for
// perspective by the clip w of each corner.
[Differentiable]
public float2 textureCoordinate(float3 p0, float3 p1, float3 p2, float2 uv0, float2 uv1, float2 uv2,
                                no_diff float4x4 camera, no_diff float2 pixel, no_diff float2 size) {
    const float4 c0 = mul(camera, float4(p0, 1.0)), c1 = mul(camera, float4(p1, 1.0)), c2 = mul(camera, float4(p2, 1.0));
    const float2 s0 = (c0.xy / c0.w * 0.5 + 0.5) * size;
    const float2 s1 = (c1.xy / c1.w * 0.5 + 0.5) * size;
    const float2 s2 = (c2.xy / c2.w * 0.5 + 0.5) * size;
    const float area = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x);
    const float l0 = ((s1.x - pixel.x) * (s2.y - pixel.y) - (s1.y - pixel.y) * (s2.x - pixel.x)) / area;
    const float l1 = ((s2.x - pixel.x) * (s0.y - pixel.y) - (s2.y - pixel.y) * (s0.x - pixel.x)) / area;
    const float l2 = 1.0 - l0 - l1;
    const float q0 = l0 / c0.w, q1 = l1 / c1.w, q2 = l2 / c2.w;
    return (q0 * uv0 + q1 * uv1 + q2 * uv2) / (q0 + q1 + q2);
}

// The texels and weights of a bilinear lookup at texture coordinate `uv`.
public struct BilinearTaps {
    public uint index[4];   // texel indices: (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
    public float2 fraction; // position between the texel centres
};

public BilinearTaps bilinearTaps(float2 uv) {
    const float x = uv.x * float(pc.textureWidth) - 0.5, y = uv.y * float(pc.textureHeight) - 0.5;
    const float fx = floor(x), fy = floor(y);
    BilinearTaps taps;
    taps.fraction = float2(x - fx, y - fy);
    const int w = int(pc.textureWidth), h = int(pc.textureHeight);
    const int i0 = ((int(fx) % w) + w) % w, i1 = (i0 + 1) % w;
    const int j0 = clamp(int(fy), 0, h - 1), j1 = clamp(int(fy) + 1, 0, h - 1);
    taps.index[0] = uint(j0 * w + i0);
    taps.index[1] = uint(j0 * w + i1);
    taps.index[2] = uint(j1 * w + i0);
    taps.index[3] = uint(j1 * w + i1);
    return taps;
}

// The bilinear blend of four texel colours at `fraction` (texel colours are constants here; their gradients are
// the blend weights times the output gradient).
[Differentiable]
public float3 bilinear(float2 fraction, no_diff float3 t00, no_diff float3 t10, no_diff float3 t01, no_diff float3 t11) {
    return lerp(lerp(t00, t10, fraction.x), lerp(t01, t11, fraction.x), fraction.y);
}

// The blend weights of the four taps.
public float4 bilinearWeights(float2 fraction) {
    return float4((1.0 - fraction.x) * (1.0 - fraction.y), fraction.x * (1.0 - fraction.y),
                  (1.0 - fraction.x) * fraction.y, fraction.x * fraction.y);
}

// What the shading of one covered pixel reads.
public struct PixelSurface {
    public uint triangle;
    public uint vertex[3];
    public float3 corner[3];
    public float2 cornerUv[3];
    public float4x4 camera;
    public float2 pixel;
};

// The surface of pixel (x, y), or false for background.
public bool pixelSurface(uint x, uint y, out PixelSurface surface) {
    const uint id = visibility[y * pc.width + x];
    surface = {};
    if (id == 0) return false;
    surface.triangle = id - 1;
    for (uint c = 0; c < 3; ++c) {
        surface.vertex[c] = triangles[3 * surface.triangle + c];
        surface.corner[c] = positions[surface.vertex[c]].xyz;
        surface.cornerUv[c] = cornerUvs[3 * surface.triangle + c];
    }
    surface.camera = cameras[viewIndex[0]];
    surface.pixel = float2(x, y) + 0.5;
    return true;
}

// The colour of a covered pixel.
public float3 shade(PixelSurface s) {
    const float2 uv = textureCoordinate(s.corner[0], s.corner[1], s.corner[2], s.cornerUv[0], s.cornerUv[1],
                                        s.cornerUv[2], s.camera, s.pixel, float2(pc.width, pc.height));
    const BilinearTaps taps = bilinearTaps(uv);
    return bilinear(taps.fraction, texels[taps.index[0]].rgb, texels[taps.index[1]].rgb, texels[taps.index[2]].rgb,
                    texels[taps.index[3]].rgb);
}
