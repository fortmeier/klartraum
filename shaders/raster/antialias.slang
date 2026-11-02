// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Analytic antialiasing of silhouettes (after Laine et al. 2020, nvdiffrast), the source of gradients that move
// silhouettes. For two neighbouring pixels showing different surfaces, the nearer pixel's triangle is the front
// triangle; if the segment between the pixel centres crosses one of its edges, and the other pixel's triangle does
// not share that edge (else the edge is interior to the surface), the edge is a silhouette. Where it crosses at
// parameter t from the front pixel's centre:
//   t > 0.5: the front surface covers part of the other pixel, which blends towards the front colour by t - 0.5;
//   t < 0.5: the other surface covers part of the front pixel, which blends towards the other colour by 0.5 - t.
// t is a differentiable function of the edge's projected endpoints, so colour and coverage errors at a silhouette
// move its vertices. Colours are premultiplied (rgb times coverage); background is zero.
module antialias;

import differentiable_shading;

// The four neighbours of a pixel.
public static const int2 NEIGHBOURS[4] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1) };

// The parameter along the segment from pixel centre `from` to pixel centre `to` where it crosses the line through
// the projected points a and b.
[Differentiable]
public float crossing(float3 a, float3 b, no_diff float4x4 camera, no_diff float2 from, no_diff float2 to,
                      no_diff float2 size) {
    const float4 ca = mul(camera, float4(a, 1.0)), cb = mul(camera, float4(b, 1.0));
    const float2 sa = (ca.xy / ca.w * 0.5 + 0.5) * size, sb = (cb.xy / cb.w * 0.5 + 0.5) * size;
    const float2 edge = sb - sa, segment = to - from;
    // from + t segment lies on the line: cross(edge, from + t segment - sa) = 0
    const float numerator = edge.x * (sa.y - from.y) - edge.y * (sa.x - from.x);
    const float denominator = edge.x * segment.y - edge.y * segment.x;
    return numerator / denominator;
}

// The NDC depth of the triangle at a pixel centre (for choosing the front triangle).
float pixelDepth(PixelSurface s) {
    const float2 size = float2(pc.width, pc.height);
    float3 screen[3];
    for (uint c = 0; c < 3; ++c) {
        const float4 clip = mul(s.camera, float4(s.corner[c], 1.0));
        screen[c] = float3((clip.xy / clip.w * 0.5 + 0.5) * size, clip.z / clip.w);
    }
    const float area = (screen[1].x - screen[0].x) * (screen[2].y - screen[0].y) -
                       (screen[1].y - screen[0].y) * (screen[2].x - screen[0].x);
    const float l0 = ((screen[1].x - s.pixel.x) * (screen[2].y - s.pixel.y) -
                      (screen[1].y - s.pixel.y) * (screen[2].x - s.pixel.x)) / area;
    const float l1 = ((screen[2].x - s.pixel.x) * (screen[0].y - s.pixel.y) -
                      (screen[2].y - s.pixel.y) * (screen[0].x - s.pixel.x)) / area;
    return l0 * screen[0].z + l1 * screen[1].z + (1.0 - l0 - l1) * screen[2].z;
}

// A silhouette between a pixel and one of its neighbours.
public struct Silhouette {
    public bool found;
    public bool frontIsSelf; // whether this pixel holds the front triangle
    public float t;          // crossing parameter from the front pixel's centre
    public float3 a, b;      // the silhouette edge's endpoints
    public uint vertexA, vertexB;
    public float4x4 camera;
    public float2 frontPixel, otherPixel;
};

// The silhouette between pixel `self` and its neighbour `other`, if any.
public Silhouette findSilhouette(uint2 self, uint2 other) {
    Silhouette result = {};
    PixelSurface selfSurface, otherSurface;
    const bool selfCovered = pixelSurface(self.x, self.y, selfSurface);
    const bool otherCovered = pixelSurface(other.x, other.y, otherSurface);
    if (!selfCovered && !otherCovered) return result;
    if (selfCovered && otherCovered && selfSurface.triangle == otherSurface.triangle) return result;
    result.frontIsSelf = selfCovered && (!otherCovered || pixelDepth(selfSurface) < pixelDepth(otherSurface));
    const PixelSurface front = result.frontIsSelf ? selfSurface : otherSurface;
    const bool backCovered = result.frontIsSelf ? otherCovered : selfCovered;
    const PixelSurface back = result.frontIsSelf ? otherSurface : selfSurface;
    result.frontPixel = front.pixel;
    result.otherPixel = float2(result.frontIsSelf ? other : self) + 0.5;
    result.camera = front.camera;
    const float2 size = float2(pc.width, pc.height);
    float best = 2.0;
    for (uint e = 0; e < 3; ++e) {
        const uint i = e, j = (e + 1) % 3;
        // An edge the other pixel's triangle shares is interior to the surface.
        if (backCovered) {
            bool sharesI = false, sharesJ = false;
            for (uint c = 0; c < 3; ++c) {
                sharesI = sharesI || back.vertex[c] == front.vertex[i];
                sharesJ = sharesJ || back.vertex[c] == front.vertex[j];
            }
            if (sharesI && sharesJ) continue;
        }
        const float t = crossing(front.corner[i], front.corner[j], front.camera, result.frontPixel, result.otherPixel,
                                 size);
        if (t > 0.0 && t < 1.0 && t < best) {
            best = t;
            result.found = true;
            result.t = t;
            result.a = front.corner[i];
            result.b = front.corner[j];
            result.vertexA = front.vertex[i];
            result.vertexB = front.vertex[j];
        }
    }
    return result;
}

// How much this pixel blends towards the neighbour's colour across the silhouette (0 if it does not).
public float blendWeight(Silhouette s) {
    if (!s.found) return 0.0;
    return s.frontIsSelf ? max(0.5 - s.t, 0.0) : max(s.t - 0.5, 0.0);
}
