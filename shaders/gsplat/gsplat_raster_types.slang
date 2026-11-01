// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The per-splat record of the raster backend.
module gsplat_raster_types;

// Splat2D record written by gsplat_raster_project.comp and read by the raster
// vertex and mesh shaders (scalar layout: 16 floats, 64-byte stride). color is
// premultiplied (rgb already scaled by alpha).
public struct Splat2D {
    public float4 color;     // premultiplied rgb, alpha
    public float3 conic;     // inverse projected covariance (a, b, c) for [[a,b],[b,c]]
    public float2 centerPx;  // splat centre in pixels (fragment-shader falloff origin)
    public float2 ndcCenter; // splat centre in NDC
    public float2 ndcAxis1;  // major eigen-axis * radius, in NDC (quad corner.x basis)
    public float2 ndcAxis2;  // minor eigen-axis * radius, in NDC (quad corner.y basis)
    public float depth;      // clip.z / clip.w
};

// The interface from the vertex or mesh stage to gsplat_raster.frag.
public struct SplatVertex {
    public float4 position : SV_Position;
    [[vk::location(0)]] public float4 color;
    [[vk::location(1)]] public float3 conic;
    [[vk::location(2)]] public float2 centerPx;
    // The splat's NDC depth, for gsplat_raster_depth.frag.
    [[vk::location(3)]] public nointerpolation float depth;
};

// Corners {(-1,-1), (1,-1), (-1,1), (1,1)} of a triangle-strip quad. The
// axes a corner scales along are the covariance eigenvectors (precomputed),
// so the quad is oriented per splat.
public static const float2 CORNERS[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };

// The vertex of corner `corner` of the quad of splat `s`.
public SplatVertex splatVertex(Splat2D s, uint corner) {
    float2 c = CORNERS[corner];
    float2 ndcOffset = s.ndcAxis1 * c.x + s.ndcAxis2 * c.y;
    SplatVertex v;
    v.position = float4(s.ndcCenter + ndcOffset, s.depth, 1.0);
    v.color = s.color;
    v.conic = s.conic;
    v.centerPx = s.centerPx;
    v.depth = s.depth;
    return v;
}
