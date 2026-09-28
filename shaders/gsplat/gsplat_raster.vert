#version 450
#extension GL_EXT_scalar_block_layout : enable

// Trivial vertex shader for the raster backend: all per-splat math (covariance
// projection, eigendecomposition, SH colour) is precomputed once per splat by
// gsplat_raster_project.comp and stored in the Splat2D buffer. This stage just
// reads one record, places the quad corner along the precomputed eigen-axes, and
// passes conic + colour + centre through to the fragment shader.

// set 1, binding 0: precomputed per-splat 2D attributes, keyed by splat id.
// binding 1: the sorted (back-to-front) splat-id permutation.
struct Splat2D {
    vec4  color;      // premultiplied rgb, alpha
    vec3  conic;      // inverse projected covariance (a, b, c)
    vec2  centerPx;   // splat centre in pixels
    vec2  ndcCenter;  // splat centre in NDC
    vec2  ndcAxis1;   // major eigen-axis * radius, in NDC
    vec2  ndcAxis2;   // minor eigen-axis * radius, in NDC
    float depth;      // clip.z / clip.w
};
layout(scalar, set = 1, binding = 0) readonly buffer Splat2DBuf  { Splat2D splat2D[];    };
layout(scalar, set = 1, binding = 1) readonly buffer SortedIndices{ uint   sortedIndices[];};

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec3 outConic;
layout(location = 2) out vec2 outCenterPx;

// {(-1,-1), (1,-1), (-1,1), (1,1)} — a triangle-strip quad from gl_VertexIndex,
// drawn with no vertex buffer. The axes the corner scales along are the
// covariance eigenvectors (precomputed), so the quad is oriented per-splat.
vec2 quadCorner(uint vertexIndex) {
    const vec2 corners[4] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    return corners[vertexIndex];
}

void main() {
    uint splatId = sortedIndices[gl_InstanceIndex];
    Splat2D s = splat2D[splatId];

    vec2 corner = quadCorner(gl_VertexIndex);
    vec2 ndcOffset = s.ndcAxis1 * corner.x + s.ndcAxis2 * corner.y;

    gl_Position = vec4(s.ndcCenter + ndcOffset, s.depth, 1.0);

    outColor    = s.color;
    outConic    = s.conic;
    outCenterPx = s.centerPx;
}
