// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// A leaf of one or more lobes (Kuth et al., Sec. 3.3, after Garg 2011). Each
// half lobe is a fixed set of triangles: strips between the polyline of
// treeLeafPolyline() and the leaf axis, plus one triangle per Bezier curve
// whose edge the fragment shader cuts with Loop and Blinn's test. With
// distance the curves flatten and the lobes turn into one, until the leaf
// is drawn as the polygon of its curve ends.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"
#include "tree_leaf_common.glsl"

layout(std430, set = 1, binding = 0) readonly buffer Leaves {
    LeafRecord leaves[];
};
layout(std430, set = 1, binding = 1) readonly buffer Lists {
    uint lists[];
};
layout(std430, set = 1, binding = 2) readonly buffer Params {
    TreeTypeParams params[];
};
layout(std430, set = 1, binding = 3) readonly buffer Settings {
    TreeRenderSettings settings[];
};

layout(push_constant) uniform Push {
    uint bucket;
    uint capacity;
    uint lobes;
}
pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColor;
layout(location = 3) out vec2 outCurveUv;
layout(location = 4) out float outCurveSide; // 0 inner triangle, 1 keep the chord side, -1 the control side
layout(location = 5) out vec2 outLeafPos;    // position on the lobe, in leaf lengths

void main() {
    LeafRecord leaf = leaves[lists[pc.bucket * pc.capacity + gl_InstanceIndex]];
    TreeTypeParams p = params[treeType(leaf.ids)];
    TreeRenderSettings s = settings[0];
    vec2 lod = treeLeafLodOf(leaf, s);
    float flatten = lod.x;

    uint halfVertices = treeLeafHalfVertices(pc.bucket);
    uint vertex = uint(gl_VertexIndex);
    uint lobe = vertex / (2u * halfVertices);
    bool mirrored = (vertex / halfVertices) % 2u == 1u;
    uint local = vertex % halfVertices;
    uint triangle = local / 3u;
    uint corner = local % 3u;

    uint lobes = uint(clamp(p.leafLobes.x, 1.0f, float(pc.lobes)));
    if (lobe >= lobes) {
        gl_Position = vec4(2.0f, 2.0f, 2.0f, 1.0f); // outside the view: the triangle is clipped
        return;
    }

    TreeLeafOutline o = treeLeafOutline(p.leafShape, 0.5f * p.leafCount.z);
    vec2 q;
    vec2 uv = vec2(0.0f);
    float side = 0.0f;
    if (pc.bucket == 0u && triangle >= 16u) {
        int c = int(triangle) - 16;
        vec2 a = o.v[2 * c];
        vec2 b = treeLeafControl(o, 2 * c + 1, flatten);
        vec2 e = o.v[2 * c + 2];
        q = corner == 0u ? a : (corner == 1u ? b : e);
        uv = corner == 0u ? vec2(0.0f) : (corner == 1u ? vec2(0.5f, 0.0f) : vec2(1.0f));
        side = treeLeafCurveConvex(a, b, e) ? 1.0f : -1.0f;
    } else {
        // Strip k between the polyline and the axis.
        uint k = triangle / 2u;
        uint stride = pc.bucket == 0u ? 1u : 2u; // the simplified leaf uses the curve ends only
        vec2 p0 = treeLeafPolyline(o, int(k * stride), flatten);
        vec2 p1 = treeLeafPolyline(o, int((k + 1u) * stride), flatten);
        vec2 a0 = vec2(0.0f, p0.y);
        vec2 a1 = vec2(0.0f, p1.y);
        if (triangle % 2u == 0u) {
            q = corner == 0u ? p0 : (corner == 1u ? p1 : a1);
        } else {
            q = corner == 0u ? p0 : (corner == 1u ? a1 : a0);
        }
    }
    if (mirrored) {
        q.x = -q.x;
    }
    outLeafPos = q;

    // Lobes fan out around the leaf base; with distance they turn into one.
    float offset = float(lobe) - 0.5f * float(lobes - 1u);
    float angle = radians(offset * p.leafLobes.y) * (1.0f - flatten);
    float lobeScale = pow(p.leafLobes.z, abs(offset));
    q = lobeScale * vec2(cos(angle) * q.x + sin(angle) * q.y, -sin(angle) * q.x + cos(angle) * q.y);

    vec3 x = qrot(leaf.rot, vec3(1.0f, 0.0f, 0.0f));
    vec3 z = qforward(leaf.rot);
    // Flattening takes away the area between chords and curves; growing the leaf keeps its area.
    float areaScale = sqrt(treeLeafHalfArea(o, 0.0f) / max(treeLeafHalfArea(o, flatten), 1e-6f));
    float size = leaf.posScale.w * lod.y * areaScale;
    vec3 worldPos = leaf.posScale.xyz + size * (q.x * x + q.y * z);

    // A little colour variation per leaf.
    float variation = treeRand(leaf.ids.y, 0x78u) - 0.5f;
    outColor = p.leafColor.rgb * (1.0f + 0.35f * variation) + vec3(0.04f, 0.03f, 0.0f) * variation;
    outWorldPos = worldPos;
    outNormal = qrot(leaf.rot, vec3(0.0f, 1.0f, 0.0f));
    outCurveUv = uv;
    outCurveSide = side;
    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0f);
}
