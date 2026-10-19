// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The procedural tree model of Kuth et al., "Real-Time GPU Tree Generation"
// (HPG 2025), which extends Weber and Penn, "Creation and Rendering of Realistic
// Trees" (SIGGRAPH 1995).
//
// This file is both GLSL and C++: the tree shaders include it, and so does
// tree_model.hpp, which runs the same generation on the CPU as the reference
// the GPU results are tested against. It is written in the subset both
// languages share: float literals carry an `f`, structs are filled member by
// member, and there are no swizzles. The includer defines vec2/vec3/vec4/uvec4/
// uint and the GLSL functions (glm in C++), and TREE_FN, which prefixes every
// function (`inline` in C++), before including this file. The
// generation itself, which needs inputs and outputs, is in tree_generate.hpp.
//
// All structs consist of vec4/uvec4 members only, so their std430 layout is
// the C++ layout.

#ifndef KLARTRAUM_TREE_TREE_SHARED_HPP
#define KLARTRAUM_TREE_TREE_SHARED_HPP

// Threads that generate one stem together; clones split them into blocks
// (Kuth et al., Table 1).
#define TREE_WAVE_SIZE 32u
#define TREE_MAX_LEVELS 4

// Per tree type. Angles are in degrees, lengths in the units of Weber and Penn
// (metres for the presets) before the instance scale is applied. Per-level
// values hold the stem levels 0 (trunk) to 3 in x to w.
struct TreeTypeParams {
    vec4 shapeBase;     // x shape (0..7), y baseSize, z scale, w scaleV
    vec4 ratios;        // x levels (1..4), y ratio, z ratioPower, w flare
    vec4 leafCount;     // x leaves per stem, y leafScale, z leafScaleX, w attractionUp
    vec4 leafAngles;    // x leafDownAngle, y leafDownAngleV, z leafRotate, w leafRotateV
    vec4 leafShape;     // lobe outline: x alpha, y beta, z gamma (degrees), w offset o of the widest point
    vec4 leafLobes;     // x lobe count n, y angle between lobes lambda (degrees), z lobe scale falloff k, w petiole
    vec4 barkColor;     // rgb, w unused
    vec4 leafColor;     // rgb in summer, w unused
    vec4 leafColorFall; // rgb in fall, w seasonal offset of the type
    vec4 nLength;
    vec4 nLengthV;
    vec4 nTaper;
    vec4 nSegSplits;
    vec4 nSplitAngle;
    vec4 nSplitAngleV;
    vec4 nCurveRes;
    vec4 nCurve;
    vec4 nCurveBack;
    vec4 nCurveV;
    vec4 nDownAngle;
    vec4 nDownAngleV;
    vec4 nRotate;
    vec4 nRotateV;
    vec4 nBranches;
};

// One tree in the scene.
struct TreeInstance {
    vec4 posScale; // xyz base of the trunk, w scale applied to all lengths
    vec4 rot;      // orientation quaternion (x, y, z, w); the trunk grows along its +z
    uvec4 ids;     // x tree type, y seed, zw unused
};

// The start of a stem level: the S record of Kuth et al.
struct StemRecord {
    vec4 posLength; // xyz base, w length
    vec4 rot;       // quaternion; the stem grows along its +z
    vec4 dims;      // x base radius, y parent length, z offset on the parent, w lengthChildMax
    uvec4 ids;      // x type | level << 16, y seed, z instance, w unused
};

// One segment of a stem between two rings: the D record.
struct SegmentRecord {
    vec4 fromPosRadius;
    vec4 fromRot;
    vec4 toPosRadius;
    vec4 toRot;
    vec4 shape; // x dr/dz at the from ring, y dr/dz at the to ring, z position along the stem in [0, 1], w stem length
    uvec4 ids;  // x type | level << 16, y seed of the clone, z instance, w segment index
};

// One leaf: the L record.
struct LeafRecord {
    vec4 posScale; // xyz leaf base, w length of the leaf
    vec4 rot;      // quaternion; the leaf points along +z, its blade lies in the xz plane
    uvec4 ids;     // x type, y seed, z instance, w unused
};

// A vertex of the static scene meshes (pot, soil, table).
struct TreeSceneVertex {
    vec4 pos;    // xyz, w unused
    vec4 normal; // xyz, w unused
    vec4 color;  // rgb albedo, w material (TREE_MATERIAL_*)
};

// Values the whole generation reads.
struct TreeFrame {
    vec4 worldUp;    // xyz unit vector up, w unused
    vec4 wind;       // xyz direction the wind blows to, w strength
    vec4 timeSeason; // x time in seconds, y season in [0, 4), zw unused
    vec4 reserved;
};

// Per-frame values of the tree renderers.
struct TreeRenderSettings {
    vec4 viewport; // x width, y height in pixels, z pixels per triangle edge (Delta), w minimum stem radius in pixels
    vec4 sun;      // xyz unit vector towards the sun, w intensity
    vec4 sky;      // rgb ambient light from above, w unused
    vec4 ground;   // rgb ambient light from below, w unused
    // Continuous leaf level of detail, by the leaf's length in pixels: x where leaves start to flatten and
    // their lobes to merge, y where they are fully simplified, z below which leaves thin out, w the fraction
    // of leaves kept at the farthest distance.
    vec4 leafLod;
    // x: 1 to hide fragments behind the scene depth buffer (Gaussians drawn before, e.g. the lantern),
    // 0 to ignore it; yzw unused.
    vec4 occlusion;
};

const float TREE_PI = 3.14159265f;

// Stem draw buckets: bucket b holds segments that need at most
// treeStemBucketAround(b) x treeStemBucketAlong(b) quads.
#define TREE_STEM_BUCKETS 5u

// Random keys, one range per purpose so that the decisions are independent.
const uint TREE_KEY_SCALE = 1u;
const uint TREE_KEY_LENGTH = 2u;
const uint TREE_KEY_COUNT = 3u;
const uint TREE_KEY_CURVE = 0x100u;
const uint TREE_KEY_SPLIT_ANGLE = 0x200u;
const uint TREE_KEY_SPLIT_PHI = 0x300u;
const uint TREE_KEY_CHILD = 0x10000u;
const uint TREE_KEY_SHARE = 0x20000u;
const uint TREE_KEY_LENGTH_V = 0x30000u;
const uint TREE_KEY_ROTATE_V = 0x40000u;
const uint TREE_KEY_DOWN_V = 0x50000u;

// --- hashing -------------------------------------------------------------

// PCG hash (Jarzynski and Olano, "Hash Functions for GPU Rendering", JCGT 2020).
TREE_FN uint treeHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

TREE_FN uint treeHash2(uint a, uint b) { return treeHash(a ^ treeHash(b + 0x9e3779b9u)); }

// Uniform in [0, 1); 24 bits, so CPU and GPU convert to the same float.
TREE_FN float treeRand(uint seed, uint key) { return float(treeHash2(seed, key) >> 8u) * (1.0f / 16777216.0f); }

// Uniform in [-1, 1).
TREE_FN float treeRandSigned(uint seed, uint key) { return 2.0f * treeRand(seed, key) - 1.0f; }

// --- quaternions (x, y, z, w) ---------------------------------------------

TREE_FN vec4 qmul(vec4 a, vec4 b) {
    return vec4(a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

TREE_FN vec4 qaxis(vec3 axis, float angle) {
    float s = sin(0.5f * angle);
    return vec4(axis.x * s, axis.y * s, axis.z * s, cos(0.5f * angle));
}

TREE_FN vec3 qrot(vec4 q, vec3 v) {
    vec3 u = vec3(q.x, q.y, q.z);
    vec3 t = 2.0f * cross(u, v);
    return v + q.w * t + cross(u, t);
}

TREE_FN vec3 qforward(vec4 q) { return qrot(q, vec3(0.0f, 0.0f, 1.0f)); }

// Normalized linear interpolation along the shorter arc.
TREE_FN vec4 qnlerp(vec4 a, vec4 b, float t) {
    vec4 bb = dot(a, b) < 0.0f ? -b : b;
    return normalize(mix(a, bb, t));
}

// --- curves ----------------------------------------------------------------

// Cubic Hermite spline between p0 and p1 with tangents m0 and m1.
TREE_FN vec3 treeHermite(vec3 p0, vec3 p1, vec3 m0, vec3 m1, float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return (2.0f * t3 - 3.0f * t2 + 1.0f) * p0 + (t3 - 2.0f * t2 + t) * m0 + (-2.0f * t3 + 3.0f * t2) * p1 +
           (t3 - t2) * m1;
}

TREE_FN vec3 treeHermiteTangent(vec3 p0, vec3 p1, vec3 m0, vec3 m1, float t) {
    float t2 = t * t;
    return (6.0f * t2 - 6.0f * t) * p0 + (3.0f * t2 - 4.0f * t + 1.0f) * m0 + (-6.0f * t2 + 6.0f * t) * p1 +
           (3.0f * t2 - 2.0f * t) * m1;
}

// --- the Weber-Penn model ------------------------------------------------

// Weber and Penn's shape ratio for the shapes 0 (conical) to 7 (tend flame).
TREE_FN float treeShapeRatio(int shape, float ratio) {
    float r = clamp(ratio, 0.0f, 1.0f);
    if (shape == 0) {
        return 0.2f + 0.8f * r;
    } else if (shape == 1) {
        return 0.2f + 0.8f * sin(TREE_PI * r);
    } else if (shape == 2) {
        return 0.2f + 0.8f * sin(0.5f * TREE_PI * r);
    } else if (shape == 3) {
        return 1.0f;
    } else if (shape == 4) {
        return 0.5f + 0.5f * r;
    } else if (shape == 5) {
        return r <= 0.7f ? r / 0.7f : (1.0f - r) / 0.3f;
    } else if (shape == 6) {
        return 1.0f - 0.8f * r;
    }
    return r <= 0.7f ? 0.5f + 0.5f * r / 0.7f : 0.5f + 0.5f * (1.0f - r) / 0.3f;
}

TREE_FN int treeLevel(uvec4 ids) { return int(ids.x >> 16u); }

TREE_FN uint treeType(uvec4 ids) { return ids.x & 0xffffu; }

// Radius of a stem at z in [0, 1] along it: linear taper, and the flare at the
// trunk's base.
TREE_FN float treeStemRadius(TreeTypeParams p, StemRecord s, float z) {
    int level = treeLevel(s.ids);
    float r = s.dims.x * (1.0f - clamp(p.nTaper[level], 0.0f, 1.0f) * z);
    if (level == 0 && p.ratios.w > 0.0f) {
        float y = max(0.0f, 1.0f - 8.0f * z);
        r *= p.ratios.w * (pow(100.0f, y) - 1.0f) / 100.0f + 1.0f;
    }
    return r;
}

// Change of the radius per unit length along the stem, at a ring. It depends
// only on the ring's z, so both segments that share a ring see the same value.
TREE_FN float treeStemRadiusSlope(TreeTypeParams p, StemRecord s, float z) {
    float e = 1.0f / 64.0f;
    float z0 = max(z - e, 0.0f);
    float z1 = min(z + e, 1.0f);
    return (treeStemRadius(p, s, z1) - treeStemRadius(p, s, z0)) / ((z1 - z0) * s.posLength.w);
}

// --- shading -----------------------------------------------------------------

// Lighting of the classical renderer, also baked into the Gaussians: a sun and
// an ambient term from the sky above and the ground below.
TREE_FN vec3 treeShade(vec3 albedo, vec3 normal, TreeRenderSettings s) {
    float sun = max(dot(normal, vec3(s.sun.x, s.sun.y, s.sun.z)), 0.0f) * s.sun.w;
    float hemisphere = 0.5f + 0.5f * normal.y;
    vec3 ambient = mix(vec3(s.ground.x, s.ground.y, s.ground.z), vec3(s.sky.x, s.sky.y, s.sky.z), hemisphere);
    return albedo * (ambient + vec3(sun, sun, sun));
}

// The colour as the render targets store it: as displayed, with gamma 2.2.
TREE_FN vec3 treeDisplayColor(vec3 linearColor) {
    float g = 1.0f / 2.2f;
    return pow(clamp(linearColor, vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f)), vec3(g, g, g));
}

// Value noise in [0, 1] on the integer lattice, smoothly interpolated.
TREE_FN float treeValueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 w = f * f * (vec3(3.0f, 3.0f, 3.0f) - 2.0f * f);
    uint x = uint(int(i.x));
    uint y = uint(int(i.y));
    uint z = uint(int(i.z));
    float n000 = treeRand(treeHash2(x, y), z);
    float n100 = treeRand(treeHash2(x + 1u, y), z);
    float n010 = treeRand(treeHash2(x, y + 1u), z);
    float n110 = treeRand(treeHash2(x + 1u, y + 1u), z);
    float n001 = treeRand(treeHash2(x, y), z + 1u);
    float n101 = treeRand(treeHash2(x + 1u, y), z + 1u);
    float n011 = treeRand(treeHash2(x, y + 1u), z + 1u);
    float n111 = treeRand(treeHash2(x + 1u, y + 1u), z + 1u);
    return mix(mix(mix(n000, n100, w.x), mix(n010, n110, w.x), w.y),
               mix(mix(n001, n101, w.x), mix(n011, n111, w.x), w.y), w.z);
}

// Three octaves of value noise.
TREE_FN float treeFbm(vec3 p) {
    return 0.5f * treeValueNoise(p) + 0.3f * treeValueNoise(2.03f * p) + 0.2f * treeValueNoise(4.11f * p);
}

// Scene materials, by the w of a vertex colour.
#define TREE_MATERIAL_PLAIN 0.0f
#define TREE_MATERIAL_MOSS 1.0f
#define TREE_MATERIAL_WOOD 2.0f
#define TREE_MATERIAL_GLAZE 3.0f

// The albedo of a scene surface at world position pos: plain colour, moss
// over soil, wood grain along x, or a glaze with fine mottling.
TREE_FN vec3 treeSceneAlbedo(vec4 color, vec3 pos) {
    vec3 base = vec3(color.x, color.y, color.z);
    if (color.w == TREE_MATERIAL_MOSS) {
        float coverage = smoothstep(0.35f, 0.6f, treeFbm(pos * 9.0f));
        float tint = treeFbm(pos * 31.0f + vec3(7.0f, 0.0f, 3.0f));
        vec3 soil = vec3(0.16f, 0.11f, 0.07f) * (0.7f + 0.6f * tint);
        vec3 moss = base * (0.75f + 0.5f * tint);
        return mix(soil, moss, coverage);
    }
    if (color.w == TREE_MATERIAL_WOOD) {
        float grain = treeFbm(vec3(pos.x * 1.5f, pos.y * 40.0f, pos.z * 40.0f));
        float rings = 0.5f + 0.5f * sin(40.0f * pos.z + 8.0f * grain);
        return base * (0.75f + 0.25f * rings + 0.2f * grain);
    }
    if (color.w == TREE_MATERIAL_GLAZE) {
        return base * (0.85f + 0.3f * treeFbm(pos * 14.0f));
    }
    return base;
}

// --- stem tessellation (Kuth et al., Sec. 3.2) ----------------------------

TREE_FN uint treeStemBucketAround(uint b) { return b == 0u ? 1u : 2u << b; }

TREE_FN uint treeStemBucketAlong(uint b) { return 1u << b; }

// Half the angle of the arc of a ring that faces the camera. drdz is the
// change of the radius along the growth direction, x the cosine between the
// growth direction and the direction to the camera.
TREE_FN float treeOpeningAngle(float drdz, float x) {
    float s = sqrt(max(1.0f - x * x, 1e-12f));
    return acos(clamp(drdz * x / s, -1.0f, 1.0f));
}

// Position in [0, 1] of vertex i of an edge with fractional tessellation
// factor F (at most maxSegments segments), with fractional odd spacing: all
// segments have the length 1 / F except the middle one, which shrinks to
// nothing as F falls to the next lower integer. Vertices beyond the last one
// coincide with it.
TREE_FN float treeTessSpacing(uint i, float factor, uint maxSegments) {
    float f = clamp(factor, 1.0f, float(maxSegments));
    uint n = uint(ceil(f));
    uint k = min(i, n);
    uint h = n / 2u;
    float s = k <= h ? float(k) / f : 1.0f - float(n - k) / f;
    return clamp(s, 0.0f, 1.0f);
}

// The smallest bucket with at least `around` x `along` quads, or the largest.
TREE_FN uint treeStemBucket(float around, float along) {
    for (uint b = 0u; b < TREE_STEM_BUCKETS; ++b) {
        if (float(treeStemBucketAround(b)) >= ceil(around) && float(treeStemBucketAlong(b)) >= ceil(along)) {
            return b;
        }
    }
    return TREE_STEM_BUCKETS - 1u;
}

// --- leaves (Kuth et al., Sec. 3.3) ----------------------------------------

// The right half of a lobe outline, in units of the leaf length, from the base
// v0 = (0, 0) over the widest point v4 = (halfWidth, offset) to the tip
// v8 = (0, 1): four quadratic Bezier curves (v0, v1, v2), (v2, v3, v4),
// (v4, v5, v6), (v6, v7, v8). alpha is the angle at which the outline leaves
// the base, beta its tilt at the widest point and gamma the angle at which it
// arrives at the tip, all against the leaf axis. v2 and v6 lie halfway
// between their neighbouring control points, so the outline is smooth.
struct TreeLeafOutline {
    vec2 v[9];
};

// Control points (xy, zw) of two joined quadratic curves from a, leaving in
// direction da, to b, arriving in direction db: 60 % of the way to where the
// two tangents meet, which keeps the outline convex; where they do not meet
// ahead of both points, half the chord length along the tangents, which
// makes the outline concave.
TREE_FN vec4 treeLeafControls(vec2 a, vec2 da, vec2 b, vec2 db) {
    vec2 ab = b - a;
    float denominator = da.x * db.y - da.y * db.x;
    if (abs(denominator) > 1e-4f) {
        float s = (ab.x * db.y - ab.y * db.x) / denominator;
        float t = (ab.y * da.x - ab.x * da.y) / denominator;
        if (s > 0.0f && t > 0.0f) {
            vec2 c0 = a + 0.6f * s * da;
            vec2 c1 = b - 0.6f * t * db;
            return vec4(c0.x, c0.y, c1.x, c1.y);
        }
    }
    float l = 0.5f * length(ab);
    vec2 c0 = a + l * da;
    vec2 c1 = b - l * db;
    return vec4(c0.x, c0.y, c1.x, c1.y);
}

TREE_FN TreeLeafOutline treeLeafOutline(vec4 shape, float halfWidth) {
    vec2 dAlpha = vec2(sin(radians(shape.x)), cos(radians(shape.x)));
    vec2 dBeta = vec2(sin(radians(shape.y)), cos(radians(shape.y)));
    vec2 dGamma = vec2(-sin(radians(shape.z)), cos(radians(shape.z)));
    TreeLeafOutline o;
    o.v[0] = vec2(0.0f, 0.0f);
    o.v[4] = vec2(halfWidth, shape.w);
    o.v[8] = vec2(0.0f, 1.0f);
    vec4 lower = treeLeafControls(o.v[0], dAlpha, o.v[4], dBeta);
    vec4 upper = treeLeafControls(o.v[4], dBeta, o.v[8], dGamma);
    o.v[1] = vec2(lower.x, lower.y);
    o.v[3] = vec2(lower.z, lower.w);
    o.v[5] = vec2(upper.x, upper.y);
    o.v[7] = vec2(upper.z, upper.w);
    o.v[2] = 0.5f * (o.v[1] + o.v[3]);
    o.v[6] = 0.5f * (o.v[5] + o.v[7]);
    return o;
}

// Whether the curve (a, b, c), running from the base towards the tip on the
// right half, bulges outwards: its control point lies right of the chord.
TREE_FN bool treeLeafCurveConvex(vec2 a, vec2 b, vec2 c) {
    vec2 chord = c - a;
    vec2 toControl = b - a;
    return chord.x * toControl.y - chord.y * toControl.x < 0.0f;
}

// Control point i (odd) of an outline flattened by `flatten` in [0, 1]: moved
// towards the middle of its chord, which turns the curve into a straight line
// at 1 (the low level of detail).
TREE_FN vec2 treeLeafControl(TreeLeafOutline o, int i, float flatten) {
    return mix(o.v[i], 0.5f * (o.v[i - 1] + o.v[i + 1]), flatten);
}

// Point i (0..8) of the polyline that bounds the inner triangles of a half
// lobe: the curve ends, and for each curve its control point if the curve is
// concave (it then cuts into the inner area) or its chord's middle if convex
// (it then adds to it).
TREE_FN vec2 treeLeafPolyline(TreeLeafOutline o, int i, float flatten) {
    if (i % 2 == 0) {
        return o.v[i];
    }
    vec2 control = treeLeafControl(o, i, flatten);
    vec2 middle = 0.5f * (o.v[i - 1] + o.v[i + 1]);
    return treeLeafCurveConvex(o.v[i - 1], control, o.v[i + 1]) ? middle : control;
}

// Area of a half lobe flattened by `flatten`: the polygon of the curve ends
// and the axis, plus for each curve the 2/3 of its control triangle that lies
// between chord and curve, which flattening shrinks linearly.
TREE_FN float treeLeafHalfArea(TreeLeafOutline o, float flatten) {
    float area = 0.0f;
    for (int c = 0; c < 4; ++c) {
        vec2 a = o.v[2 * c];
        vec2 b = o.v[2 * c + 1];
        vec2 e = o.v[2 * c + 2];
        area += 0.5f * (a.x + e.x) * (e.y - a.y);
        vec2 chord = e - a;
        vec2 toControl = b - a;
        area -= (1.0f - flatten) * (2.0f / 3.0f) * 0.5f * (chord.x * toControl.y - chord.y * toControl.x);
    }
    return area;
}

// Loop and Blinn's implicit form of a quadratic Bezier curve with the texture
// coordinates (0, 0), (0.5, 0), (1, 1) at its points: negative on the side of
// the chord, positive on the side of the control point.
TREE_FN float treeLoopBlinn(vec2 uv) { return uv.x * uv.x - uv.y; }

// Level of detail of a leaf `pixels` long on screen: x how far its outline is
// flattened and its lobes merged (0 full detail, 1 simplest), y the factor its
// size is scaled by (0: not drawn), given its random number r in [0, 1).
// Distant leaves are thinned out and the remaining ones grow so that the
// foliage keeps its coverage; leaves near the threshold shrink away smoothly.
TREE_FN vec2 treeLeafLod(float pixels, vec4 lod, float r) {
    float flatten = clamp((lod.x - pixels) / max(lod.x - lod.y, 1e-3f), 0.0f, 1.0f);
    float keep = clamp(pixels / max(lod.z, 1e-3f), lod.w, 1.0f);
    float fade = clamp((keep - r) / 0.1f, 0.0f, 1.0f);
    return vec2(flatten, fade / sqrt(keep));
}

#endif // KLARTRAUM_TREE_TREE_SHARED_HPP
