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

// Values the whole generation reads.
struct TreeFrame {
    vec4 worldUp;    // xyz unit vector up, w unused
    vec4 wind;       // xyz direction the wind blows to, w strength
    vec4 timeSeason; // x time in seconds, y season in [0, 4), zw unused
    vec4 reserved;
};

const float TREE_PI = 3.14159265f;

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

#endif // KLARTRAUM_TREE_TREE_SHARED_HPP
