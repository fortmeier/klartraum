// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Generation of the tree model in tree_shared.hpp, shared by the tree shaders
// and the CPU reference (see tree_shared.hpp for the language subset). Before
// including this file, the includer defines
// - TREE_CTX_PARAM / TREE_CTX_ARG: an extra trailing parameter/argument through
//   which C++ passes its inputs and output lists (empty in the shaders),
// - TREE_PARAMS(type), TREE_INSTANCE(index), TREE_FRAME: the read-only inputs,
// - treeEmitStem(), treeEmitSegment(), treeEmitLeaf(): append a record.

#ifndef KLARTRAUM_TREE_TREE_GENERATE_HPP
#define KLARTRAUM_TREE_TREE_GENERATE_HPP

// The trunk of an instance.
TREE_FN StemRecord treeRootStem(TreeInstance inst, uint instanceIndex TREE_CTX_PARAM) {
    uint type = inst.ids.x;
    uint seed = inst.ids.y;
    TreeTypeParams p = TREE_PARAMS(type);
    float treeScale = p.shapeBase.z + p.shapeBase.w * treeRandSigned(seed, TREE_KEY_SCALE);
    float length0 = treeScale * (p.nLength[0] + p.nLengthV[0] * treeRandSigned(seed, TREE_KEY_LENGTH));
    float scale = inst.posScale.w;
    StemRecord s;
    s.posLength = vec4(inst.posScale.x, inst.posScale.y, inst.posScale.z, length0 * scale);
    s.rot = inst.rot;
    s.dims = vec4(length0 * p.ratios.y * scale, 0.0f, 0.0f, 0.0f);
    s.ids = uvec4(type, treeHash2(seed, 0x7f000000u), instanceIndex, 0u);
    return s;
}

// Orientation of a stem at ring i: the previous orientation turned by the
// stem's curvature and, for levels 2 and up, towards (attractionUp > 0) or away
// from the sky.
TREE_FN vec4 treeCurveRing(TreeTypeParams p, int level, vec4 rot, int i, int segments, uint cloneSeed, vec3 up) {
    float curve = p.nCurve[level];
    float back = p.nCurveBack[level];
    float n = float(segments);
    float angle;
    if (back == 0.0f) {
        angle = curve / n;
    } else {
        float halfCount = max(n * 0.5f, 1.0f);
        angle = i < (segments + 1) / 2 ? curve / halfCount : back / halfCount;
    }
    angle += p.nCurveV[level] * treeRandSigned(cloneSeed, TREE_KEY_CURVE + uint(i)) / n;
    vec4 q = qmul(rot, qaxis(vec3(1.0f, 0.0f, 0.0f), radians(angle)));

    float attraction = p.leafCount.w;
    if (level >= 2 && attraction != 0.0f) {
        vec3 f = qforward(q);
        float declination = acos(clamp(dot(f, up), -1.0f, 1.0f));
        vec3 axis = cross(f, up);
        float axisLength = length(axis);
        if (axisLength > 1e-5f) {
            float turn = clamp(attraction * declination / n, declination - TREE_PI, declination);
            q = normalize(qmul(qaxis(axis / axisLength, turn), q));
        }
    }
    return q;
}

// A ring's orientation bent by the wind of frame f (Kuth et al. Sec. 3.5):
// turned towards the wind's direction by gusts that travel downwind, plus a
// sway at a frequency that falls with the inverse square root of the stem's
// length. Thin stems (radius in the units of the tree type, before the
// instance's scale) bend more. Every ring of a stem bends, and the rings and
// children after it start from the bent orientation, so the whole tree moves
// as a skeleton with one bone per segment.
TREE_FN vec4 treeWind(TreeFrame f, vec4 q, vec3 pos, float radius, float stemLength, uint seed, int segments) {
    float strength = f.wind.w;
    if (strength == 0.0f) {
        return q;
    }
    vec3 direction = vec3(f.wind.x, f.wind.y, f.wind.z);
    vec3 axis = cross(qforward(q), direction);
    float axisLength = length(axis);
    if (axisLength < 1e-5f) {
        return q;
    }
    float t = f.timeSeason.x;
    float gust = treeValueNoise(vec3(0.35f * t - 0.8f * dot(pos, direction), 0.0f, 0.0f));
    float frequency = 0.6f / sqrt(max(stemLength, 0.05f));
    float sway = sin(2.0f * TREE_PI * (frequency * t + treeRand(seed, 0x90u)));
    float flex = 0.04f / (radius + 0.01f);
    float angle = strength * flex * (0.6f * gust + 0.25f * sway * (0.5f + gust)) / float(segments);
    return normalize(qmul(qaxis(axis / axisLength, angle), q));
}

TREE_FN float treeBranchCount(TreeTypeParams p, StemRecord s, int level) {
    float branches = p.nBranches[level + 1];
    if (level == 0) {
        return branches;
    }
    float parentLength = max(s.dims.y, 1e-6f);
    if (level == 1) {
        return branches * (0.2f + 0.8f * (s.posLength.w / parentLength) / max(s.dims.w, 1e-6f));
    }
    return branches * (1.0f - 0.5f * s.dims.z / parentLength);
}

TREE_FN float treeLeafCount(TreeTypeParams p, StemRecord s) {
    float ratio = s.dims.y > 0.0f ? s.dims.z / s.dims.y : 1.0f;
    return p.leafCount.x * treeShapeRatio(4, ratio);
}

TREE_FN StemRecord treeChildStem(TreeTypeParams p, StemRecord s, int k, uint cloneSeed, float offset, float base,
                                 vec3 pos, vec4 rot, float parentRadius) {
    int level = treeLevel(s.ids);
    int c = level + 1;
    float parentLength = s.posLength.w;
    uint seed = treeHash2(cloneSeed, TREE_KEY_CHILD + uint(k));
    float lengthChildMax = p.nLength[c] + p.nLengthV[c] * treeRandSigned(seed, TREE_KEY_LENGTH_V);
    float childLength;
    if (level == 0) {
        float ratio = (parentLength - offset) / max(parentLength - base, 1e-6f);
        childLength = parentLength * lengthChildMax * treeShapeRatio(int(p.shapeBase.x), ratio);
    } else {
        childLength = lengthChildMax * (parentLength - 0.6f * offset);
    }
    childLength = max(childLength, 0.0f);
    float radius = s.dims.x * pow(childLength / max(parentLength, 1e-6f), p.ratios.z);
    radius = min(radius, parentRadius);

    float rotate = p.nRotate[c];
    float phi;
    if (rotate >= 0.0f) {
        phi = float(k) * rotate;
    } else {
        // Negative rotate: the branches alternate between two sides.
        phi = (k % 2 == 0 ? 1.0f : -1.0f) * (180.0f + rotate);
    }
    phi += p.nRotateV[c] * treeRandSigned(seed, TREE_KEY_ROTATE_V);

    float downV = p.nDownAngleV[c];
    float down;
    if (downV >= 0.0f) {
        down = p.nDownAngle[c] + downV * treeRandSigned(seed, TREE_KEY_DOWN_V);
    } else {
        float ratio = (parentLength - offset) / max(parentLength - base, 1e-6f);
        down = p.nDownAngle[c] + downV * (1.0f - 2.0f * treeShapeRatio(0, ratio));
    }

    StemRecord child;
    child.posLength = vec4(pos.x, pos.y, pos.z, childLength);
    child.rot = normalize(
        qmul(rot, qmul(qaxis(vec3(0.0f, 0.0f, 1.0f), radians(phi)), qaxis(vec3(1.0f, 0.0f, 0.0f), radians(down)))));
    child.dims = vec4(radius, parentLength, offset, lengthChildMax);
    child.ids = uvec4(treeType(s.ids) | (uint(c) << 16u), seed, s.ids.z, 0u);
    return child;
}

TREE_FN LeafRecord treeLeaf(TreeTypeParams p, StemRecord s, int k, uint cloneSeed, vec3 pos, vec4 rot TREE_CTX_PARAM) {
    uint seed = treeHash2(cloneSeed, TREE_KEY_CHILD + uint(k));
    float phi = float(k) * p.leafAngles.z + p.leafAngles.w * treeRandSigned(seed, TREE_KEY_ROTATE_V);
    float down = p.leafAngles.x + p.leafAngles.y * treeRandSigned(seed, TREE_KEY_DOWN_V);
    vec4 q = normalize(
        qmul(rot, qmul(qaxis(vec3(0.0f, 0.0f, 1.0f), radians(phi)), qaxis(vec3(1.0f, 0.0f, 0.0f), radians(down)))));
    TreeFrame frame = TREE_FRAME;
    if (frame.wind.w != 0.0f) {
        // Leaves flutter about their stalk.
        float t = frame.timeSeason.x;
        float flutter = sin(2.0f * TREE_PI * (2.5f * t + treeRand(seed, 0x91u)));
        q = normalize(qmul(q, qaxis(vec3(1.0f, 0.0f, 0.0f), 0.5f * frame.wind.w * flutter)));
    }
    TreeInstance inst = TREE_INSTANCE(s.ids.z);
    float scale = inst.posScale.w;
    vec3 state = treeLeafSeason(frame.timeSeason.y, treeRand(seed, 0x92u), p.leafColorFall.w, p.trunk.w > 0.5f);
    float leafLength = p.leafCount.y * scale * state.x;
    vec3 base = pos + qforward(q) * (p.leafLobes.w * scale);
    LeafRecord leaf;
    leaf.posScale = vec4(base.x, base.y, base.z, leafLength);
    leaf.rot = q;
    leaf.ids = uvec4(treeType(s.ids), seed, s.ids.z, treeLeafSeasonPack(state));
    return leaf;
}

// The work of thread t (0 <= t < TREE_WAVE_SIZE) for stem s: Kuth et al.'s
// Level node. All threads walk the stem's rings; at a split into clones the
// threads are divided into one block per clone, and every thread then follows
// its block's clone. The first thread of a block writes the clone's segments,
// and the block's threads take turns writing the children.
TREE_FN void treeGenerateStem(StemRecord s, uint t TREE_CTX_PARAM) {
    uint type = treeType(s.ids);
    int level = treeLevel(s.ids);
    TreeTypeParams p = TREE_PARAMS(type);
    vec3 up = vec3(TREE_FRAME.worldUp.x, TREE_FRAME.worldUp.y, TREE_FRAME.worldUp.z);
    if (TREE_FRAME.reserved.x > 0.5f) {
        // Each tree's own up, along its instance's trunk axis (turned copies baked for impostors).
        up = qforward(TREE_INSTANCE(s.ids.z).rot);
    }
    bool lastLevel = level >= int(p.ratios.x) - 1 || level >= TREE_MAX_LEVELS - 1;

    float stemLength = s.posLength.w;
    int segments = max(1, int(p.nCurveRes[level]));
    float segmentLength = stemLength / float(segments);

    // Children sit evenly along the stem, from the base size up for the trunk.
    float base = level == 0 ? p.shapeBase.y * stemLength : 0.0f;
    float countF = lastLevel ? treeLeafCount(p, s) : treeBranchCount(p, s, level);
    int children = int(floor(max(countF, 0.0f) + treeRand(s.ids.y, TREE_KEY_COUNT)));
    float childSpan = max(stemLength - base, 1e-6f);

    uint blockStart = 0u;
    uint blockSize = TREE_WAVE_SIZE;
    uint cloneSeed = s.ids.y;
    float cloneShare = 1.0f;
    float splitError = 0.0f;

    TreeFrame frame = TREE_FRAME;
    float scale = TREE_INSTANCE(s.ids.z).posScale.w;
    float typeLength = stemLength / scale;

    vec3 pos = vec3(s.posLength);
    vec4 rot = treeCurveRing(p, level, s.rot, 0, segments, cloneSeed, up);
    rot = treeWind(frame, rot, pos, treeStemRadius(p, s, 0.0f) / scale, typeLength, s.ids.y, segments);
    for (int i = 0; i < segments; ++i) {
        vec3 nextPos = pos + qforward(rot) * segmentLength;
        float z0 = float(i) / float(segments);
        float z1 = float(i + 1) / float(segments);
        vec4 nextRot = rot;
        if (i + 1 < segments) {
            nextRot = treeCurveRing(p, level, rot, i + 1, segments, cloneSeed, up);
            nextRot =
                treeWind(frame, nextRot, nextPos, treeStemRadius(p, s, z1) / scale, typeLength, s.ids.y, segments);
        }
        float r0 = treeStemRadius(p, s, z0);
        float r1 = treeStemRadius(p, s, z1);

        if (t == blockStart) {
            SegmentRecord d;
            d.fromPosRadius = vec4(pos.x, pos.y, pos.z, r0);
            d.fromRot = rot;
            d.toPosRadius = vec4(nextPos.x, nextPos.y, nextPos.z, r1);
            d.toRot = nextRot;
            d.shape = vec4(treeStemRadiusSlope(p, s, z0), treeStemRadiusSlope(p, s, z1), z0, stemLength);
            d.ids = uvec4(s.ids.x, cloneSeed, s.ids.z, uint(i));
            treeEmitSegment(d TREE_CTX_ARG);
        }

        // Children k with offsets base + (k + 0.5) / children * childSpan in this segment.
        float x0 = (float(i) * segmentLength - base) / childSpan * float(children) - 0.5f;
        float x1 = (float(i + 1) * segmentLength - base) / childSpan * float(children) - 0.5f;
        int kStart = clamp(int(ceil(x0)), 0, children);
        int kEnd = i + 1 == segments ? children : clamp(int(ceil(x1)), 0, children);
        vec3 m0 = qforward(rot) * segmentLength;
        vec3 m1 = qforward(nextRot) * segmentLength;
        for (int k = kStart + int(t - blockStart); k < kEnd; k += int(blockSize)) {
            if (cloneShare < 1.0f && treeRand(cloneSeed, TREE_KEY_SHARE + uint(k)) >= cloneShare) {
                continue;
            }
            float offset = base + (float(k) + 0.5f) / float(children) * childSpan;
            float f = clamp((offset - float(i) * segmentLength) / segmentLength, 0.0f, 1.0f);
            vec3 cp = treeHermite(pos, nextPos, m0, m1, f);
            vec4 cq = qnlerp(rot, nextRot, f);
            if (lastLevel) {
                LeafRecord leaf = treeLeaf(p, s, k, cloneSeed, cp, cq TREE_CTX_ARG);
                // Leaves of the season that are not there (yet or any more), or too small to see, are left out.
                if (leaf.posScale.w > 1e-3f * p.leafCount.y * TREE_INSTANCE(s.ids.z).posScale.w) {
                    treeEmitLeaf(leaf TREE_CTX_ARG);
                }
            } else {
                float parentRadius = treeStemRadius(p, s, offset / stemLength);
                treeEmitStem(treeChildStem(p, s, k, cloneSeed, offset, base, cp, cq, parentRadius) TREE_CTX_ARG);
            }
        }

        // Splitting into clones at the next ring, with error diffusion of the
        // fractional split count as in Weber and Penn.
        if (i + 1 < segments) {
            float wanted = p.nSegSplits[level] + splitError;
            int splits = int(floor(wanted));
            splitError = wanted - float(splits);
            uint clones = min(uint(splits + 1), blockSize);
            if (clones > 1u) {
                uint sub = blockSize / clones;
                uint c = min((t - blockStart) / sub, clones - 1u);
                blockStart += c * sub;
                blockSize = c == clones - 1u ? blockSize - c * sub : sub;
                cloneShare /= float(clones);
                float angle = p.nSplitAngle[level] +
                              p.nSplitAngleV[level] * treeRandSigned(cloneSeed, TREE_KEY_SPLIT_ANGLE + uint(i));
                float phi =
                    2.0f * TREE_PI * (treeRand(cloneSeed, TREE_KEY_SPLIT_PHI + uint(i)) + float(c) / float(clones));
                nextRot = normalize(qmul(
                    nextRot, qmul(qaxis(vec3(0.0f, 0.0f, 1.0f), phi), qaxis(vec3(1.0f, 0.0f, 0.0f), radians(angle)))));
                cloneSeed = treeHash2(cloneSeed, c + 1u);
            }
        }
        pos = nextPos;
        rot = nextRot;
    }
}

#endif // KLARTRAUM_TREE_TREE_GENERATE_HPP
