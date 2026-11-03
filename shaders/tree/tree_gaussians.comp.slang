// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// Turns the generated trees into 3D Gaussians for the splatting backends: one
// thread per Gaussian slot. The buffers start with pc.staticCount Gaussians
// uploaded once; then come pc.surfelCount surfels, surface samples of the
// static meshes that are lit anew every frame, so the trees' shadows fall on
// them; then the trees. Of the tree slots, [0, segmentCapacity * 36) belong to stem
// segments: 12 around times 3 along the tube, flat Gaussians on its surface.
// The leaf slots follow, 8 per leaf, along the axes of its lobes. Slots of
// records that were not generated get opacity 0. The colours carry the
// lighting of the classical renderer (treeShadeLit() with the light volume),
// as degree 0 SH.
//
// The level of detail follows the size on screen, continuously, so that nothing
// pops and the Gaussians the cull drops cost next to nothing:
// - A segment thinner than settings.gaussianLod.x pixels (radius) draws its
//   tube with one round Gaussian per step along it: the others move onto the
//   axis and fade out from gaussianLod.x to half of it. Segments thinner than
//   gaussianLod.y pixels fade out (to half of it).
// - Leaves merge their lobes as the classical renderer's (settings.leafLod.xy,
//   treeLeafLod()) and thin out below gaussianLod.z pixels, down to the share
//   gaussianLod.w: while their lobes merge, the 8 Gaussians move into one leaf-sized Gaussian
//   (their moments) that takes over their coverage as the others fade out;
//   once thinning out, the remaining ones cover for the
//   departed (more opaque, wider where needed) and the departing ones shrink
//   away.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 12
#include "tree_render_common.slang"
#include "tree_shading.slang"
#define TREE_LIGHT_SET 0
#define TREE_LIGHT_BINDING 13
#include "tree_light.slang"

[[vk::binding(0, 0)]] StructuredBuffer<SegmentRecord> segments;
[[vk::binding(1, 0)]] StructuredBuffer<LeafRecord> leaves;
[[vk::binding(2, 0)]] StructuredBuffer<uint> counters;
[[vk::binding(3, 0)]] StructuredBuffer<TreeTypeParams> params;
[[vk::binding(4, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(5, 0)]] RWScalarBuffer<vec3> outPos;
[[vk::binding(6, 0)]] RWScalarBuffer<vec4> outRot;
[[vk::binding(7, 0)]] RWScalarBuffer<vec3> outScale;
[[vk::binding(8, 0)]] RWScalarBuffer<vec4> outColAlpha;

[[vk::binding(14, 0)]] StructuredBuffer<TreeSceneVertex> surfels; // pos.w: the spacing of the samples
// The captured colours and positions of the static Gaussians, tinted every
// frame by settings.capturedTint and their shadows.
[[vk::binding(15, 0)]] ScalarBuffer<vec4> staticColors;
[[vk::binding(16, 0)]] ScalarBuffer<vec3> staticPositions;
[[vk::binding(17, 0)]] ScalarBuffer<vec3> staticNormals;

struct Push {
    uint segmentCapacity;
    uint leafCapacity;
    uint staticCount;
    uint surfelCount;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const uint STEM_AROUND = 12u;
static const uint STEM_ALONG = 3u;
static const uint STEM_GAUSSIANS = STEM_AROUND * STEM_ALONG;
static const uint LEAF_GAUSSIANS = 8u;
static const float LEAF_ALPHA = 0.9f;
static const float STEM_ALPHA = 0.95f;
static const float SH_C0 = 0.28209479f;

vec3 colorToSh0(vec3 c) {
    return (c - vec3(0.5f)) / SH_C0;
}

// The colour as the classical renderer outputs it, for a surface at pos with
// this normal; the pot (glaze) takes the light through the trees only, as there.
vec3 displayColor(vec3 albedo, vec3 pos, vec3 normal, bool glaze) {
    TreeRenderSettings s = settings[0];
    vec3 at = pos + 0.5f * s.lightGridMin.w * normal;
    vec2 light = glaze ? treeLightOfTreesAt(s, at) : treeLightAt(s, at);
    vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, pos, s);
    return treeOutputColor(treeAtmosphere(color, pos, camera.cameraWorldPos.xyz, s), s).rgb;
}

// Slots count from the first Gaussian after the static ones.
void writeGaussian(uint slot, vec3 pos, vec4 rot, vec3 scale, vec3 color, float alpha) {
    uint i = pc.staticCount + slot;
    outPos[i] = pos;
    outRot[i] = rot;
    outScale[i] = scale;
    outColAlpha[i] = vec4(colorToSh0(color), alpha);
}

void hide(uint slot) {
    uint i = pc.staticCount + slot;
    outPos[i] = vec3(0.0f);
    outRot[i] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    outScale[i] = vec3(1e-6f);
    outColAlpha[i] = vec4(0.0f);
}

void stemGaussian(uint i, uint segment, uint local) {
    SegmentRecord d = segments[segment];
    TreeRenderSettings s = settings[0];
    float pixels = max(treeRingPixelRadius(stemFrom(d), s), treeRingPixelRadius(stemTo(d), s));
    // Thinner than gaussianLod.y: fading out to half of it.
    float alpha = STEM_ALPHA * clamp(2.0f * pixels / max(s.gaussianLod.y, 1e-6f) - 1.0f, 0.0f, 1.0f);
    // How far the tube has become one round Gaussian per step along it.
    float roundPixels = s.gaussianLod.x;
    float round = roundPixels > 0.0f ? clamp(2.0f - 2.0f * pixels / roundPixels, 0.0f, 1.0f) : 0.0f;
    uint k = local % STEM_AROUND;
    if (k != 0u) {
        alpha *= 1.0f - round;
    }
    if (alpha < 1.0f / 255.0f) {
        hide(i);
        return;
    }
    uint m = local / STEM_AROUND;
    float w = (float(m) + 0.5f) / float(STEM_ALONG);
    float segmentLength = length(d.toPosRadius.xyz - d.fromPosRadius.xyz);
    vec3 center = treeHermite(d.fromPosRadius.xyz, d.toPosRadius.xyz, qforward(d.fromRot) * segmentLength,
                              qforward(d.toRot) * segmentLength, w);
    vec4 ring = qnlerp(d.fromRot, d.toRot, w);
    float radius = mix(d.fromPosRadius.w, d.toPosRadius.w, w);
    float phi = 2.0f * TREE_PI * (float(k) + 0.5f * float(m)) / float(STEM_AROUND);
    // Local axes: x radial, y around, z along the stem.
    vec4 rot = normalize(qmul(ring, qaxis(vec3(0.0f, 0.0f, 1.0f), phi)));
    vec3 radial = qrot(rot, vec3(1.0f, 0.0f, 0.0f));
    float around = TREE_PI * radius / float(STEM_AROUND);
    float along = 0.35f * segmentLength / float(STEM_ALONG);
    vec3 scale = vec3(0.3f * around, 1.2f * around, max(along, around));
    TreeTypeParams p = params[treeType(d.ids)];
    // The mean of the classical renderer's bark relief.
    vec3 albedo = 0.9f * p.barkColor.rgb;
    float z = d.shape.z + w * segmentLength / d.shape.w;
    vec3 pos = center + (1.0f - round) * 0.85f * radius * treeTrunkLobes(p, treeLevel(d.ids), z, phi) * radial;
    // The round Gaussian covers the tube's width (its integral across, sqrt(2 pi) sigma alpha, is 2 radius) and
    // joins its neighbours along the segment (sigma 0.6 times their spacing).
    vec3 roundScale = vec3(0.84f * radius, 0.84f * radius, max(0.6f * segmentLength / float(STEM_ALONG), scale.z));
    scale = mix(scale, roundScale, k == 0u ? round : 0.0f);
    // The round Gaussian shows the visible half of the tube: lit as the side facing the camera.
    vec3 normal = radial;
    if (k == 0u && round > 0.0f) {
        vec3 axis = qforward(ring);
        vec3 toCamera = camera.cameraWorldPos.xyz - center;
        vec3 facing = toCamera - dot(toCamera, axis) * axis;
        if (dot(facing, facing) > 1e-12f) {
            normal = normalize(mix(radial, normalize(facing), round));
        }
    }
    albedo = mix(albedo, vec3(0.8f), treeSnowOn(normal, pos, settings[0].season.y));
    writeGaussian(i, pos, rot, scale, displayColor(albedo, pos, normal, false), alpha);
}

// What a Gaussian of opacity `alpha` and in-plane sigmas `sx`, `sz` (leaf lengths) covers on screen, in square
// pixels, for a leaf `pixels` long: the splatting widens every Gaussian's covariance by 0.3 square pixels (its
// anti-aliasing) without lowering its opacity, so sub-pixel Gaussians cover more than their size.
float leafCoverPixels(float sx, float sz, float alpha, float pixels) {
    return 2.0f * TREE_PI * alpha * sqrt((sx * sx * pixels * pixels + 0.3f) * (sz * sz * pixels * pixels + 0.3f));
}

// The 8 Gaussians of a leaf of length 1 merged into one, in the leaf's frame (x across, z along), as the
// splatting shows them for a leaf `pixels` long: its middle (xz of placement), the angle of its long axis about
// the normal (placement.z) and its scale match their moments as widened by the splatting's anti-aliasing,
// weighted by what each covers; once drawn (and widened in turn) the merged Gaussian spreads as they do.
// w of scale: the opacity with which it covers `boost` times as much as they do (a leaf that stands in for others
// thinned out covers theirs too). They lie within the shape of their moments,
// whose area is twice the integral of a Gaussian with those moments (its footprint F), and overlap there as if
// scattered: they cover 2 F (1 - exp(-their coverage / 2 F)). Widened where that would take more than 0.95.
TreeLeafGaussian mergedLeafGaussian(TreeTypeParams p, TreeLeafOutline o, float pixels, float boost) {
    // The anti-aliasing's 0.3 square pixels in square leaf lengths.
    float widen = 0.3f / max(pixels * pixels, 1e-8f);
    float weight = 0.0f;
    vec2 mean = vec2(0.0f);
    vec3 second = vec3(0.0f); // xx, xz, zz
    for (uint k = 0u; k < LEAF_GAUSSIANS; ++k) {
        TreeLeafGaussian g = treeLeafGaussianShape(p, o, k, LEAF_GAUSSIANS);
        float wk = leafCoverPixels(g.scale.x, g.scale.z, LEAF_ALPHA, pixels);
        float c = cos(g.placement.z), sn = sin(g.placement.z);
        // The piece's covariance turned by its angle about the normal (y): x' = c x + s z, z' = -s x + c z,
        // widened as the splatting widens it.
        float sx2 = g.scale.x * g.scale.x, sz2 = g.scale.z * g.scale.z;
        vec3 cov = vec3(c * c * sx2 + sn * sn * sz2 + widen, c * sn * (sz2 - sx2), sn * sn * sx2 + c * c * sz2 + widen);
        vec2 m = vec2(g.placement.x, g.placement.y);
        weight += wk;
        mean += wk * m;
        second += wk * (cov + vec3(m.x * m.x, m.x * m.y, m.y * m.y));
    }
    mean /= weight;
    vec3 cov = second / weight - vec3(mean.x * mean.x, mean.x * mean.y, mean.y * mean.y);
    // Eigen decomposition of [[xx, xz], [xz, zz]]: the long axis's angle from z towards x.
    float middle = 0.5f * (cov.x + cov.z);
    float spread = sqrt(max(0.25f * (cov.x - cov.z) * (cov.x - cov.z) + cov.y * cov.y, 0.0f));
    float major = middle + spread, minor = middle - spread;
    float angle = 0.5f * atan(2.0f * cov.y, cov.z - cov.x);
    TreeLeafGaussian merged;
    merged.placement = vec4(mean.x, mean.y, angle, 0.0f);
    // Its own size: the splatting widens it again.
    float sx = sqrt(max(minor - widen, 1e-8f)), sz = sqrt(max(major - widen, 1e-8f));
    float footprint = leafCoverPixels(sx, sz, 1.0f, pixels);
    // The leaves it covers for lie elsewhere: their coverage adds.
    float covered = boost * 2.0f * footprint * (1.0f - exp(-weight / (2.0f * footprint)));
    float alpha = covered / footprint;
    if (alpha > 0.95f) {
        // Wider until it covers as much at 0.95 (its covered area grows about with the square of the factor).
        float grow = 1.0f;
        for (int stepIndex = 0; stepIndex < 4; ++stepIndex) {
            grow *= sqrt(covered / leafCoverPixels(grow * sx, grow * sz, 0.95f, pixels));
        }
        sx *= grow;
        sz *= grow;
        alpha = 0.95f;
    }
    merged.scale = vec4(sx, 0.01f, sz, alpha);
    return merged;
}

void leafGaussian(uint i, uint leafIndex, uint local) {
    LeafRecord leaf = leaves[leafIndex];
    TreeTypeParams p = params[treeType(leaf.ids)];
    TreeRenderSettings s = settings[0];
    // As the classical renderer's leaves; the random number below 0.9, so that all leaves are kept in full
    // where none thin out.
    vec3 middle = leaf.posScale.xyz + 0.5f * leaf.posScale.w * qforward(leaf.rot);
    float pixels = leaf.posScale.w * treeFocalPixels(s) / max(treeViewDepth(middle), 1e-4f);
    // Merging as leafLod.xy, thinning out as gaussianLod.zw.
    vec4 leafLod = vec4(s.leafLod.xy, s.gaussianLod.zw);
    vec2 lod = treeLeafLod(pixels, leafLod, 0.9f * treeRand(leaf.ids.y, 0x77u));
    // Of the leaves kept, each covers for the ones thinned out; departing leaves shrink away. Leaves thin out
    // only once their lobes have merged (gaussianLod.z below leafLod.y).
    float keep = clamp(pixels / max(leafLod.z, 1e-3f), leafLod.w, 1.0f);
    float shrink = lod.y * sqrt(keep);
    float alpha = LEAF_ALPHA;
    if (local != 0u) {
        alpha *= 1.0f - lod.x;
    }
    if (shrink <= 0.0f || alpha < 1.0f / 255.0f) {
        hide(i);
        return;
    }
    TreeLeafOutline o = treeLeafOutline(p.leafShape, 0.5f * p.leafCount.z);
    TreeLeafGaussian shape = treeLeafGaussianShape(p, o, local, LEAF_GAUSSIANS);
    if (lod.x > 0.0f) {
        TreeLeafGaussian merged = mergedLeafGaussian(p, o, pixels, 1.0f / keep);
        shape.placement.xy = mix(shape.placement.xy, merged.placement.xy, lod.x);
        shape.placement.z = mix(shape.placement.z, merged.placement.z, lod.x);
        shape.scale.xyz = mix(shape.scale.xyz, merged.scale.xyz, lod.x);
        if (local == 0u) {
            alpha = mix(LEAF_ALPHA, merged.scale.w, lod.x);
        }
    }
    vec2 q = vec2(shape.placement.x, shape.placement.y);
    float angle = shape.placement.z;

    vec3 x = qrot(leaf.rot, vec3(1.0f, 0.0f, 0.0f));
    vec3 z = qforward(leaf.rot);
    // A departing leaf shrinks away towards its stalk.
    float size = leaf.posScale.w * shrink;
    vec3 pos = leaf.posScale.xyz + size * (q.x * x + q.y * z);
    // Local axes: x across the blade, y the leaf normal, z along the lobe.
    vec4 rot = normalize(qmul(leaf.rot, qaxis(vec3(0.0f, 1.0f, 0.0f), angle)));
    vec3 scale = size * vec3(shape.scale);

    vec3 normal = qrot(leaf.rot, vec3(0.0f, 1.0f, 0.0f));
    if (dot(normal, settings[0].sun.xyz) < 0.0f) {
        normal = -normal;
    }
    vec3 albedo = treeLeafSeasonAlbedo(p, leaf.ids.y, treeHash(leaf.ids.z), leaf.ids.w);
    albedo = mix(albedo, vec3(0.8f), 0.7f * treeSnowOn(normal, pos, settings[0].season.y));
    writeGaussian(i, pos, rot, scale, displayColor(albedo, pos, normal, false), alpha);
}

// A flat Gaussian on a static surface, its thin axis along the normal.
void surfelGaussian(uint slot, uint index) {
    TreeSceneVertex v = surfels[index];
    vec3 normal = normalize(v.normal.xyz);
    float spin = 2.0f * TREE_PI * treeRand(index, 0x79u);
    vec4 rot = normalize(qmul(qfromZ(normal), qaxis(vec3(0.0f, 0.0f, 1.0f), spin)));
    float spacing = v.pos.w;
    vec3 scale = vec3(0.9f * spacing, 0.9f * spacing, 0.08f * spacing);
    if (v.color.w == TREE_MATERIAL_CAPTURED) {
        // Tinted like the captured Gaussians (see main()).
        TreeRenderSettings s = settings[0];
        float shade = treeCapturedShadeAt(s, v.pos.xyz, normal);
        writeGaussian(slot, v.pos.xyz, rot, scale, treeCapturedColor(v.color.rgb, normal, v.pos.xyz, shade, s), 0.95f);
        return;
    }
    // Finer than the surfels' spacing, the structure is averaged.
    vec3 albedo = treeSceneAlbedoDetail(v.color, v.pos.xyz, 0.0f);
    albedo = mix(albedo, vec3(0.8f), treeSnowOn(normal, v.pos.xyz, settings[0].season.y));
    writeGaussian(slot, v.pos.xyz, rot, scale, displayColor(albedo, v.pos.xyz, normal, v.color.w == TREE_MATERIAL_GLAZE),
                  0.95f);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g < pc.staticCount) {
        vec4 c = staticColors[g];
        TreeRenderSettings s = settings[0];
        // The side that is seen: the Gaussian's thinnest axis turned towards the camera, counting as far as the
        // Gaussian is flat (the length of staticNormals); a round Gaussian's side is unknown.
        vec3 axis = staticNormals[g];
        float flatness = length(axis);
        vec3 toCamera = camera.cameraWorldPos.xyz - staticPositions[g];
        vec3 seen = flatness > 0.0f ? axis / flatness : vec3(0.0f, 1.0f, 0.0f);
        seen = dot(seen, toCamera) < 0.0f ? -seen : seen;
        float shade = treeCapturedShadeAt(s, staticPositions[g], flatness * seen);
        vec3 displayed = treeCapturedColor(vec3(0.5f) + SH_C0 * c.rgb, seen, staticPositions[g], shade, s);
        outColAlpha[g] = vec4((displayed - vec3(0.5f)) / SH_C0, c.a);
        return;
    }
    uint slot = g - pc.staticCount;
    if (slot < pc.surfelCount) {
        surfelGaussian(slot, slot);
        return;
    }
    uint i = slot - pc.surfelCount;
    uint stemSlots = pc.segmentCapacity * STEM_GAUSSIANS;
    uint total = stemSlots + pc.leafCapacity * LEAF_GAUSSIANS;
    if (i >= total) {
        return;
    }
    if (i < stemSlots) {
        uint segment = i / STEM_GAUSSIANS;
        if (segment < min(counters[4], pc.segmentCapacity)) {
            stemGaussian(slot, segment, i % STEM_GAUSSIANS);
        } else {
            hide(slot);
        }
    } else {
        uint j = i - stemSlots;
        uint leaf = j / LEAF_GAUSSIANS;
        if (leaf < min(counters[5], pc.leafCapacity)) {
            leafGaussian(slot, leaf, j % LEAF_GAUSSIANS);
        } else {
            hide(slot);
        }
    }
}
