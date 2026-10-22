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
    vec4 trunk; // x lobes and y lobe depth of the trunk's base (Weber and Penn), z bark furrows per unit length,
                // w 1 for deciduous leaves (they follow the seasons), 0 for evergreen ones
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
    uvec4 ids;     // x type, y seed, z instance, w seasonal colour: bits 0-7 freshness, 8-15 fall colouring
};

// A vertex of the static scene meshes (pot, soil, table).
struct TreeSceneVertex {
    vec4 pos;    // xyz, w unused
    vec4 normal; // xyz, w unused
    vec4 color;  // rgb albedo, w material (TREE_MATERIAL_*)
};

// A Gaussian of a level-of-detail library (TreeMossLod), in its patch's frame.
struct TreeLodSplat {
    vec4 pos;   // xyz, w unused
    vec4 rot;   // quaternion (x, y, z, w)
    vec4 scale; // xyz, w unused
    vec4 color; // degree 0 SH coefficients rgb, w opacity
};

// One patch of a library: where its four levels start and how many Gaussians they have.
struct TreeLodPatch {
    uvec4 offset;
    uvec4 count;
    vec4 bounds; // x radius around the patch's origin, yzw unused
};

// One placed copy of a patch, lying on a plane.
struct TreeLodInstance {
    vec4 posYaw; // xyz the patch origin's position, w the turn about +y in radians
    vec4 plane;  // x dy/dx and y dy/dz of the ground, z scale, w unused
    uvec4 ids;   // x patch, y how far the copy has faded out, z how much of the sun the terrain takes away (both float
                 // bits; 0: nothing), w 1 when the patch's colours are albedos to light (trees), 0 for captured
                 // colours (moss)
};

// Values the whole generation reads.
struct TreeFrame {
    vec4 worldUp;    // xyz unit vector up, w unused
    vec4 wind;       // xyz direction the wind blows to, w strength
    vec4 timeSeason; // x time in seconds, y season in [0, 4), zw unused
    vec4 reserved;   // x 1: up is each instance's trunk axis instead of worldUp (for baking impostors), yzw unused
};

// Per-frame values of the tree renderers.
struct TreeRenderSettings {
    vec4 viewport; // x width, y height in pixels, z pixels per triangle edge (Delta), w minimum stem radius in pixels
    vec4 sun;      // xyz unit vector towards the sun, w intensity
    vec4 sky;      // rgb ambient light from above, w exposure of the tone curve (treeDisplayColor())
    vec4 ground;   // rgb ambient light from below, w unused
    // Continuous leaf level of detail, by the leaf's length in pixels: x where leaves start to flatten and
    // their lobes to merge, y where they are fully simplified, z below which leaves thin out, w the fraction
    // of leaves kept at the farthest distance.
    vec4 leafLod;
    // x: 1 to hide fragments behind the scene depth buffer (Gaussians drawn before, e.g. the lantern),
    // 0 to ignore it; y: what the trees' surfaces write (0 their colour; while baking impostors 1 their albedo,
    // 2 their normal, see treeSurfaceOutput()); zw unused.
    vec4 occlusion;
    // The light volume (TreeLightVolume): a grid of sun transmittance and sky visibility over the scene.
    vec4 lightGridMin;    // xyz the grid's lower corner, w the voxel size
    uvec4 lightGridSize;  // xyz voxels per axis, w 1 when the grid is used, 0 for no shadows
    vec4 lightGridParams; // x optical depth of a voxel per leaf area / voxel face area, yzw unused
    vec4 season;          // x season S as in TreeFrame, y snow cover in [0, 1] (treeSnowCover()), zw unused
    vec4 lightColor;      // rgb colour of the directional light (sun or moon), w unused
    vec4 pointLight;      // xyz position of a point light without shadows (the lantern's flame), w unused
    vec4 pointColor;      // rgb radiance of the point light at distance 1, w unused
    vec4 capturedTint;    // rgb factor on the colours of captured Gaussians (the lantern) for the current light
    // The miniature world's atmosphere (treeAtmosphere()): x metres per unit (0: none), y the visibility in metres
    // (the haze takes 63 % of a colour that far away), z how much thicker the haze is in the valleys, w unused.
    vec4 atmosphere;
    vec4 atmosphereBox; // the box the atmosphere fills: x, y its half length and width, z its bottom, w its top
    // x, y the tree sizes in pixels between which the forest canopy hands over to the trees (treeCanopyWeight());
    // per frame, so that the visible trees fit their buffers. 0: the terrain's own (TreeTerrainConfig). z, w the
    // same for the library copies' cut-off (TreeGaussianForest; 0: none): the canopy covers the larger band.
    vec4 canopyBand;
};

// The ground in the pot (treeTerrainHeight()): the soil with its mound, and the relief of a landscape that the
// scale gives (BonsaiScale::reliefUnits()), falling to the soil level towards the rim.
struct TreeTerrainParams {
    vec4 outline; // x half length (along x), y half width (along z) of the soil's outline, z its superellipse
                  // exponent, w unused
    vec4 soil;    // x soil height at the rim, y mound towards the middle, z amplitude of the bumps, w scale of the
                  // moss patterns of the forest floor (1 up to the scale at which they show real moss, larger beyond)
    vec4 relief;  // x height of the relief in units (0: none), y feature frequency per unit, z ridge character in
                  // [0, 1], w width of the band at the rim in which the relief falls to the soil (in r, see below)
    uvec4 seed;   // x seed of the landscape, yzw unused
    // Land cover (treeLandCover()): x forest threshold on the score (calibrated per scene for a forest share),
    // y metres per unit, z tree line in metres above the valley floors, w the relief's exaggeration (slopes and
    // altitudes are judged at true proportions).
    vec4 cover;
};

// Where trees grow (treeSiteAt()): one candidate per cell of a square grid over the soil.
struct TreePlacementParams {
    vec4 grid;    // x cell size in units, y jitter of a candidate within its cell (fraction of the cell), z chance of a
                  // shrub where the land cover is scrub, w metres per unit
    vec4 species; // x altitude in metres above the valley floors where conifers take over, y half the width of
                  // that change, z share of conifers in the valleys, w unused
    vec4 size;    // x mean height of the trees in metres, y their spread (fraction), z mean height of shrubs in
                  // metres, w unused
    uvec4 seed;   // x seed of the placement, yzw unused
};

// The terrain's quadtree for Continuous Distance-Dependent Level of Detail (CDLOD, Strugar 2009). Level 0 has
// the finest nodes, level levels.x - 1 is the root; a node of level L is twice the size of one of level L - 1.
struct TreeTerrainLod {
    vec4 root;    // xy the root node's lower corner (x, z), z its size, w unused
    vec4 ranges;  // x range factor: a node of size s is drawn up to x * s from the camera; y where its morph towards
                  // the next coarser level starts, as a fraction of the band between its range and the finer
                  // level's; z quads per node side; w unused
    uvec4 levels; // x levels of the quadtree, y levels of the height bounds pyramid, z capacity of selected nodes,
                  // w capacity of each selection queue
    vec4 reserved;
};

// What the terrain passes read: the ground, its quadtree, and the forest canopy that stands in for trees too small
// to draw one by one (treeCanopyWeight()).
struct TreeTerrainConfig {
    TreeTerrainParams terrain;
    TreeTerrainLod lod;
    TreePlacementParams placement; // of the trees whose canopy the terrain carries
    vec4 canopy; // x tree size in pixels below which the canopy stands in fully, y above which not at all, z unused,
                 // w 1 for the terrain's shadows of the settings' sun (treeTerrainSunVisibility());
                 // the canopy is off where x <= 0
};

// A cell's tree, if any.
struct TreeSite {
    vec4 posHeight; // xyz the stem's foot on the ground, w the height in units (0: no tree in this cell)
    vec4 kind;      // x species (TREE_SITE_*), y turn about +y in radians, z forest weight there (of a visible tree:
                    // the sun's visibility over the terrain), w unused (of a visible tree: how far it has faded in)
};

// The layout of the impostor atlases (TreeImpostorAtlas): per tree type and season sample, views x views views
// of pixels x pixels texels each, rows from the top.
struct TreeImpostorParams {
    uvec4 grid;     // x views per side of the hemi-octahedral grid, y texels per view side, z season samples, w types
    vec4 seasons;   // the seasons S of the samples, ascending (up to 4)
    vec4 bounds[2]; // per type: xyz the centre of the baked tree at scale 1 (its foot at the origin), w its radius
    vec4 heights;   // per type (x, y) the baked tree's height at scale 1, zw unused
};

// Grass near the camera (treeGrassCell(), TreeGrass): a clump of blades per cell of a grid in world metres.
struct TreeGrassParams {
    vec4 grid;    // x cell size in units, y radius in units beyond which there is no grass, z blades per clump near
                  // the camera, w metres per unit
    vec4 blade;   // x height in metres, y width in metres, zw unused
    uvec4 counts; // x cells per side of the grid around the camera, y clumps at most, z seed, w unused
};

// One clump of grass.
struct TreeGrassClump {
    vec4 posHeight; // xyz the clump's middle on the ground, w the blades' height in units
    vec4 kind;      // x blades, y turn in radians, z the sun's visibility over the terrain, w a random number
};

// A clump of grass that tree_grass.comp found visible this frame.
struct TreeGrassVisible {
    vec4 posHeight; // as TreeGrassClump
    vec4 kind;      // as TreeGrassClump, z the sun's visibility over the terrain
    vec4 ground;    // x, y the ground's slope along x and z, z the snow on it, w the clump's distance in units
    vec4 albedo;    // rgb the grass's albedo, w unused
};

// What tree_grass.comp reads.
struct TreeGrassSetup {
    TreeTerrainParams terrain;
    TreeGrassParams grass;
};

// The forest floor near the camera (treeFloorCell(), TreeForestFloor): a copy of a captured moss patch per cell of a
// grid in units where the forest grows.
struct TreeFloorSetup {
    TreeTerrainParams terrain;
    vec4 grid;    // x cell size in units, y radius in units beyond which there are no copies, z the copies' scale, w
                  // the share of the radius from which they fade out
    uvec4 counts; // x cells per side of the grid around the camera, y patches, z seed, w unused
};

// What the tile passes of the placement read (TreePlacementTiles).
struct TreeTileConfig {
    TreeTerrainParams terrain;
    TreePlacementParams placement;
    uvec4 tiles; // x cells per tile side, y slots of the pool, z tiles generated per frame at most, w unused
};

// Weights of the land-cover classes at a place, summing to 1.
struct TreeLandCover {
    vec4 vegetation; // x forest, y meadow, z field, w scrub
    vec4 bare;       // x rock, y water (reserved, always 0), zw unused
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

// The seed of the tree of a placed site turned by `yaw`: its own (treeHash2() of the scene's seed and the yaw), or
// with `variants` > 0 one of the seeds 1 to `variants` of the Gaussian tree library (TreeGaussianForest), so
// that a tree drawn in full and as its library copy is the same tree.
TREE_FN uint treeSiteSeed(uint seed, float yaw, uint variants) {
    uint own = treeHash2(seed, floatBitsToUint(yaw));
    return variants > 0u ? 1u + own % variants : own;
}

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

// The rotation that turns +z to the unit vector `to`.
TREE_FN vec4 qfromZ(vec3 to) {
    float d = to.z;
    if (d < -0.999999f) {
        return vec4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    return normalize(vec4(-to.y, to.x, 0.0f, 1.0f + d));
}

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
// an ambient term from the sky above and the ground below. light.x is the
// fraction of the sunlight that reaches the point, light.y the fraction of
// the sky it sees (both 1 without a light volume).
TREE_FN vec3 treeShadeLit(vec3 albedo, vec3 normal, TreeRenderSettings s, vec2 light) {
    float sun = max(dot(normal, vec3(s.sun.x, s.sun.y, s.sun.z)), 0.0f) * s.sun.w * light.x;
    float hemisphere = 0.5f + 0.5f * normal.y;
    vec3 ambient = mix(vec3(s.ground.x, s.ground.y, s.ground.z), vec3(s.sky.x, s.sky.y, s.sky.z), hemisphere);
    return albedo * (ambient * light.y + sun * vec3(s.lightColor.x, s.lightColor.y, s.lightColor.z));
}

// The light of the point light (TreeRenderSettings::pointLight) on a surface
// at pos, falling off with the squared distance.
TREE_FN vec3 treePointLight(vec3 albedo, vec3 normal, vec3 pos, TreeRenderSettings s) {
    vec3 toLight = vec3(s.pointLight.x, s.pointLight.y, s.pointLight.z) - pos;
    float d2 = dot(toLight, toLight);
    float lambert = max(dot(normal, toLight * inversesqrt(max(d2, 1e-8f))), 0.0f);
    return albedo * vec3(s.pointColor.x, s.pointColor.y, s.pointColor.z) * (lambert / (d2 + 0.01f));
}

TREE_FN vec3 treeShade(vec3 albedo, vec3 normal, TreeRenderSettings s) {
    return treeShadeLit(albedo, normal, s, vec2(1.0f, 1.0f));
}

// The albedo of one leaf: the type's colour, tinted per tree (treeSeed) a
// little and per leaf (leafSeed) more, between yellower and bluer green, and
// lighter or darker. The mean over many leaves stays near `base`.
TREE_FN vec3 treeLeafAlbedo(vec3 base, uint leafSeed, uint treeSeed) {
    float treeTint = treeRand(treeSeed, 0x80u) - 0.5f;
    float hue = treeRand(leafSeed, 0x78u) - 0.5f;
    float value = treeRand(leafSeed, 0x7au) - 0.5f;
    vec3 yellower = vec3(1.25f, 1.08f, 0.6f);
    vec3 bluer = vec3(0.8f, 0.97f, 1.3f);
    vec3 tint = mix(bluer, yellower, clamp(0.5f + 0.9f * hue + 0.5f * treeTint, 0.0f, 1.0f));
    return base * tint * (1.0f + 0.5f * value);
}

// The underside of a leaf: paler and greyer than its top.
TREE_FN vec3 treeLeafUnderside(vec3 top) {
    float luma = dot(top, vec3(0.3f, 0.59f, 0.11f));
    return 1.3f * mix(top, vec3(luma, luma, luma), 0.35f);
}

// How much of the light reaches a captured Gaussian, whose colours already
// contain the light of the capture: the share of the current sky and sun
// (weighted by their brightness) that the light volume lets through.
TREE_FN float treeCapturedShade(TreeRenderSettings s, vec2 light) {
    vec3 luma = vec3(0.3f, 0.59f, 0.11f);
    float ambient = dot(vec3(s.sky.x, s.sky.y, s.sky.z), luma);
    float sun = 0.5f * s.sun.w * dot(vec3(s.lightColor.x, s.lightColor.y, s.lightColor.z), luma);
    return (ambient * light.y + sun * light.x) / max(ambient + sun, 1e-4f);
}

// The haze between eye and pos in the miniature world (TreeRenderSettings::atmosphere): rgb its colour (linear),
// a how much of the surface's colour it replaces. Only the part of the view ray inside the world's box counts (a camera
// outside the pot looks through clear air until it enters), in the world's metres; the haze thickens towards the valley
// floor (the box's bottom) and takes the colour of the sky, brightened by the sun. treeAtmosphere() applies it to a
// linear colour.
TREE_FN vec4 treeHaze(vec3 pos, vec3 eye, TreeRenderSettings s) {
    if (s.atmosphere.x <= 0.0f) {
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    vec3 lo = vec3(-s.atmosphereBox.x, s.atmosphereBox.z, -s.atmosphereBox.y);
    vec3 hi = vec3(s.atmosphereBox.x, s.atmosphereBox.w, s.atmosphereBox.y);
    vec3 d = pos - eye;
    float enter = 0.0f;
    float leave = 1.0f;
    for (int a = 0; a < 3; ++a) {
        if (abs(d[a]) < 1e-8f) {
            if (eye[a] < lo[a] || eye[a] > hi[a]) {
                return vec4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        } else {
            float t1 = (lo[a] - eye[a]) / d[a];
            float t2 = (hi[a] - eye[a]) / d[a];
            enter = max(enter, min(t1, t2));
            leave = min(leave, max(t1, t2));
        }
    }
    float meters = max(leave - enter, 0.0f) * length(d) * s.atmosphere.x;
    float above = max(pos.y - s.atmosphereBox.z, 0.0f) * s.atmosphere.x;
    float density = 1.0f + s.atmosphere.z * exp(-above / 150.0f);
    float haze = 1.0f - exp(-meters * density / s.atmosphere.y);
    vec3 sky = vec3(s.sky.x, s.sky.y, s.sky.z);
    vec3 light = vec3(s.lightColor.x, s.lightColor.y, s.lightColor.z);
    vec3 color = 1.4f * sky + 0.25f * light * s.sun.w * max(s.sun.y, 0.0f);
    return vec4(color.x, color.y, color.z, haze);
}

TREE_FN vec3 treeAtmosphere(vec3 color, vec3 pos, vec3 eye, TreeRenderSettings s) {
    vec4 haze = treeHaze(pos, eye, s);
    return mix(color, vec3(haze.x, haze.y, haze.z), haze.w);
}

// Index of voxel (x, y, z) in the light volume.
TREE_FN uint treeLightIndex(uint x, uint y, uint z, uvec4 size) { return (z * size.y + y) * size.x + x; }

// The colour as the render targets store it: exposed with an exponential
// tone curve, so bright light (snow in the sun) keeps its shading instead of
// clipping, and with gamma 2.2.
TREE_FN vec3 treeDisplayColor(vec3 linearColor, float exposure) {
    vec3 c = max(linearColor, vec3(0.0f, 0.0f, 0.0f)) * exposure;
    vec3 mapped = vec3(1.0f, 1.0f, 1.0f) - exp(-c);
    float g = 1.0f / 2.2f;
    return pow(mapped, vec3(g, g, g));
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

// --- terrain ---------------------------------------------------------------

// Where (x, z) lies relative to the soil's outline: 0 in the middle, 1 on the outline and beyond.
TREE_FN float treeTerrainRadius(TreeTerrainParams t, float x, float z) {
    float e = t.outline.z;
    float m = pow(abs(x / t.outline.x), e) + pow(abs(z / t.outline.y), e);
    return min(pow(m, 1.0f / e), 1.0f);
}

// Gradient noise (Perlin) in about [-1, 1] on the integer lattice of the plane, a different field per seed.
// Its zero crossings form connected networks of lines, which ridged noise turns into mountain ridges.
TREE_FN float treeGradientNoise(vec2 p, uint seed) {
    vec2 i = floor(p);
    vec2 f = p - i;
    vec2 u = f * f * f * (f * (f * 6.0f - vec2(15.0f, 15.0f)) + vec2(10.0f, 10.0f));
    uint x = uint(int(i.x));
    uint y = uint(int(i.y));
    float a00 = 2.0f * TREE_PI * treeRand(treeHash2(seed, x), y);
    float a10 = 2.0f * TREE_PI * treeRand(treeHash2(seed, x + 1u), y);
    float a01 = 2.0f * TREE_PI * treeRand(treeHash2(seed, x), y + 1u);
    float a11 = 2.0f * TREE_PI * treeRand(treeHash2(seed, x + 1u), y + 1u);
    float n00 = cos(a00) * f.x + sin(a00) * f.y;
    float n10 = cos(a10) * (f.x - 1.0f) + sin(a10) * f.y;
    float n01 = cos(a01) * f.x + sin(a01) * (f.y - 1.0f);
    float n11 = cos(a11) * (f.x - 1.0f) + sin(a11) * (f.y - 1.0f);
    return 1.414f * mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}

// Fractal sums of gradient noise at p: x rounded hills, y ridges. Each octave is twice as fine as the one
// before and turned by 37 degrees, which hides the noise lattice; amplitudes fall by 0.32 per octave, so the
// fine octaves roughen the slopes without steepening them much. Ridges are finer on high ground than in the
// valleys (ridged multifractal, Musgrave), as erosion leaves them.
TREE_FN vec2 treeTerrainOctaves(vec2 p, uint seed) {
    float rounded = 0.0f;
    float ridged = 0.0f;
    float amplitude = 0.5f;
    float weight = 1.0f;
    for (int i = 0; i < 6; ++i) {
        float n = treeGradientNoise(p, treeHash2(seed, uint(i)));
        rounded += amplitude * n;
        float ridge = 1.0f - abs(n);
        ridge *= ridge;
        ridged += amplitude * ridge * weight;
        weight = clamp(1.5f * ridge, 0.0f, 1.0f);
        amplitude *= 0.32f;
        p = vec2(1.6f * p.x - 1.2f * p.y, 1.2f * p.x + 1.6f * p.y);
    }
    return vec2(rounded, ridged);
}

// The landscape's shape in [0, 1] at (x, z) in units: valleys near 0, peaks near 1. Fractal noise on a
// domain-warped plane, blended from rounded hills to ridges by t.relief.z, with a network of valleys with flat
// floors carved into it.
TREE_FN float treeTerrainShape(TreeTerrainParams t, float x, float z) {
    uint seed = treeHash2(t.seed.x, 0x7e44a1u);
    vec2 p = vec2(x, z) * t.relief.y;
    // Domain warping bends the features, so ridges and valleys meander.
    vec2 q = 0.5f * p;
    p = p + 0.6f * vec2(treeGradientNoise(q, seed + 101u), treeGradientNoise(q, seed + 211u));
    vec2 sums = treeTerrainOctaves(p, seed);
    // Stretched to [0, 1] from their 1st to 99th percentiles (measured over many seeds).
    float h = mix(smoothstep(-0.36f, 0.35f, sums.x), smoothstep(0.09f, 0.68f, sums.y), t.relief.z);
    // Valleys: along the lines where a coarse noise crosses 0, the land sinks to a flat floor.
    float v = treeGradientNoise(0.4f * p + vec2(2.3f, 4.7f), seed + 307u);
    float valley = smoothstep(0.02f, 0.55f, abs(v));
    return h * (0.15f + 0.85f * valley);
}

// Height of the landscape above the soil at (x, z), falling to 0 in the band at the rim.
TREE_FN float treeTerrainRelief(TreeTerrainParams t, float x, float z) {
    if (t.relief.x <= 0.0f) {
        return 0.0f;
    }
    float fade = 1.0f - smoothstep(1.0f - t.relief.w, 1.0f, treeTerrainRadius(t, x, z));
    return t.relief.x * fade * treeTerrainShape(t, x, z);
}

// Height of the soil at (x, z), without the landscape's relief: its mound and bumps.
TREE_FN float treeTerrainSoil(TreeTerrainParams t, float x, float z) {
    float r = treeTerrainRadius(t, x, z);
    float bumps = t.soil.z * (treeFbm(vec3(x * 3.0f, 0.0f, z * 3.0f)) - 0.5f);
    return t.soil.x + t.soil.y * (1.0f - r * r) + bumps * (1.0f - r);
}

// Height of the ground at (x, z) inside the pot: the soil with its mound and bumps, and the landscape's relief.
TREE_FN float treeTerrainHeight(TreeTerrainParams t, float x, float z) {
    return treeTerrainSoil(t, x, z) + treeTerrainRelief(t, x, z);
}

// How much of the sun reaches pos over the terrain: 1 in the open, 0 behind a ridge, soft between (a penumbra
// widening with the distance to what casts it, about 5 degrees). Marches 16 steps towards the sun, from 0.004 to
// about 1 unit away, comparing the ray's height with the ground's.
TREE_FN float treeTerrainSunVisibility(TreeTerrainParams t, vec3 pos, vec3 sun) {
    float across = sqrt(sun.x * sun.x + sun.z * sun.z);
    if (t.relief.x <= 0.0f || sun.y <= 0.0f || across < 1e-4f) {
        return 1.0f;
    }
    vec2 dir = vec2(sun.x, sun.z) / across;
    float rise = sun.y / across;
    float visibility = 1.0f;
    float d = 0.004f;
    for (int i = 0; i < 16; ++i) {
        vec2 p = vec2(pos.x, pos.z) + d * dir;
        if (treeTerrainRadius(t, p.x, p.y) >= 1.0f) {
            break;
        }
        float above = pos.y + d * rise - treeTerrainHeight(t, p.x, p.y);
        visibility = min(visibility, clamp(0.5f + above / (0.09f * d), 0.0f, 1.0f));
        d *= 1.45f;
    }
    return visibility;
}

// Altitude in metres above the valley floors, at true proportions, of a point on the ground.
TREE_FN float treeTerrainAltitude(TreeTerrainParams t, vec3 pos) {
    return (pos.y - treeTerrainSoil(t, pos.x, pos.z)) / t.cover.w * t.cover.y;
}

// --- impostors -------------------------------------------------------------------

// Hemi-octahedral map of the directions of the upper hemisphere (y up) to [-1, 1]^2; directions below the
// horizon are taken at the horizon.
TREE_FN vec2 treeHemiOctEncode(vec3 d) {
    float y = max(d.y, 0.0f);
    float n = abs(d.x) + y + abs(d.z);
    float px = d.x / n;
    float pz = d.z / n;
    return vec2(px + pz, px - pz);
}

TREE_FN vec3 treeHemiOctDecode(vec2 uv) {
    float px = 0.5f * (uv.x + uv.y);
    float pz = 0.5f * (uv.x - uv.y);
    return normalize(vec3(px, max(1.0f - abs(px) - abs(pz), 0.0f), pz));
}

// The axes of an impostor's card seen from direction d (towards the viewer): right and up on the screen, up as
// close to +y as d allows (+z looking straight down).
TREE_FN vec3 treeImpostorRight(vec3 d) {
    vec3 hint = abs(d.y) > 0.999f ? vec3(0.0f, 0.0f, -1.0f) : vec3(0.0f, 1.0f, 0.0f);
    return normalize(cross(hint, d));
}

TREE_FN vec3 treeImpostorUp(vec3 d) { return cross(d, treeImpostorRight(d)); }

// --- terrain level of detail ---------------------------------------------------

#define TREE_TERRAIN_CULL 0u
#define TREE_TERRAIN_DRAW 1u
#define TREE_TERRAIN_SPLIT 2u

TREE_FN float treeTerrainNodeSize(TreeTerrainLod lod, uint level) {
    return lod.root.z * exp2(float(level) - float(lod.levels.x - 1u));
}

// How far from the camera nodes of `level` are drawn.
TREE_FN float treeTerrainRange(TreeTerrainLod lod, uint level) {
    return lod.ranges.x * treeTerrainNodeSize(lod, level);
}

// The factor on distances for the terrain's level of detail with the projection `proj`: 1 for a vertical field of
// view of 40 degrees, less for a narrower one (a telephoto view resolves more detail at the same distance).
TREE_FN float treeTerrainZoom(mat4 proj) { return 1.0f / (abs(proj[1][1]) * 0.36397f); }

// How far a vertex at distance `dist` (times treeTerrainZoom()) of a node of `level` has morphed towards the next
// coarser level, in [0, 1].
TREE_FN float treeTerrainMorph(TreeTerrainLod lod, uint level, float dist) {
    float end = treeTerrainRange(lod, level);
    float start = mix(0.5f * end, end, lod.ranges.y);
    return clamp((dist - start) / (end - start), 0.0f, 1.0f);
}

// Index of the height bounds of node (ix, iz) of `level` in the pyramid: level p of the pyramid has 2^p x 2^p
// cells (rows along x) and starts at (4^p - 1) / 3; nodes finer than the pyramid use their ancestor's cell.
TREE_FN uint treeTerrainBoundsIndex(TreeTerrainLod lod, uint ix, uint iz, uint level) {
    uint depth = lod.levels.x - 1u - level;
    uint p = min(depth, lod.levels.y - 1u);
    uint shift = depth - p;
    return ((1u << (2u * p)) - 1u) / 3u + (iz >> shift) * (1u << p) + (ix >> shift);
}

// Distance from `eye` to the box [lo, hi].
TREE_FN float treeBoxDistance(vec3 eye, vec3 lo, vec3 hi) {
    return length(max(max(lo - eye, eye - hi), vec3(0.0f, 0.0f, 0.0f)));
}

// Whether the box [lo, hi] lies entirely on the negative side of `plane`.
TREE_FN bool treeBoxBehindPlane(vec4 plane, vec3 lo, vec3 hi) {
    vec3 far = vec3(plane.x >= 0.0f ? hi.x : lo.x, plane.y >= 0.0f ? hi.y : lo.y, plane.z >= 0.0f ? hi.z : lo.z);
    return dot(vec3(plane.x, plane.y, plane.z), far) + plane.w < 0.0f;
}

// Whether the box [lo, hi] is outside the view of `viewProj` (Vulkan clip space), by its four side planes and the
// near plane.
TREE_FN bool treeBoxOutsideFrustum(mat4 m, vec3 lo, vec3 hi) {
    vec4 r0 = vec4(m[0][0], m[1][0], m[2][0], m[3][0]);
    vec4 r1 = vec4(m[0][1], m[1][1], m[2][1], m[3][1]);
    vec4 r2 = vec4(m[0][2], m[1][2], m[2][2], m[3][2]);
    vec4 r3 = vec4(m[0][3], m[1][3], m[2][3], m[3][3]);
    return treeBoxBehindPlane(r3 + r0, lo, hi) || treeBoxBehindPlane(r3 - r0, lo, hi) ||
           treeBoxBehindPlane(r3 + r1, lo, hi) || treeBoxBehindPlane(r3 - r1, lo, hi) || treeBoxBehindPlane(r2, lo, hi);
}

// Whether the sphere around `center` is outside the view of `viewProj`, by its four side planes and the near plane.
TREE_FN bool treeSphereOutsideFrustum(mat4 m, vec3 center, float radius) {
    vec4 r0 = vec4(m[0][0], m[1][0], m[2][0], m[3][0]);
    vec4 r1 = vec4(m[0][1], m[1][1], m[2][1], m[3][1]);
    vec4 r2 = vec4(m[0][2], m[1][2], m[2][2], m[3][2]);
    vec4 r3 = vec4(m[0][3], m[1][3], m[2][3], m[3][3]);
    vec4 planes[5] = {r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2};
    for (int i = 0; i < 5; ++i) {
        vec4 p = planes[i] / length(vec3(planes[i]));
        if (dot(vec3(p), center) + p.w < -radius) {
            return true;
        }
    }
    return false;
}

// Whether the rectangle [lo, hi] in (x, z) reaches into the soil's outline.
TREE_FN bool treeTerrainRectInSoil(TreeTerrainParams t, vec2 lo, vec2 hi) {
    vec2 nearest = clamp(vec2(0.0f, 0.0f), lo, hi);
    return treeTerrainRadius(t, nearest.x, nearest.y) < 1.0f;
}

// What the selection does with node (ix, iz) of `level`, whose heights lie in `heights`: cull it (outside the
// soil or the view), draw it, or split it into its four children because the finer level's range reaches it.
// Distances count times `zoom` (treeTerrainZoom()).
TREE_FN uint treeTerrainNodeAction(TreeTerrainConfig c, mat4 viewProj, vec3 eye, float zoom, uint ix, uint iz,
                                   uint level, vec2 heights) {
    float size = treeTerrainNodeSize(c.lod, level);
    vec2 lo = vec2(c.lod.root.x + float(ix) * size, c.lod.root.y + float(iz) * size);
    vec2 hi = lo + vec2(size, size);
    if (!treeTerrainRectInSoil(c.terrain, lo, hi)) {
        return TREE_TERRAIN_CULL;
    }
    vec3 lo3 = vec3(lo.x, heights.x, lo.y);
    vec3 hi3 = vec3(hi.x, heights.y, hi.y);
    if (treeBoxOutsideFrustum(viewProj, lo3, hi3)) {
        return TREE_TERRAIN_CULL;
    }
    if (level == 0u || zoom * treeBoxDistance(eye, lo3, hi3) > treeTerrainRange(c.lod, level - 1u)) {
        return TREE_TERRAIN_DRAW;
    }
    return TREE_TERRAIN_SPLIT;
}

// --- land cover --------------------------------------------------------------

// What decides the land cover at (x, z): x the slope (rise per run, over 0.015 units: about 40 m at scale 1, so
// the finest roughness does not count) and y the altitude in metres above the valley floors, both at true
// proportions; z the forest score, a low-frequency noise in about [0, 1] raised on
// steep ground (people clear the flat land, the forest stays on the slopes); w the field parcels' noise.
TREE_FN vec4 treeLandSite(TreeTerrainParams t, float x, float z) {
    float e = 0.0075f;
    float hx = treeTerrainHeight(t, x + e, z) - treeTerrainHeight(t, x - e, z);
    float hz = treeTerrainHeight(t, x, z + e) - treeTerrainHeight(t, x, z - e);
    float slope = sqrt(hx * hx + hz * hz) / (2.0f * e * t.cover.w);
    float altitude = treeTerrainRelief(t, x, z) / t.cover.w * t.cover.y;
    uint seed = treeHash2(t.seed.x, 0x1a2dc0u);
    vec2 p = vec2(x, z) * (1.6f * t.relief.y);
    vec2 q = vec2(1.6f * p.x - 1.2f * p.y, 1.2f * p.x + 1.6f * p.y);
    float n = 0.62f * treeGradientNoise(p, seed) + 0.26f * treeGradientNoise(q, seed + 1u) +
              0.12f * treeGradientNoise(2.1f * q, seed + 2u);
    float score = 0.5f + 0.5f * n + 0.35f * smoothstep(0.08f, 0.6f, slope);
    float parcels = treeGradientNoise(4.0f * p, seed + 5u);
    return vec4(slope, altitude, score, parcels);
}

// The land cover from what decides it (treeLandSite()). Rock on cliffs (from 48 degrees) and on the peaks well
// above the tree line; forest where the score passes the threshold, up to the tree line; of the open land, scrub
// just outside the forest's edge and around the tree line, fields in parcels on flat, low ground, meadows
// elsewhere (alpine meadows above the tree line).
TREE_FN TreeLandCover treeLandCoverFrom(TreeTerrainParams t, vec4 site) {
    float slope = site.x;
    float altitude = site.y;
    float threshold = t.cover.x;
    float treeLine = t.cover.z;
    float rock = max(smoothstep(1.1f, 1.6f, slope), 0.85f * smoothstep(treeLine + 200.0f, treeLine + 330.0f, altitude));
    float forested = smoothstep(threshold - 0.025f, threshold + 0.025f, site.z) *
                     (1.0f - smoothstep(treeLine - 150.0f, treeLine, altitude));
    float rest = 1.0f - rock;
    float forest = rest * forested;
    float open = rest - forest;
    float edge = smoothstep(threshold - 0.1f, threshold - 0.03f, site.z);
    float krummholz = smoothstep(treeLine - 250.0f, treeLine - 100.0f, altitude) *
                      (1.0f - smoothstep(treeLine + 50.0f, treeLine + 200.0f, altitude));
    float scrub = open * max(0.7f * edge, 0.6f * krummholz);
    open -= scrub;
    float fieldness = (1.0f - smoothstep(0.12f, 0.2f, slope)) * (1.0f - smoothstep(500.0f, 800.0f, altitude)) *
                      smoothstep(-0.2f, 0.1f, site.w);
    float field = open * fieldness;
    TreeLandCover c;
    c.vegetation = vec4(forest, open - field, field, scrub);
    c.bare = vec4(rock, 0.0f, 0.0f, 0.0f);
    return c;
}

TREE_FN TreeLandCover treeLandCover(TreeTerrainParams t, float x, float z) {
    return treeLandCoverFrom(t, treeLandSite(t, x, z));
}

// --- tree placement ------------------------------------------------------------

// The share of conifers among the trees at `altitude` metres above the valley floors.
TREE_FN float treeConiferShare(TreePlacementParams p, float altitude) {
    return mix(p.species.z, 1.0f, smoothstep(p.species.x - p.species.y, p.species.x + p.species.y, altitude));
}

// The mean height of the trees at `altitude` in units: the placement's height, smaller towards the tree line.
TREE_FN float treeMeanTreeHeight(TreeTerrainParams t, TreePlacementParams p, float altitude) {
    float treeLine = t.cover.z;
    return p.size.x / p.grid.w * (1.0f - 0.6f * smoothstep(treeLine - 300.0f, treeLine, altitude));
}

// How much the forest canopy stands in for the trees at a place where a tree spans `pixels` pixels: 1 below
// canopy.x, 0 above canopy.y, smoothly between (the trees' proxies fade the other way, treeProxyFade()).
TREE_FN float treeCanopyWeight(vec4 canopy, float pixels) {
    if (canopy.x <= 0.0f) {
        return 0.0f;
    }
    return 1.0f - smoothstep(canopy.x, canopy.y, pixels);
}

// How much a tree's proxy shows when it spans `pixels` pixels: the complement of treeCanopyWeight().
TREE_FN float treeProxyFade(vec4 canopy, float pixels) { return 1.0f - treeCanopyWeight(canopy, pixels); }

// The trees' sizes on screen in TREE_SIZE_BINS bins of equal width in log2(pixels), from 0.5 to 4096 pixels: the
// library copies (TreeGaussianForest) go to the largest trees, down to a cut-off between bins, and fade in over
// TREE_COPY_FADE_WIDTH (in log2 pixels, four bins) above it.
#define TREE_SIZE_BINS 256u
#define TREE_SIZE_LOG_MIN -1.0f
#define TREE_SIZE_BIN_WIDTH 0.05078125f
#define TREE_COPY_FADE_WIDTH 0.203125f

TREE_FN uint treeSizeBin(float pixels) {
    float x = (log2(max(pixels, 1e-6f)) - TREE_SIZE_LOG_MIN) / TREE_SIZE_BIN_WIDTH;
    return uint(clamp(floor(x), 0.0f, float(TREE_SIZE_BINS - 1u)));
}

// The lower edge of `bin` in log2(pixels).
TREE_FN float treeSizeBinEdge(uint bin) { return TREE_SIZE_LOG_MIN + float(bin) * TREE_SIZE_BIN_WIDTH; }

// How many times its size on screen a tree counts when the library copies are given to the largest trees, for the
// forest weight at its foot: the canopy stands in for the trees of a forest below the cut-off, but not for those in
// the open.
TREE_FN float treeCopyPriority(float forest) { return mix(8.0f, 1.0f, clamp(forest, 0.0f, 1.0f)); }

// How much a library copy shows at `pixels` for the cut-off `cut` (log2 pixels): none at the cut-off, all
// TREE_COPY_FADE_WIDTH above it.
TREE_FN float treeCopyFade(float pixels, float cut) {
    return clamp((log2(max(pixels, 1e-6f)) - cut) / TREE_COPY_FADE_WIDTH, 0.0f, 1.0f);
}

#define TREE_SITE_BROADLEAF 0.0f
#define TREE_SITE_CONIFER 1.0f
#define TREE_SITE_SHRUB 2.0f

// The tree of cell (cx, cz), whose square spans [cx, cx + 1) x [cz, cz + 1) cells from the origin. The candidate
// lies near the cell's middle, jittered by up to half of p.grid.y cells each way, so trees stand at least
// (1 - p.grid.y) cells apart. It becomes a tree with the probability of the forest weight there, else a shrub with
// the scrub weight times p.grid.z; species and height come from the altitude (conifers higher up, smaller trees
// towards the tree line). Each cell's decisions hash its own indices, so a cell's tree is the same whenever and
// in whatever order it is evaluated.
TREE_FN TreeSite treeSiteAt(TreeTerrainParams t, TreePlacementParams p, int cx, int cz) {
    TreeSite site;
    site.posHeight = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    site.kind = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    uint key = treeHash2(treeHash2(uint(cx), uint(cz)), p.seed.x);
    float cell = p.grid.x;
    float x = (float(cx) + 0.5f + p.grid.y * (treeRand(key, 1u) - 0.5f)) * cell;
    float z = (float(cz) + 0.5f + p.grid.y * (treeRand(key, 2u) - 0.5f)) * cell;
    // Off the pot's wall.
    if (treeTerrainRadius(t, x, z) >= 0.98f) {
        return site;
    }
    vec4 land = treeLandSite(t, x, z);
    TreeLandCover c = treeLandCoverFrom(t, land);
    float u = treeRand(key, 3u);
    float forest = c.vegetation.x;
    bool shrub = u >= forest;
    if (shrub && u >= forest + p.grid.z * c.vegetation.w) {
        return site;
    }
    float altitude = land.y;
    float meters;
    float species;
    if (shrub) {
        species = TREE_SITE_SHRUB;
        meters = p.size.z * (0.6f + 0.8f * treeRand(key, 5u));
    } else {
        species = treeRand(key, 4u) < treeConiferShare(p, altitude) ? TREE_SITE_CONIFER : TREE_SITE_BROADLEAF;
        meters = treeMeanTreeHeight(t, p, altitude) * p.grid.w * (1.0f + p.size.y * (2.0f * treeRand(key, 5u) - 1.0f));
    }
    site.posHeight = vec4(x, treeTerrainHeight(t, x, z), z, meters / p.grid.w);
    site.kind = vec4(species, 2.0f * TREE_PI * treeRand(key, 6u), forest, 0.0f);
    return site;
}

// Scene materials, by the w of a vertex colour.
#define TREE_MATERIAL_PLAIN 0.0f
#define TREE_MATERIAL_MOSS 1.0f
#define TREE_MATERIAL_WOOD 2.0f
#define TREE_MATERIAL_GLAZE 3.0f

// Where moss grows on the soil: patches a few centimetres wide at the
// lantern's scale (it is 0.28 tall), 0 bare soil, 1 moss.
TREE_FN float treeMossCoverage(vec3 pos) {
    float large = treeFbm(pos * 4.0f + vec3(3.0f, 0.0f, 1.0f)) - 0.5f;
    return smoothstep(0.4f, 0.56f, treeFbm(pos * 22.0f) + 0.35f * large);
}

// Fine moss cushions in [0, 1], about 7 mm wide at the lantern's scale.
TREE_FN float treeMossCushions(vec3 pos) { return treeValueNoise(pos * 140.0f); }

// Height of the moss relief, in units, for the normal of the classical renderer.
TREE_FN float treeMossHeight(vec3 pos) {
    return 0.004f * treeMossCoverage(pos) * (0.4f + 0.6f * treeMossCushions(pos));
}

// The albedo of mossy soil: bare soil with grit and a few pebbles, moss in
// patches with lighter cushion tops. `detail` in [0, 1] fades the finest
// structure (cushions, grit) to its mean where it cannot be resolved.
TREE_FN vec3 treeMossAlbedo(vec3 base, vec3 pos, float detail) {
    float coverage = treeMossCoverage(pos);
    float tint = treeFbm(pos * 60.0f + vec3(7.0f, 0.0f, 3.0f));
    float cushions = mix(0.5f, treeMossCushions(pos), detail);
    vec3 moss = base * (0.7f + 0.6f * tint) * (0.6f + 0.8f * cushions);
    float grit = mix(0.5f, treeValueNoise(pos * 260.0f), detail);
    vec3 soil = vec3(0.055f, 0.04f, 0.028f) * (0.75f + 0.5f * grit);
    float pebble = smoothstep(0.8f, 0.86f, treeValueNoise(pos * 90.0f + vec3(11.0f, 0.0f, 5.0f)));
    soil = mix(soil, vec3(0.2f, 0.19f, 0.17f) * (0.8f + 0.4f * tint), pebble);
    return mix(soil, moss, coverage);
}

// The albedo of a scene surface at world position pos: plain colour, moss
// over soil, wood grain along x, or a glaze with fine mottling. `detail` as
// for treeMossAlbedo().
TREE_FN vec3 treeSceneAlbedoDetail(vec4 color, vec3 pos, float detail) {
    vec3 base = vec3(color.x, color.y, color.z);
    if (color.w == TREE_MATERIAL_MOSS) {
        return treeMossAlbedo(base, pos, detail);
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

TREE_FN vec3 treeSceneAlbedo(vec4 color, vec3 pos) { return treeSceneAlbedoDetail(color, pos, 1.0f); }

// Factor on the radius of a stem of `level` at z in [0, 1] along it and angle
// phi around it: root buttresses at the trunk's base, Weber and Penn's lobes,
// that fade out over the lowest sixth of the trunk.
TREE_FN float treeTrunkLobes(TreeTypeParams p, int level, float z, float phi) {
    if (level != 0 || p.trunk.y == 0.0f) {
        return 1.0f;
    }
    float w = max(0.0f, 1.0f - 6.0f * z);
    return 1.0f + p.trunk.y * w * w * (0.5f + 0.5f * sin(p.trunk.x * phi));
}

// Bark relief in [0, 1] at uv (around, along the stem, in furrows): ridges
// that run along the stem, separated by furrows (0).
TREE_FN float treeBarkHeight(vec2 uv) {
    float n = treeFbm(vec3(uv.x, 0.15f * uv.y, 0.0f));
    float ridge = 1.0f - abs(2.0f * n - 1.0f);
    return ridge * ridge;
}

// --- seasons (Kuth et al., Sec. 3.4) --------------------------------------

// The season S in [0, 4): 0 winter, 1 spring, 2 summer, 3 fall, continuous in
// between; 4 is winter again.
TREE_FN float treeSeasonWrap(float s) { return s - 4.0f * floor(s / 4.0f); }

// State of a deciduous leaf in season S: x its size (0: not there), y its
// freshness (young spring leaves are lighter), z its fall colouring. r in
// [0, 1) is the leaf's own random number, so leaves bud, colour and drop at
// their own times; `offset` shifts the type's seasons. Evergreen leaves stay
// fully grown and green.
TREE_FN vec3 treeLeafSeason(float season, float r, float offset, bool deciduous) {
    if (!deciduous) {
        return vec3(1.0f, 0.0f, 0.0f);
    }
    float s = treeSeasonWrap(season - offset);
    float size = smoothstep(0.6f + 0.3f * r, 1.3f + 0.3f * r, s);
    size *= 1.0f - smoothstep(3.2f + 0.6f * r, 3.3f + 0.6f * r, s);
    float fresh = 1.0f - smoothstep(1.2f, 2.0f, s);
    float fall = smoothstep(2.5f + 0.4f * r, 3.1f + 0.4f * r, s);
    return vec3(size, fresh, fall);
}

// The crown of a proxy tree (TreeProxies) in season S: rgb its albedo, w how dense it is (1 in leaf or for
// conifers, sparse twigs for bare broadleaves). Conifers are dark green; broadleaves fresh green in spring, green
// in summer, yellow to red in fall, bare in winter, each tree a little apart from its neighbours.
TREE_FN vec4 treeProxyCrown(TreeSite site, float season) {
    float r = treeRand(floatBitsToUint(site.kind.y), 0x31u);
    bool conifer = site.kind.x == TREE_SITE_CONIFER;
    if (conifer) {
        return vec4(vec3(0.025f, 0.065f, 0.03f) * (0.8f + 0.4f * r), 1.0f);
    }
    vec3 state = treeLeafSeason(season, r, 0.0f, true);
    vec3 green = vec3(0.05f, 0.12f, 0.03f) * (0.8f + 0.4f * r);
    green = mix(green, green * 1.4f + vec3(0.02f, 0.03f, 0.0f), 0.5f * state.y);
    vec3 fall = mix(vec3(0.3f, 0.09f, 0.02f), vec3(0.36f, 0.25f, 0.03f), r);
    vec3 leaves = mix(green, fall, state.z);
    vec3 twigs = vec3(0.1f, 0.085f, 0.07f);
    return vec4(mix(twigs, leaves, state.x), mix(0.25f, 1.0f, state.x));
}

// The shape of a proxy tree of `species` and height h: x the bottom of its crown, y the height of its widest ring,
// z that ring's radius, w the trunk's radius, all as fractions of h. Its crown is two cones meeting at the ring
// and reaching up to h.
TREE_FN vec4 treeProxyShape(float species) {
    if (species == TREE_SITE_CONIFER) {
        return vec4(0.1f, 0.22f, 0.24f, 0.025f);
    }
    if (species == TREE_SITE_SHRUB) {
        return vec4(0.05f, 0.45f, 0.5f, 0.0f);
    }
    return vec4(0.3f, 0.62f, 0.32f, 0.03f);
}

// Snow cover in season S: in deep winter.
TREE_FN float treeSnowCover(float season) {
    float s = treeSeasonWrap(season);
    return max(1.0f - smoothstep(0.1f, 0.5f, s), smoothstep(3.7f, 3.95f, s));
}

TREE_FN uint treeLeafSeasonPack(vec3 state) {
    return uint(clamp(state.y, 0.0f, 1.0f) * 255.0f + 0.5f) | (uint(clamp(state.z, 0.0f, 1.0f) * 255.0f + 0.5f) << 8u);
}

// Freshness (x) and fall colouring (y) of a leaf record.
TREE_FN vec2 treeLeafSeasonUnpack(uint packed) {
    return vec2(float(packed & 255u), float((packed >> 8u) & 255u)) / 255.0f;
}

// The albedo of a leaf in its season: treeLeafAlbedo() of the type's colour,
// lighter and yellower while fresh, turning to the fall colour, which each
// leaf shifts towards yellow or red.
TREE_FN vec3 treeLeafSeasonAlbedo(TreeTypeParams p, uint leafSeed, uint treeSeed, uint packedSeason) {
    vec2 state = treeLeafSeasonUnpack(packedSeason);
    vec3 green = treeLeafAlbedo(vec3(p.leafColor.x, p.leafColor.y, p.leafColor.z), leafSeed, treeSeed);
    green = mix(green, green * vec3(1.9f, 1.6f, 0.8f), state.x);
    float hue = treeRand(leafSeed, 0x7bu);
    vec3 fallBase = vec3(p.leafColorFall.x, p.leafColorFall.y, p.leafColorFall.z);
    vec3 fall = fallBase * mix(vec3(1.25f, 0.45f, 0.6f), vec3(0.95f, 1.35f, 0.8f), hue) *
                (0.75f + 0.5f * treeRand(leafSeed, 0x7cu));
    return mix(green, fall, state.y);
}

// Snow lying on a surface with this normal, by the snow cover: on what faces
// upwards, patchy where it is steep.
TREE_FN float treeSnowOn(vec3 normal, vec3 pos, float cover) {
    if (cover <= 0.0f) {
        return 0.0f;
    }
    float upwards = smoothstep(0.15f, 0.75f, normal.y);
    float patches = smoothstep(0.3f, 0.6f, treeValueNoise(pos * 60.0f) + 0.6f * cover - 0.3f);
    return cover * upwards * mix(patches, 1.0f, upwards * cover);
}

// --- terrain materials --------------------------------------------------------

// Fades a pattern of features `size` wide to its mean once a pixel covers more than half a feature, as the moss
// detail does: 1 while resolved, 0 when not. `footprint` and `size` in the same unit.
TREE_FN float treeTerrainResolved(float footprint, float size) {
    return clamp(1.5f - 2.0f * footprint / size, 0.0f, 1.0f);
}

// Field parcels: x a random number of the parcel, y the direction of its rows in radians, z how far a point lies
// inside the parcel (0 on a border, 0.5 in the middle), from the position in metres. Parcels are strips of about
// 220 x 90 m, turned by an angle that changes over kilometres, their borders bent by noise so that they are not
// a grid.
TREE_FN vec3 treeFieldParcel(vec2 meters, uint seed) {
    float angle = TREE_PI * treeGradientNoise(meters * 0.0004f, seed + 10u);
    vec2 turned = vec2(cos(angle) * meters.x - sin(angle) * meters.y, sin(angle) * meters.x + cos(angle) * meters.y);
    vec2 bend = vec2(treeGradientNoise(meters * 0.005f, seed + 11u), treeGradientNoise(meters * 0.005f, seed + 12u));
    vec2 cell = turned / vec2(220.0f, 90.0f) + 0.45f * bend;
    vec2 index = floor(cell);
    vec2 inside = cell - index;
    uint key = treeHash2(uint(int(index.x)), uint(int(index.y)));
    float border = min(min(inside.x, 1.0f - inside.x), min(inside.y, 1.0f - inside.y));
    return vec3(treeRand(seed, key), TREE_PI * treeRand(seed + 1u, key), border);
}

// The albedo of a field in season S: one of four crops per parcel (wheat, maize, plough land, grassland for hay),
// in rows that fade out at a distance, with darker borders.
TREE_FN vec3 treeFieldAlbedo(vec2 meters, float footprint, float season, uint seed) {
    vec3 parcel = treeFieldParcel(meters, seed);
    float s = treeSeasonWrap(season);
    // Green from spring, ripe in summer, stubble or bare in fall and winter.
    float growth = smoothstep(0.8f, 1.6f, s) * (1.0f - smoothstep(3.1f, 3.6f, s));
    float ripe = smoothstep(1.5f, 2.1f, s);
    vec3 bare = vec3(0.13f, 0.09f, 0.06f);
    vec3 crop;
    if (parcel.x < 0.3f) {
        crop = mix(bare, mix(vec3(0.11f, 0.19f, 0.045f), vec3(0.34f, 0.28f, 0.11f), ripe), growth);
    } else if (parcel.x < 0.5f) {
        crop = mix(bare, mix(vec3(0.09f, 0.17f, 0.04f), vec3(0.2f, 0.2f, 0.08f), 0.5f * ripe), growth);
    } else if (parcel.x < 0.7f) {
        crop = vec3(0.15f, 0.1f, 0.065f);
    } else {
        crop = mix(vec3(0.1f, 0.13f, 0.05f), vec3(0.13f, 0.21f, 0.06f), growth);
    }
    float c = cos(parcel.y);
    float d = sin(parcel.y);
    float rows = 0.5f + 0.5f * sin(2.0f * TREE_PI * (c * meters.x + d * meters.y) / 1.5f);
    crop *= 1.0f + 0.25f * (rows - 0.5f) * treeTerrainResolved(footprint, 1.5f);
    float hedge = 1.0f - smoothstep(0.0f, 0.03f, parcel.z);
    return mix(crop, vec3(0.05f, 0.08f, 0.03f), hedge * treeTerrainResolved(footprint, 8.0f));
}

// The albedo of the ground of land cover `c` at world position pos, seen with `footprint` world units per
// pixel, in season S. The forest floor is the mossy soil of the pot, its patterns scaled by t.soil.w; the other
// classes are patterned in metres. Every pattern fades to its mean where a pixel covers it.
TREE_FN vec3 treeTerrainAlbedo(TreeTerrainParams t, TreeLandCover c, vec3 pos, float footprint, float season) {
    vec3 pd = pos * t.soil.w;
    float fd = footprint * t.soil.w;
    float detail = clamp(1.5f - 140.0f * fd, 0.0f, 1.0f);
    vec3 moss = treeMossAlbedo(vec3(0.075f, 0.11f, 0.025f), pd, detail);
    vec3 albedo = c.vegetation.x * mix(vec3(0.062f, 0.072f, 0.027f), moss, clamp(1.5f - 40.0f * fd, 0.0f, 1.0f));
    float rest = 1.0f - c.vegetation.x;
    if (rest <= 0.0f) {
        return albedo;
    }
    vec3 pm = pos * t.cover.y;
    float fm = footprint * t.cover.y;
    vec2 m = vec2(pm.x, pm.z);
    float broad = treeFbm(pm * 0.025f);
    float fine = mix(0.5f, treeFbm(pm * 0.4f), treeTerrainResolved(fm, 2.5f));
    float s = treeSeasonWrap(season);
    float dry = smoothstep(1.8f, 2.6f, s) * (1.0f - smoothstep(3.5f, 3.9f, s));
    vec3 meadow = mix(vec3(0.1f, 0.17f, 0.045f), vec3(0.2f, 0.18f, 0.08f), dry * smoothstep(0.45f, 0.7f, broad));
    meadow *= 0.8f + 0.4f * broad + 0.3f * (fine - 0.5f);
    vec3 scrub = vec3(0.06f, 0.095f, 0.035f) * (0.7f + 0.6f * fine) * (0.85f + 0.3f * broad);
    float strata = 0.5f + 0.5f * sin(pm.y / 6.0f + 4.0f * broad);
    vec3 cliff = vec3(0.15f, 0.145f, 0.14f) * (0.75f + 0.5f * broad) *
                 (1.0f + 0.2f * (strata - 0.5f) * treeTerrainResolved(fm, 6.0f) + 0.3f * (fine - 0.5f));
    vec3 scree = vec3(0.2f, 0.195f, 0.18f) * (0.85f + 0.3f * fine);
    vec3 rock = mix(scree, cliff, smoothstep(0.35f, 0.85f, c.bare.x));
    albedo += c.vegetation.y * meadow + c.vegetation.z * treeFieldAlbedo(m, fm, season, t.seed.x) +
              c.vegetation.w * scrub + c.bare.x * rock;
    return albedo;
}

// The albedo of the forest canopy at world position pos (TreeTerrainConfig::canopy) in season S, seen with
// `footprint` world units per pixel: the crowns of broadleaves and conifers mixed by their share at the altitude,
// each crown (a cell of the placement) a little apart in colour, darker in the gaps between crowns; the crowns
// and gaps fade to their mean where a pixel covers them.
TREE_FN vec3 treeCanopyAlbedo(TreeTerrainParams t, TreePlacementParams p, vec3 pos, float footprint, float season) {
    float altitude = treeTerrainAltitude(t, pos);
    TreeSite crown;
    crown.posHeight = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    crown.kind = vec4(TREE_SITE_BROADLEAF, 0.0f, 1.0f, 0.0f);
    vec3 broadleaf = vec3(treeProxyCrown(crown, season));
    crown.kind.x = TREE_SITE_CONIFER;
    vec3 conifer = vec3(treeProxyCrown(crown, season));
    vec3 mean = mix(broadleaf, conifer, treeConiferShare(p, altitude));
    // Crowns about a cell wide: their own shade of the mix, lit tops and dark gaps.
    float cell = p.grid.x;
    vec3 c = pos / cell;
    float resolved = treeTerrainResolved(footprint, cell);
    float tone = mix(0.5f, treeValueNoise(vec3(c.x, 0.0f, c.z) * 1.3f), resolved);
    float gaps = mix(0.5f, treeFbm(vec3(c.x, 0.0f, c.z) * 2.1f + vec3(3.1f, 0.0f, 7.7f)), resolved);
    float conifers = treeConiferShare(p, altitude);
    vec3 albedo = mix(broadleaf, conifer, clamp(conifers + 0.6f * (tone - 0.5f), 0.0f, 1.0f));
    albedo *= 0.75f + 0.5f * smoothstep(0.3f, 0.7f, gaps);
    // Far away the canopy is its mean colour (as the mean of the crowns and gaps above), as bright as the trees'
    // impostors it takes over from.
    return mix(mean, albedo, resolved);
}

// The snow line in metres above the valley floors in season S: 1,300 m in summer, 1,050 m in spring and fall,
// 800 m in winter.
TREE_FN float treeSnowLine(float season) {
    return 1050.0f + 250.0f * cos(0.5f * TREE_PI * (treeSeasonWrap(season) - 2.0f));
}

// Snow on the ground at world position pos with `normal`: the winter's cover everywhere (treeSnowOn()), and
// above the snow line on all but steep ground.
TREE_FN float treeTerrainSnow(TreeTerrainParams t, vec3 pos, vec3 normal, float season, float cover) {
    float winter = treeSnowOn(normal, pos * t.soil.w, cover);
    float line = treeSnowLine(season);
    float high = smoothstep(line, line + 150.0f, treeTerrainAltitude(t, pos)) * smoothstep(0.35f, 0.7f, normal.y);
    return max(winter, high);
}

// --- grass ---------------------------------------------------------------------

// How tall a field's crop stands in season S, relative to grass, at a position in metres: grain and other crops
// grow from spring to summer and stand until the fall, then stubble; nothing on ploughed parcels; pasture like
// grass (the parcels of treeFieldAlbedo()).
TREE_FN float treeFieldCropHeight(vec2 meters, float season, uint seed) {
    vec3 parcel = treeFieldParcel(meters, seed);
    float s = treeSeasonWrap(season);
    float growth = smoothstep(0.8f, 1.6f, s) * (1.0f - smoothstep(3.1f, 3.6f, s));
    if (parcel.x < 0.3f) {
        return 0.15f + 2.0f * growth;
    }
    if (parcel.x < 0.5f) {
        return 0.15f + 1.2f * growth;
    }
    if (parcel.x < 0.7f) {
        return 0.0f;
    }
    return 1.0f;
}

// The clump of grass of cell (cx, cz) seen from eye in season S: where meadow, field (and some scrub) grows, with
// the probability of their weight, jittered in its cell; with fewer blades from a third of the radius on and none
// beyond it; in fields as tall as their crop (treeFieldCropHeight()). Hashing the cell's indices keeps a clump the
// same while the camera moves. No grass: x of kind is 0.
TREE_FN TreeGrassClump treeGrassCell(TreeTerrainParams t, TreeGrassParams g, int cx, int cz, vec3 eye, float season) {
    TreeGrassClump clump;
    clump.posHeight = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    clump.kind = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    uint key = treeHash2(treeHash2(uint(cx), uint(cz)), g.counts.z);
    float x = (float(cx) + treeRand(key, 1u)) * g.grid.x;
    float z = (float(cz) + treeRand(key, 2u)) * g.grid.x;
    float d = length(vec2(x - eye.x, z - eye.z));
    if (d > g.grid.y || treeTerrainRadius(t, x, z) >= 0.98f) {
        return clump;
    }
    TreeLandCover c = treeLandCover(t, x, z);
    float open = c.vegetation.y + c.vegetation.z + 0.6f * c.vegetation.w;
    float u = treeRand(key, 3u);
    if (u >= open) {
        return clump;
    }
    // Field or other open land, by where u falls among their weights.
    float crop = u >= c.vegetation.y && u < c.vegetation.y + c.vegetation.z
                     ? treeFieldCropHeight(vec2(x, z) * t.cover.y, season, t.seed.x)
                     : 1.0f;
    float meters = g.blade.x * (0.6f + 0.8f * treeRand(key, 4u)) * crop;
    if (meters < 0.02f) {
        return clump;
    }
    float blades = max(1.0f, floor(g.grid.z * (1.0f - smoothstep(0.33f * g.grid.y, g.grid.y, d)) + 0.5f));
    clump.posHeight = vec4(x, treeTerrainHeight(t, x, z), z, meters / g.grid.w);
    clump.kind = vec4(blades, 2.0f * TREE_PI * treeRand(key, 5u), 1.0f, treeRand(key, 6u));
    return clump;
}

// --- forest floor ----------------------------------------------------------------

// The copy of a moss patch in cell (cx, cz) seen from eye: where the forest grows, with the probability of its
// weight, jittered in its cell, turned at random, on the plane of the ground; fading out (ids.y) from grid.w of the
// radius on, none beyond it. Hashing the cell's indices keeps a copy the same while the camera moves. No copy:
// scale (plane.z) 0.
TREE_FN TreeLodInstance treeFloorCell(TreeFloorSetup f, int cx, int cz, vec3 eye) {
    TreeLodInstance inst;
    inst.posYaw = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    inst.plane = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    inst.ids = uvec4(0u, 0u, 0u, 0u);
    uint key = treeHash2(treeHash2(uint(cx), uint(cz)), f.counts.z);
    float x = (float(cx) + 0.2f + 0.6f * treeRand(key, 1u)) * f.grid.x;
    float z = (float(cz) + 0.2f + 0.6f * treeRand(key, 2u)) * f.grid.x;
    float d = length(vec2(x - eye.x, z - eye.z));
    if (d > f.grid.y || treeTerrainRadius(f.terrain, x, z) >= 0.98f) {
        return inst;
    }
    if (treeRand(key, 3u) >= treeLandCover(f.terrain, x, z).vegetation.x) {
        return inst;
    }
    float y = treeTerrainHeight(f.terrain, x, z);
    float e = 0.5f * f.grid.x;
    inst.posYaw = vec4(x, y, z, 2.0f * TREE_PI * treeRand(key, 4u));
    inst.plane = vec4((treeTerrainHeight(f.terrain, x + e, z) - y) / e,
                      (treeTerrainHeight(f.terrain, x, z + e) - y) / e, f.grid.z, 0.0f);
    float fade = smoothstep(f.grid.w * f.grid.y, f.grid.y, d);
    inst.ids = uvec4(treeHash2(key, 5u) % max(f.counts.y, 1u), floatBitsToUint(fade), 0u, 0u);
    return inst;
}

// The normal of a flat Gaussian: its thinnest axis, turned up.
TREE_FN vec3 treeGaussianNormal(vec4 rot, vec3 scale) {
    vec3 axis = scale.x <= scale.y && scale.x <= scale.z
                    ? vec3(1.0f, 0.0f, 0.0f)
                    : (scale.y <= scale.z ? vec3(0.0f, 1.0f, 0.0f) : vec3(0.0f, 0.0f, 1.0f));
    vec3 n = qrot(rot, axis);
    return n.y < 0.0f ? -n : n;
}

// The displayed colour of a captured Gaussian under the current light: its
// captured colour tinted and shaded, covered with snow where it faces up in
// winter.
TREE_FN vec3 treeCapturedColor(vec3 displayed, vec3 normal, vec3 pos, float shade, TreeRenderSettings s) {
    vec3 tint = vec3(s.capturedTint.x, s.capturedTint.y, s.capturedTint.z) * shade;
    float snow = treeSnowOn(normal, pos, s.season.y);
    return mix(displayed, vec3(0.85f, 0.87f, 0.9f), snow) * tint;
}

// --- Gaussian level of detail (after klartraum-spacegame) -----------------

// Diameter in pixels of a sphere of `radius` at `center`, seen from `eye` with
// `focalPixels` pixels per unit at distance 1.
TREE_FN float treePixelDiameter(vec3 eye, vec3 center, float radius, float focalPixels) {
    vec3 d = center - eye;
    return 2.0f * focalPixels * radius / sqrt(max(dot(d, d) - radius * radius, 1e-6f));
}

// The level of detail (x, 0 = full) for a projected diameter in pixels, and
// how far (y in [0, 1]) it has faded to the next coarser level. Level k is
// used above boundaries[k]; within +-band of a boundary the two levels
// crossfade, so changes in distance never switch detail at once.
TREE_FN vec2 treeLodBlend(float pixels, vec3 boundaries, float band) {
    for (int k = 0; k < 3; ++k) {
        float upper = boundaries[k] * (1.0f + band);
        float lower = boundaries[k] * (1.0f - band);
        if (pixels >= upper) {
            return vec2(float(k), 0.0f);
        }
        if (pixels > lower) {
            return vec2(float(k), (upper - pixels) / (upper - lower));
        }
    }
    return vec2(3.0f, 0.0f);
}

// The Gaussians a copy of `source` reserves at `pixels`: those of its level and, while crossfading, of the next
// coarser one.
TREE_FN uint treeLodReserved(TreeLodPatch source, float pixels, vec3 boundaries, float band) {
    vec2 lod = treeLodBlend(pixels, boundaries, band);
    uint level = uint(lod.x);
    uint count = source.count[level];
    if (lod.y > 0.0f && level < 3u) {
        count += source.count[level + 1u];
    }
    return count;
}

// TreeMossLod's fitting to its pool tries the level boundaries times treeLodFitFactor(k) for k below
// TREE_LOD_FIT_STEPS: from 1 up to 2^3.75 in steps of 2^0.25.
#define TREE_LOD_FIT_STEPS 16u

TREE_FN float treeLodFitFactor(uint step) { return exp2(0.25f * float(step)); }

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

// Half width of a lobe at height y in [0, 1], along the polygon of its curve ends.
TREE_FN float treeLobeHalfWidth(TreeLeafOutline o, float y) {
    for (int c = 0; c < 4; ++c) {
        vec2 a = o.v[2 * c];
        vec2 b = o.v[2 * c + 2];
        if (y <= b.y || c == 3) {
            float t = clamp((y - a.y) / max(b.y - a.y, 1e-6f), 0.0f, 1.0f);
            return mix(a.x, b.x, t);
        }
    }
    return 0.0f;
}

// Gaussian `local` of the `count` Gaussians tree_gaussians.comp lays along the lobes of a leaf of type `p`, per
// unit of leaf length: xyz its scale (x across the blade, y the leaf normal, z along the lobe; lobes at least 0.02
// wide), and xy its place in the leaf's plane and z the lobe's angle about the normal.
struct TreeLeafGaussian {
    vec4 scale;
    vec4 placement;
};

TREE_FN TreeLeafGaussian treeLeafGaussianShape(TreeTypeParams p, TreeLeafOutline o, uint local, uint count) {
    uint lobes = uint(clamp(p.leafLobes.x, 1.0f, float(count)));
    uint lobe = local % lobes;
    uint row = local / lobes;
    uint rows = (count - lobe + lobes - 1u) / lobes;
    float y = (float(row) + 0.5f) / float(rows);
    float halfWidth = max(treeLobeHalfWidth(o, y), 0.02f);
    float offset = float(lobe) - 0.5f * float(lobes - 1u);
    float angle = radians(offset * p.leafLobes.y);
    float lobeScale = pow(p.leafLobes.z, abs(offset));
    TreeLeafGaussian g;
    g.scale = vec4(lobeScale * 0.55f * halfWidth, lobeScale * 0.01f, lobeScale * 0.6f / float(rows), 0.0f);
    g.placement = vec4(lobeScale * sin(angle) * y, lobeScale * cos(angle) * y, angle, 0.0f);
    return g;
}

// What a leaf of length 1 and type `p` covers as the near trees draw it with `count` Gaussians of opacity
// `alpha`: the sum of their integrated opacities, 2 pi sx sz alpha.
TREE_FN float treeLeafGaussianCoverage(TreeTypeParams p, uint count, float alpha) {
    TreeLeafOutline o = treeLeafOutline(p.leafShape, 0.5f * p.leafCount.z);
    float sum = 0.0f;
    for (uint k = 0u; k < count; ++k) {
        TreeLeafGaussian g = treeLeafGaussianShape(p, o, k, count);
        sum += 2.0f * TREE_PI * g.scale.x * g.scale.z * alpha;
    }
    return sum;
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
