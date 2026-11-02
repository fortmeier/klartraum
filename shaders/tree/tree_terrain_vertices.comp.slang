// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The vertices of the selected terrain nodes (TreeTerrain, CDLOD): one
// workgroup per node computes the heights of its (quads + 1)^2 grid and a
// border of two vertices into shared memory, then each vertex's height and
// normal, morphed towards the next coarser level by its distance from the
// camera (treeTerrainMorph()). Morphing moves only heights and normals: odd
// vertices blend towards the mean of their even neighbours along the edge
// or, inside a quad, along the diagonal the triangles share, so a fully
// morphed node is the coarser level's surface and meets coarser neighbours
// without cracks. Vertex positions come from integer coordinates on the
// finest grid, so nodes compute identical values for the vertices they share.
// The land cover (treeLandCover()) of each vertex is morphed the same way.
// Where the forest canopy stands in for trees too small to draw
// (treeCanopyWeight()), the surface rises to the canopy's top: the vertices'
// heights include it, so normals, morphing and silhouettes follow it. With
// the sun of the settings, each vertex also gets how much of the sun the
// terrain lets through (treeTerrainSunVisibility()) at its morphed position.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 2
#include "tree_render_common.slang"

#define MAX_SIDE 37 // quads + 5 for at most 32 quads per side

[[vk::binding(0, 0)]] StructuredBuffer<vec4> nodes;
[[vk::binding(1, 0)]] StructuredBuffer<TreeTerrainConfig> configBuffer;
// Per node (quads + 1)^2 vertices, rows along x: xyz the normal, w the height.
[[vk::binding(3, 0)]] RWStructuredBuffer<vec4> vertices;
// The same vertices' land cover: forest, meadow, field, scrub (rock is the rest).
[[vk::binding(4, 0)]] RWStructuredBuffer<vec4> covers;
[[vk::binding(5, 0)]] StructuredBuffer<TreeRenderSettings> settings;
// The same vertices' x canopy weight (the share of the surface that is forest canopy), y the sun's visibility
// over the terrain, zw unused.
[[vk::binding(6, 0)]] RWStructuredBuffer<vec4> extras;

groupshared float heights[MAX_SIDE * MAX_SIDE];
groupshared float canopy[MAX_SIDE * MAX_SIDE];
groupshared vec4 cover[(MAX_SIDE - 4) * (MAX_SIDE - 4)];

static int side;

float H(int i, int j) {
    return heights[(j + 2) * side + i + 2];
}

float C(int i, int j) {
    return canopy[(j + 2) * side + i + 2];
}

// The normal of the coarser level at an even vertex: central differences over two vertices.
vec3 coarseNormal(int i, int j, float spacing) {
    return normalize(vec3(H(i - 2, j) - H(i + 2, j), 4.0f * spacing, H(i, j - 2) - H(i, j + 2)));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    vec4 node = nodes[gl_WorkGroupID.x];
    uint level = uint(node.w);
    int quads = int(configBuffer[0].lod.ranges.z);
    side = quads + 5;
    // The finest grid: quads per side of a finest node, times the finest nodes along the root.
    float unit = configBuffer[0].lod.root.z / (exp2(float(configBuffer[0].lod.levels.x - 1u)) * float(quads));
    int nodeStep = 1 << level;
    int gx = int(round((node.x - configBuffer[0].lod.root.x) / node.z)) * quads * nodeStep;
    int gz = int(round((node.y - configBuffer[0].lod.root.y) / node.z)) * quads * nodeStep;
    int row = quads + 1;
    vec3 eye = camera.cameraWorldPos.xyz;
    // The settings are bound only with a canopy or the terrain's shadows.
    bool withCanopy = configBuffer[0].canopy.x > 0.0f;
    // The band of the settings when they give one (it follows the number of visible trees), else the terrain's.
    vec4 band = configBuffer[0].canopy;
    if (withCanopy && settings[0].canopyBand.x > 0.0f) {
        band.xy = settings[0].canopyBand.xy;
    }
    // Where the library copies' cut-off lies above the band, the canopy covers the trees below it.
    if (withCanopy && settings[0].canopyBand.z > band.x) {
        band.xy = settings[0].canopyBand.zw;
    }
    bool withShadows = configBuffer[0].canopy.w > 0.0f;
    float focal = withCanopy ? treeFocalPixels(settings[0]) : 0.0f;
    vec3 sun = withShadows ? settings[0].sun.xyz : vec3(0.0f, 1.0f, 0.0f);
    for (int k = int(gl_LocalInvocationID.x); k < side * side; k += 256) {
        int i = k % side - 2;
        int j = k / side - 2;
        float x = configBuffer[0].lod.root.x + float(gx + i * nodeStep) * unit;
        float z = configBuffer[0].lod.root.y + float(gz + j * nodeStep) * unit;
        float h = treeTerrainHeight(configBuffer[0].terrain, x, z);
        bool inner = i >= 0 && j >= 0 && i < row && j < row;
        float c = 0.0f;
        if (inner || withCanopy) {
            vec4 vegetation = treeLandCover(configBuffer[0].terrain, x, z).vegetation;
            if (inner) {
                cover[j * row + i] = vegetation;
            }
            if (withCanopy) {
                // The canopy's top, somewhat below the tallest crowns, where a mean tree spans few pixels.
                float tree = treeMeanTreeHeight(configBuffer[0].terrain, configBuffer[0].placement,
                                                treeTerrainAltitude(configBuffer[0].terrain, vec3(x, h, z)));
                float pixels = treePixelDiameter(eye, vec3(x, h + 0.5f * tree, z), 0.5f * tree, focal);
                c = vegetation.x * treeCanopyWeight(band, pixels);
                h += 0.85f * tree * c;
            }
        }
        heights[k] = h;
        canopy[k] = c;
    }
    barrier();

    float spacing = float(nodeStep) * unit;
    float zoom = treeTerrainZoom(camera.proj);
    uint base = gl_WorkGroupID.x * uint(row * row);
    for (int v = int(gl_LocalInvocationID.x); v < row * row; v += 256) {
        int i = v % row;
        int j = v / row;
        float x = configBuffer[0].lod.root.x + float(gx + i * nodeStep) * unit;
        float z = configBuffer[0].lod.root.y + float(gz + j * nodeStep) * unit;
        float h = H(i, j);
        float k = treeTerrainMorph(configBuffer[0].lod, level, zoom * distance(eye, vec3(x, h, z)));
        vec3 fine = normalize(vec3(H(i - 1, j) - H(i + 1, j), 2.0f * spacing, H(i, j - 1) - H(i, j + 1)));
        bool oddI = (i & 1) == 1;
        bool oddJ = (j & 1) == 1;
        float coarseHeight = h;
        vec3 coarse;
        vec4 coarseCover = cover[v];
        float coarseCanopy = C(i, j);
        if (oddI && oddJ) {
            coarseHeight = 0.5f * (H(i - 1, j - 1) + H(i + 1, j + 1));
            coarse = coarseNormal(i - 1, j - 1, spacing) + coarseNormal(i + 1, j + 1, spacing);
            coarseCover = 0.5f * (cover[v - row - 1] + cover[v + row + 1]);
            coarseCanopy = 0.5f * (C(i - 1, j - 1) + C(i + 1, j + 1));
        } else if (oddI) {
            coarseHeight = 0.5f * (H(i - 1, j) + H(i + 1, j));
            coarse = coarseNormal(i - 1, j, spacing) + coarseNormal(i + 1, j, spacing);
            coarseCover = 0.5f * (cover[v - 1] + cover[v + 1]);
            coarseCanopy = 0.5f * (C(i - 1, j) + C(i + 1, j));
        } else if (oddJ) {
            coarseHeight = 0.5f * (H(i, j - 1) + H(i, j + 1));
            coarse = coarseNormal(i, j - 1, spacing) + coarseNormal(i, j + 1, spacing);
            coarseCover = 0.5f * (cover[v - row] + cover[v + row]);
            coarseCanopy = 0.5f * (C(i, j - 1) + C(i, j + 1));
        } else {
            coarse = coarseNormal(i, j, spacing);
        }
        vertices[base + uint(v)] = vec4(normalize(mix(fine, normalize(coarse), k)), mix(h, coarseHeight, k));
        covers[base + uint(v)] = mix(cover[v], coarseCover, k);
        float morphed = mix(h, coarseHeight, k);
        float sunlight = withShadows ? treeTerrainSunVisibility(configBuffer[0].terrain, vec3(x, morphed, z), sun) : 1.0f;
        extras[base + uint(v)] = vec4(mix(C(i, j), coarseCanopy, k), sunlight, 0.0f, 0.0f);
    }
}
