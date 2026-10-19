// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Leaf level of detail, shared by the leaf classification pass and the leaf
// vertex shader so that both make the same decision. Needs tree_render_common.glsl.

// Draw buckets: full detail with Bezier edges, and the simplified polygon.
#define TREE_LEAF_BUCKETS 2u

// Vertices of one half lobe: 16 inner triangles and 4 curve triangles at full
// detail, 8 inner triangles when simplified.
uint treeLeafHalfVertices(uint bucket) {
    return bucket == 0u ? 60u : 24u;
}

float treeLeafPixels(LeafRecord leaf, TreeRenderSettings s) {
    vec3 center = leaf.posScale.xyz + 0.5f * leaf.posScale.w * qforward(leaf.rot);
    return leaf.posScale.w * treeFocalPixels(s) / max(treeViewDepth(center), 1e-4f);
}

// x: flatten in [0, 1], y: size factor (0: not drawn).
vec2 treeLeafLodOf(LeafRecord leaf, TreeRenderSettings s) {
    return treeLeafLod(treeLeafPixels(leaf, s), s.leafLod, treeRand(leaf.ids.y, 0x77u));
}
