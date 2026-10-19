// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Colour output of the classical tree renderer; the lighting is treeShade() in tree_shared.hpp.

vec4 treeOutputColor(vec3 linearColor, TreeRenderSettings s) {
    return vec4(treeDisplayColor(linearColor, s.sky.w), 1.0f);
}
