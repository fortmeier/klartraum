// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Colour output of the classical tree renderer; the lighting is treeShade() in tree_shared.hpp.

vec4 treeOutputColor(vec3 linearColor) {
    return vec4(treeDisplayColor(linearColor), 1.0f);
}
