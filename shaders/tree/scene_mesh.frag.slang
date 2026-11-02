// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.slang"
#include "tree_shading.slang"
#define TREE_OCCLUSION_BINDING 2
#include "tree_occlusion.slang"
#define TREE_LIGHT_SET 1
#define TREE_LIGHT_BINDING 3
#include "tree_light.slang"
#include "tree_scene_shading.slang"

[[vk::binding(1, 1)]] StructuredBuffer<TreeRenderSettings> settings;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 outColor;

[shader("fragment")]
void main() {
    TreeRenderSettings s = settings[0];
    if (treeBehindScene(s)) {
        discard;
    }
    outColor = treeSceneColor(inWorldPos, inNormal, inColor, s);
}
