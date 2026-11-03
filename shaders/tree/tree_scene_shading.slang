// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Shading of the static scene surfaces (pot, soil, terrain) by their material
// (TREE_MATERIAL_* in the w of `color`): fine detail that fades out where it
// would be finer than about two pixels, moss cushions, snow, the light volume,
// the point light and the glaze's highlight; captured objects are tinted instead. The includer includes
// tree_render_common.glsl, tree_shading.glsl and tree_light.glsl before.

vec4 treeSceneColor(vec3 worldPos, vec3 surfaceNormal, vec4 color, TreeRenderSettings s) {
    vec3 normal = normalize(surfaceNormal);
    if (color.w == TREE_MATERIAL_CAPTURED) {
        // Shaded like the captured Gaussians it stands among (tint_gaussians.comp).
        float shade = treeCapturedShade(s, treeCapturedLightAt(s, worldPos));
        return vec4(treeCapturedColor(color.rgb, normal, worldPos, shade, s), 1.0f);
    }
    float detail = clamp(1.5f - 140.0f * length(fwidth(worldPos)), 0.0f, 1.0f);
    vec3 albedo = treeSceneAlbedoDetail(color, worldPos, detail);
    if (color.w == TREE_MATERIAL_MOSS && detail > 0.0f) {
        // The relief of the moss cushions tilts the normal.
        float e = 0.002f;
        float h = treeMossHeight(worldPos);
        float dx = (treeMossHeight(worldPos + vec3(e, 0.0f, 0.0f)) - h) / e;
        float dz = (treeMossHeight(worldPos + vec3(0.0f, 0.0f, e)) - h) / e;
        normal = normalize(normal - detail * vec3(dx, 0.0f, dz));
    }
    albedo = mix(albedo, vec3(0.8f), treeSnowOn(normalize(surfaceNormal), worldPos, s.season.y));
    // The pot is one of the static occluders, which would shadow its own walls at grazing light.
    vec2 light = color.w == TREE_MATERIAL_GLAZE ? treeLightOfTreesAt(s, worldPos + 0.5f * s.lightGridMin.w * normal)
                                               : treeLightAt(s, worldPos + 0.5f * s.lightGridMin.w * normal);
    vec3 result = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, worldPos, s);
    if (color.w == TREE_MATERIAL_GLAZE) {
        // The glaze's highlight.
        vec3 toCamera = normalize(camera.cameraWorldPos.xyz - worldPos);
        vec3 halfway = normalize(toCamera + s.sun.xyz);
        result += vec3(0.35f) * pow(max(dot(normal, halfway), 0.0f), 60.0f) * s.sun.w * light.x * s.lightColor.rgb;
    }
    return treeOutputColor(result, s);
}
