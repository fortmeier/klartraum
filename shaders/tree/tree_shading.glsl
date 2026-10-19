// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Lighting of the classical tree renderer: a sun and a sky/ground ambient term.

vec3 treeShade(vec3 albedo, vec3 normal, TreeRenderSettings s) {
    float sun = max(dot(normal, s.sun.xyz), 0.0f) * s.sun.w;
    float hemisphere = 0.5f + 0.5f * normal.y;
    vec3 ambient = mix(s.ground.rgb, s.sky.rgb, hemisphere);
    return albedo * (ambient + vec3(sun));
}

vec4 treeOutputColor(vec3 linearColor) {
    // The target stores the values as they are displayed.
    return vec4(pow(clamp(linearColor, 0.0f, 1.0f), vec3(1.0f / 2.2f)), 1.0f);
}
