// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Specular reflection of a textured mesh (specular.comp.slang, specular_backward.comp.slang), added to its diffuse
// shading (differentiable_shading.slang) so that photographs' highlights are explained rather than baked into the
// texture. Per texel the material holds an offset of the surface normal (a normal map, in object space) and the
// specular amplitude a; the environment is LIGHTS directions d_k with amplitudes b_k and one exponent e, shared by all
// views. A covered pixel adds the grey
//   a^2 sum_k b_k^2 max(dot(r, normalize(d_k)), 0)^e,   r = reflect(-v, normalize(n + offset)),   e = 6 + exp(x),
// with x the learned exponent parameter (the floor keeps the term a highlight, not a broad brightening),
// with n the smooth normal (the vertex normals interpolated perspective-correctly) and v the direction to the view's
// eye. Slang's bwd_diff gives the gradients of the material (through the bilinear lookup's weights) and of the lights.
module specular;

import differentiable_shading;

public static const uint LIGHTS = 4;

[[vk::binding(8, 0)]] public StructuredBuffer<float4> material;      // per texel: normal offset xyz, amplitude
[[vk::binding(9, 0)]] public StructuredBuffer<float4> lights;        // LIGHTS (direction xyz, amplitude), (x)
[[vk::binding(10, 0)]] public StructuredBuffer<float4> vertexNormals; // per vertex: sum of area normals, xyz
[[vk::binding(11, 0)]] public StructuredBuffer<float4> eyes;         // per view: the camera's position, xyz

// The specular grey of a surface with smooth normal `normal` seen along `toEye` (unit, towards the eye).
[Differentiable]
public float specularOf(float3 offset, float amplitude, float4 l0, float4 l1, float4 l2, float4 l3,
                        float logExponent, no_diff float3 normal, no_diff float3 toEye) {
    const float3 n = normalize(normal + offset);
    const float3 r = 2.0 * dot(n, toEye) * n - toEye;
    const float e = 6.0 + exp(logExponent);
    float sum = 0.0;
    [ForceUnroll]
    for (int k = 0; k < 4; ++k) {
        const float4 l = k == 0 ? l0 : (k == 1 ? l1 : (k == 2 ? l2 : l3));
        sum += l.w * l.w * pow(max(dot(r, normalize(l.xyz)), 1e-4), e);
    }
    return amplitude * amplitude * sum;
}

// What the specular term of a covered pixel reads: the smooth normal, the direction to the eye, and the bilinear
// taps of the material at the pixel's texture coordinate.
public struct SpecularSurface {
    public float3 normal;
    public float3 toEye;
    public BilinearTaps taps;
};

public SpecularSurface specularSurface(PixelSurface s) {
    const float2 size = float2(pc.width, pc.height);
    float4 clip[3];
    float2 screen[3];
    for (uint c = 0; c < 3; ++c) {
        clip[c] = mul(s.camera, float4(s.corner[c], 1.0));
        screen[c] = (clip[c].xy / clip[c].w * 0.5 + 0.5) * size;
    }
    const float2 p = s.pixel;
    const float area = (screen[1].x - screen[0].x) * (screen[2].y - screen[0].y) -
                       (screen[1].y - screen[0].y) * (screen[2].x - screen[0].x);
    const float l0 = ((screen[1].x - p.x) * (screen[2].y - p.y) - (screen[1].y - p.y) * (screen[2].x - p.x)) / area;
    const float l1 = ((screen[2].x - p.x) * (screen[0].y - p.y) - (screen[2].y - p.y) * (screen[0].x - p.x)) / area;
    float3 q = float3(l0 / clip[0].w, l1 / clip[1].w, (1.0 - l0 - l1) / clip[2].w);
    q /= q.x + q.y + q.z;
    SpecularSurface result;
    result.normal = normalize(q.x * vertexNormals[s.vertex[0]].xyz + q.y * vertexNormals[s.vertex[1]].xyz +
                              q.z * vertexNormals[s.vertex[2]].xyz);
    const float3 point = q.x * s.corner[0] + q.y * s.corner[1] + q.z * s.corner[2];
    result.toEye = normalize(eyes[viewIndex[0]].xyz - point);
    const float2 uv = textureCoordinate(s.corner[0], s.corner[1], s.corner[2], s.cornerUv[0], s.cornerUv[1],
                                        s.cornerUv[2], s.camera, s.pixel, size);
    result.taps = bilinearTaps(uv);
    return result;
}

// The material at a surface: the taps blended by their bilinear weights.
public float4 materialAt(SpecularSurface surface) {
    const float4 w = bilinearWeights(surface.taps.fraction);
    return w.x * material[surface.taps.index[0]] + w.y * material[surface.taps.index[1]] +
           w.z * material[surface.taps.index[2]] + w.w * material[surface.taps.index[3]];
}
