// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Relights Gaussians (GaussianLighting): the captured SH degree-0 colour is
// the albedo; ambient light is weighted by the splat's average visibility,
// each point light by the visibility towards it (unless it is unshadowed), a
// Lambert term and a GGX microfacet term; splats near a glowing light emit. The lit colour replaces the captured one, blended by the
// relight value; at relight 0 the colour passes through unchanged.
import scalar_buffer;

struct Counts {
    uint4 counts; // x: number of Gaussians
};
[[vk::push_constant]] ConstantBuffer<Counts> pc;

struct Material {
    float4 normalRoughness;
    float4 visibility0;
    float4 visibility1;
    float4 parameters; // x: visibility coefficient 8, y: metalness, z: specular scale
};

struct Light {
    float4 positionRange;
    float4 colourIntensity;
    float4 shape; // x: reference distance, y: radius, z: 1 = unshadowed, w: glow radius
};

struct Frame {
    float4 camera;
    float4 ambient;
    uint4 counts;
    float4 control;
    Light lights[8];
};

[[vk::binding(0, 0)]] ScalarBuffer<float3> positions;
[[vk::binding(1, 0)]] ScalarBuffer<float4> colours;
[[vk::binding(2, 0)]] ScalarBuffer<Material> materials;
// A single Frame.
[[vk::binding(3, 0)]] ScalarBuffer<Frame> frames;
[[vk::binding(4, 0)]] RWScalarBuffer<float4> lit;

static const float kC0 = 0.282094792;
static const float kPi = 3.14159265358979;

float visibility(Material m, float3 d) {
    float4 b0 = float4(0.282094792, 0.488602512 * d.y, 0.488602512 * d.z, 0.488602512 * d.x);
    float4 b1 = float4(1.092548431 * d.x * d.y, 1.092548431 * d.y * d.z, 0.315391565 * (3.0 * d.z * d.z - 1.0),
                       1.092548431 * d.x * d.z);
    float b8 = 0.546274215 * (d.x * d.x - d.y * d.y);
    return clamp(dot(m.visibility0, b0) + dot(m.visibility1, b1) + m.parameters.x * b8, 0.0, 1.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint i = globalId.x;
    if (i >= pc.counts.x) {
        return;
    }
    float4 c = colours[i];
    float relight = clamp(frames[0].control.x, 0.0, 1.0);
    if (relight <= 0.0) {
        lit[i] = c;
        return;
    }

    Material m = materials[i];
    float3 albedo = max(c.rgb * kC0 + 0.5, float3(0.0));
    float3 p = positions[i];
    float3 n = m.normalRoughness.xyz;
    float3 toCamera = frames[0].camera.xyz - p;
    float3 v = toCamera * rsqrt(max(dot(toCamera, toCamera), 1e-12));
    // Thin translucent splats scatter light on either face.
    if (c.a < 0.8 && dot(n, v) < 0.0) {
        n = -n;
    }

    float ao = clamp(m.visibility0.x * kC0, 0.0, 1.0);
    float3 result = albedo * frames[0].ambient.rgb * ao;

    float rough = max(m.normalRoughness.w, 0.05);
    float a2 = rough * rough * rough * rough;
    float metal = m.parameters.y;
    float specularScale = m.parameters.z;
    float3 f0 = lerp(float3(0.04), albedo, metal);
    for (uint j = 0u; j < min(frames[0].counts.x, 8u); ++j) {
        Light light = frames[0].lights[j];
        float3 delta = light.positionRange.xyz - p;
        float d2 = max(dot(delta, delta), 1e-12);
        float distance = sqrt(d2);
        float range = light.positionRange.w;
        if (range > 0.0 && distance >= range) {
            continue;
        }
        float glowRadius = light.shape.w;
        if (glowRadius > 0.0 && distance < glowRadius) {
            float fade = 1.0 - distance / glowRadius;
            result += light.colourIntensity.rgb * light.colourIntensity.w * fade * fade;
        }
        float3 l = delta / distance;
        float nl = dot(n, l);
        if (nl <= 0.0) {
            continue;
        }
        float visible = light.shape.z > 0.5 ? 1.0 : visibility(m, l);
        if (visible <= 0.0) {
            continue;
        }
        float reference = light.shape.x;
        float attenuation = reference * reference / max(d2, light.shape.y * light.shape.y);
        if (range > 0.0) {
            attenuation *= pow(max(1.0 - pow(distance / range, 4.0), 0.0), 2.0);
        }
        float3 diffuse = (1.0 - metal) * albedo / kPi;
        float3 specular = float3(0.0);
        if (specularScale > 0.0) {
            float3 h = normalize(l + v);
            float nh = max(dot(n, h), 0.0);
            float nv = max(dot(n, v), 0.001);
            float vh = max(dot(v, h), 0.0);
            float denom = nh * nh * (a2 - 1.0) + 1.0;
            float distribution = a2 / (kPi * denom * denom);
            float k = (rough + 1.0) * (rough + 1.0) / 8.0;
            float geometry = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
            float3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
            specular = specularScale * distribution * geometry * fresnel / max(4.0 * nv * nl, 0.001);
        }
        result += (diffuse + specular) * light.colourIntensity.rgb * light.colourIntensity.w * attenuation * nl * visible;
    }

    float3 shaded = (max(result, float3(0.0)) - 0.5) / kC0;
    lit[i] = float4(lerp(c.rgb, shaded, relight), c.a);
}
