// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Blends Gaussian i of set A into Gaussian i of set B (GaussianMorph). The
// position follows the path of the variant, sizes blend in log space,
// orientations by slerp, colour, opacity and SH coefficients linearly. The
// progress is eased in and out; at eased progress 0 the output is A and at 1
// it is B, copied unchanged. The explosion variants move in two phases: a
// burst that starts fast and comes to rest at a point around A, then an
// eased flight from there to B, so the motion has no kink at the rest point.
import scalar_buffer;

struct PushConstants {
    uint count;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] ScalarBuffer<float3> posA;
[[vk::binding(1, 0)]] ScalarBuffer<float4> rotA;
[[vk::binding(2, 0)]] ScalarBuffer<float3> scaleA;
[[vk::binding(3, 0)]] ScalarBuffer<float4> colAlphaA;
[[vk::binding(4, 0)]] ScalarBuffer<float> shRA;
[[vk::binding(5, 0)]] ScalarBuffer<float> shGA;
[[vk::binding(6, 0)]] ScalarBuffer<float> shBA;
[[vk::binding(7, 0)]] ScalarBuffer<float3> posB;
[[vk::binding(8, 0)]] ScalarBuffer<float4> rotB;
[[vk::binding(9, 0)]] ScalarBuffer<float3> scaleB;
[[vk::binding(10, 0)]] ScalarBuffer<float4> colAlphaB;
[[vk::binding(11, 0)]] ScalarBuffer<float> shRB;
[[vk::binding(12, 0)]] ScalarBuffer<float> shGB;
[[vk::binding(13, 0)]] ScalarBuffer<float> shBB;
[[vk::binding(14, 0)]] ScalarBuffer<float> params;
[[vk::binding(15, 0)]] ScalarBuffer<float> delays;
[[vk::binding(16, 0)]] RWScalarBuffer<float3> outPos;
[[vk::binding(17, 0)]] RWScalarBuffer<float4> outRot;
[[vk::binding(18, 0)]] RWScalarBuffer<float3> outScale;
[[vk::binding(19, 0)]] RWScalarBuffer<float4> outColAlpha;
[[vk::binding(20, 0)]] RWScalarBuffer<float> outShR;
[[vk::binding(21, 0)]] RWScalarBuffer<float> outShG;
[[vk::binding(22, 0)]] RWScalarBuffer<float> outShB;

// Layout of MorphParams::values().
static const int kProgress = 0;
static const int kVariant = 1;
static const int kSwirlTurns = 2;
static const int kScatterRadius = 3;
static const int kWaveSpread = 4;
static const int kCentre = 5; // 3 floats
static const int kAxis = 8;   // 3 floats
static const int kSeed = 11;
static const int kExplodeRadius = 12;
static const int kExplodeShare = 13;
static const int kDirection = 14; // 3 floats

static const int kStraight = 0;
static const int kSwirl = 1;
static const int kScatter = 2;
static const int kWave = 3;
static const int kExplode = 4;
static const int kExplodeSwirl = 5;
static const int kShatter = 6;
static const int kSwirlScatter = 7;
static const int kSwirlWave = 8;

static const float kPi = 3.14159265358979;

float easeInOutCubic(float s) {
    return s < 0.5 ? 4.0 * s * s * s : 1.0 - pow(-2.0 * s + 2.0, 3.0) / 2.0;
}

float easeOutCubic(float s) {
    float u = 1.0 - s;
    return 1.0 - u * u * u;
}

// PCG hash, mapped to [0, 1).
float random01(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) * (1.0 / 4294967296.0);
}

float4 slerp(float4 a, float4 b, float s) {
    float d = dot(a, b);
    if (d < 0.0) {
        b = -b;
        d = -d;
    }
    if (d > 0.9995) {
        return normalize(lerp(a, b, s));
    }
    float theta = acos(clamp(d, -1.0, 1.0));
    return normalize((sin((1.0 - s) * theta) * a + sin(s * theta) * b) / sin(theta));
}

// Position on a spiral around the axis: distance from the axis, height along
// it and angle around it go from a to b, the angle with `turns` extra turns.
// `turns` must be whole, or the spiral would not end at b.
float3 swirlPosition(float3 a, float3 b, float s, float3 centre, float3 axis, float turns) {
    float3 other = abs(axis.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    float3 e1 = normalize(cross(axis, other));
    float3 e2 = cross(axis, e1);
    float3 va = a - centre;
    float3 vb = b - centre;
    float ha = dot(va, axis);
    float hb = dot(vb, axis);
    float3 ra = va - ha * axis;
    float3 rb = vb - hb * axis;
    float angleA = atan2(dot(ra, e2), dot(ra, e1));
    float angleB = atan2(dot(rb, e2), dot(rb, e1));
    float delta = angleB - angleA;
    delta -= 2.0 * kPi * floor((delta + kPi) / (2.0 * kPi)); // shortest way, in [-pi, pi)
    float angle = angleA + s * (delta + 2.0 * kPi * turns);
    float radius = lerp(length(ra), length(rb), s);
    return centre + lerp(ha, hb, s) * axis + radius * (cos(angle) * e1 + sin(angle) * e2);
}

// A random unit vector, uniform on the sphere.
float3 randomDirection(uint key) {
    float z = 2.0 * random01(key) - 1.0;
    float phi = 2.0 * kPi * random01(key + 1u);
    float r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(phi), r * sin(phi), z);
}

// Where Gaussian i comes to rest after the burst: between half and the full
// radius from its start, away from the centre (Explode, ExplodeSwirl) or
// towards `direction` (Shatter), spread by a random direction.
float3 burstPoint(float3 a, int variant, uint i, uint seed) {
    uint key = i * 5u + 0x68E31DA4u + seed * 0x9E3779B9u;
    float3 spread = randomDirection(key);
    float3 heading;
    if (variant == kShatter) {
        heading = normalize(float3(params[kDirection], params[kDirection + 1], params[kDirection + 2]));
        heading = normalize(heading + 0.8 * spread);
    } else {
        float3 away = a - float3(params[kCentre], params[kCentre + 1], params[kCentre + 2]);
        heading = length(away) > 1e-6 ? normalize(normalize(away) + 0.5 * spread) : spread;
    }
    return a + heading * params[kExplodeRadius] * (0.5 + 0.5 * random01(key + 2u));
}

// `p` turned by `angle` around the axis through `centre` (Rodrigues).
float3 turnAround(float3 p, float3 centre, float3 axis, float angle) {
    float3 v = p - centre;
    float c = cos(angle);
    return centre + v * c + cross(axis, v) * sin(angle) + axis * dot(axis, v) * (1.0 - c);
}

// Quadratic Bezier through a random point around the middle of the straight
// path, between half and the full radius away from it.
float3 scatterPosition(float3 a, float3 b, float s, float radius, uint i, uint seed) {
    uint key = i * 3u + seed * 0x9E3779B9u;
    float3 direction = randomDirection(key);
    float3 control = 0.5 * (a + b) + direction * radius * (0.5 + 0.5 * random01(key + 2u));
    float u = 1.0 - s;
    return u * u * a + 2.0 * u * s * control + s * s * b;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint i = globalId.x;
    if (i >= pc.count) {
        return;
    }
    int variant = int(params[kVariant] + 0.5);
    float progress = params[kProgress];
    if (variant == kWave || variant == kSwirlWave) {
        float spread = clamp(params[kWaveSpread], 0.0, 0.99);
        progress = (progress - delays[i] * spread) / (1.0 - spread);
    }
    float s = easeInOutCubic(clamp(progress, 0.0, 1.0));

    if (s <= 0.0 || s >= 1.0) {
        bool atEnd = s >= 1.0;
        outPos[i] = atEnd ? posB[i] : posA[i];
        outRot[i] = atEnd ? rotB[i] : rotA[i];
        outScale[i] = atEnd ? scaleB[i] : scaleA[i];
        outColAlpha[i] = atEnd ? colAlphaB[i] : colAlphaA[i];
        for (uint k = 0u; k < 15u; ++k) {
            uint j = k * pc.count + i;
            outShR[j] = atEnd ? shRB[j] : shRA[j];
            outShG[j] = atEnd ? shGB[j] : shGA[j];
            outShB[j] = atEnd ? shBB[j] : shBA[j];
        }
        return;
    }

    float3 a = posA[i];
    float3 b = posB[i];
    float3 position;
    float3 centre = float3(params[kCentre], params[kCentre + 1], params[kCentre + 2]);
    float3 axis = normalize(float3(params[kAxis], params[kAxis + 1], params[kAxis + 2]));
    float turns = round(params[kSwirlTurns]);
    if (variant == kSwirl || variant == kSwirlWave) {
        position = swirlPosition(a, b, s, centre, axis, turns);
    } else if (variant == kSwirlScatter) {
        float3 scattered = scatterPosition(a, b, s, params[kScatterRadius], i, uint(params[kSeed]));
        position = turnAround(scattered, centre, axis, 2.0 * kPi * turns * s);
    } else if (variant == kExplode || variant == kExplodeSwirl || variant == kShatter) {
        float share = clamp(params[kExplodeShare], 0.05, 0.95);
        float p = clamp(progress, 0.0, 1.0);
        float3 rest = burstPoint(a, variant, i, uint(params[kSeed]));
        if (p < share) {
            position = lerp(a, rest, easeOutCubic(p / share));
        } else {
            float q = easeInOutCubic((p - share) / (1.0 - share));
            position = variant == kExplodeSwirl ? swirlPosition(rest, b, q, centre, axis, turns)
                                                : lerp(rest, b, q);
        }
    } else if (variant == kScatter) {
        position = scatterPosition(a, b, s, params[kScatterRadius], i, uint(params[kSeed]));
    } else {
        position = lerp(a, b, s);
    }

    outPos[i] = position;
    outRot[i] = slerp(rotA[i], rotB[i], s);
    outScale[i] = exp(lerp(log(max(scaleA[i], float3(1e-12))), log(max(scaleB[i], float3(1e-12))), s));
    outColAlpha[i] = lerp(colAlphaA[i], colAlphaB[i], s);
    for (uint k = 0u; k < 15u; ++k) {
        uint j = k * pc.count + i;
        outShR[j] = lerp(shRA[j], shRB[j], s);
        outShG[j] = lerp(shGA[j], shGB[j], s);
        outShB[j] = lerp(shBA[j], shBB[j], s);
    }
}
