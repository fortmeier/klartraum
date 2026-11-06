// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// StyleTransfer's network output (3 planes at half the frame's size) resized to the frame bilinearly as PyTorch's
// interpolate does (align_corners=False), clamped to [0, 1], and blended over the frame's colour by blend.x.

struct Push {
    uint frameWidth;
    uint frameHeight;
    uint width;
    uint height;
    uint channels;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> network;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> colorImage;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] WTexture2D<float4> outputImage;
[[vk::binding(3, 0)]] StructuredBuffer<vec4> blend;

float sampleAt(uint plane, int x, int y) {
    return network[plane * pc.width * pc.height + uint(y) * pc.width + uint(x)];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= pc.frameWidth || pixel.y >= pc.frameHeight) {
        return;
    }
    float sx = max((float(pixel.x) + 0.5f) * float(pc.width) / float(pc.frameWidth) - 0.5f, 0.0f);
    float sy = max((float(pixel.y) + 0.5f) * float(pc.height) / float(pc.frameHeight) - 0.5f, 0.0f);
    int x0 = int(sx), y0 = int(sy);
    int x1 = min(x0 + 1, int(pc.width) - 1), y1 = min(y0 + 1, int(pc.height) - 1);
    float fx = sx - float(x0), fy = sy - float(y0);
    vec3 styled;
    for (uint c = 0u; c < 3u; ++c) {
        float top = mix(sampleAt(c, x0, y0), sampleAt(c, x1, y0), fx);
        float bottom = mix(sampleAt(c, x0, y1), sampleAt(c, x1, y1), fx);
        styled[c] = mix(top, bottom, fy);
    }
    vec4 color = colorImage[ivec2(pixel)];
    outputImage.Store(ivec2(pixel), vec4(mix(color.rgb, clamp(styled, 0.0f, 1.0f), blend[0].x), 1.0f));
}
