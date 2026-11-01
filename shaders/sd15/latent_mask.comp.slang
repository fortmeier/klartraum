// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The latent mask of the close-up (closeUpLatentMask() on the GPU): a latent
// pixel is generated (1) where its block or a neighbouring block holds a pixel
// of the body (the paste mask), half generated (0.5) where that holds only for
// the mask with margin, and kept (0) elsewhere. A pixel is set where its red
// channel is not zero.

struct PushConstants {
    uint width; // image size in pixels
    uint height;
    uint factor;   // pixels per latent pixel
    uint dilation; // latent pixels
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> marginMask;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> bodyMask;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> latentMask;

bool anySet(bool body, int2 lower, int2 upper) {
    for (int y = lower.y; y < upper.y; ++y) {
        for (int x = lower.x; x < upper.x; ++x) {
            float red = body ? bodyMask.Load(int2(x, y)).r : marginMask.Load(int2(x, y)).r;
            if (red > 0.0) {
                return true;
            }
        }
    }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint latentWidth = pc.width / pc.factor, latentHeight = pc.height / pc.factor;
    int2 latent = int2(globalId.xy);
    if (latent.x >= int(latentWidth) || latent.y >= int(latentHeight)) {
        return;
    }
    int reach = int(pc.dilation), factor = int(pc.factor);
    int2 lower = max(latent - reach, int2(0)) * factor;
    int2 upper = min(latent + reach + 1, int2(latentWidth, latentHeight)) * factor;
    float value = anySet(true, lower, upper) ? 1.0 : anySet(false, lower, upper) ? 0.5 : 0.0;
    latentMask[uint(latent.y) * latentWidth + uint(latent.x)] = value;
}
