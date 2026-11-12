// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The temporal consistency loss (latent_renderer.flow_warp_loss()): the first render warped onto the second view by
// the flow (each of the second view's pixels' position in the first, in pixels with centres at i + 0.5; bilinear,
// clamped to the border) and weight x the mean over all of the second render's values of valid x the squared
// difference: every value's share of it and the gradient for the second render (the warped first is the target). One
// invocation per value of the second render.

struct PushConstants {
    uint batch;
    uint channels;
    uint height;
    uint width;
    float weight;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> first;      // [batch][channels][height][width]
[[vk::binding(1, 0)]] StructuredBuffer<float> second;     // [batch][channels][height][width]
[[vk::binding(2, 0)]] StructuredBuffer<float> flow;       // [batch][2][height][width]: x, y in the first
[[vk::binding(3, 0)]] StructuredBuffer<float> valid;      // [batch][height][width]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> gradient; // of the second
[[vk::binding(5, 0)]] RWStructuredBuffer<float> shares;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint pixels = pc.height * pc.width, count = pc.batch * pc.channels * pixels;
    if (id.x >= count) return;
    const uint p = id.x % pixels, c = (id.x / pixels) % pc.channels, n = id.x / (pixels * pc.channels);
    const float mask = valid[n * pixels + p];
    if (mask == 0.0) {
        gradient[id.x] = 0.0;
        shares[id.x] = 0.0;
        return;
    }
    // grid_sample's border padding: the position clamped to the outer pixels' centres.
    const float x = clamp(flow[(n * 2u) * pixels + p] - 0.5, 0.0, float(pc.width - 1u));
    const float y = clamp(flow[(n * 2u + 1u) * pixels + p] - 0.5, 0.0, float(pc.height - 1u));
    const uint x0 = uint(x), y0 = uint(y);
    const uint x1 = min(x0 + 1u, pc.width - 1u), y1 = min(y0 + 1u, pc.height - 1u);
    const float tx = x - float(x0), ty = y - float(y0);
    const uint plane = (n * pc.channels + c) * pixels;
    const float top = (1.0 - tx) * first[plane + y0 * pc.width + x0] + tx * first[plane + y0 * pc.width + x1];
    const float bottom = (1.0 - tx) * first[plane + y1 * pc.width + x0] + tx * first[plane + y1 * pc.width + x1];
    const float d = second[id.x] - ((1.0 - ty) * top + ty * bottom);
    gradient[id.x] = mask * 2.0 * pc.weight * d / float(count);
    shares[id.x] = mask * pc.weight * d * d / float(count);
}
