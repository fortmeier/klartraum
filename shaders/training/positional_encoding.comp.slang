// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The positional encoding of surface coordinates (latent_renderer.positional_encoding()): per coordinate c of (u, v,
// id) and frequency f < frequencies the pair sin(2^f pi x_c), cos(2^f pi x_c), then the three raw coordinates; zero
// where the pixel's class is none of the texture's classes. One invocation per pixel.

struct PushConstants {
    uint batch;
    uint pixels;
    uint frequencies;
    uint classCount;
    uint4 classes; // the texture's classes, the first classCount of them
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> coordinates;  // [batch][3][pixels]
[[vk::binding(1, 0)]] StructuredBuffer<float> classes;      // [batch][pixels]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> encoding;   // [batch][3 (2 frequencies + 1)][pixels]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.batch * pc.pixels) return;
    const uint n = id.x / pc.pixels, p = id.x % pc.pixels;
    const uint pixelClass = uint(classes[id.x] + 0.5);
    bool textured = false;
    for (uint k = 0u; k < pc.classCount; ++k)
        textured = textured || pc.classes[k] == pixelClass;
    const float mask = textured ? 1.0 : 0.0;
    const uint channels = 3u * (2u * pc.frequencies + 1u);
    for (uint c = 0u; c < 3u; ++c) {
        const float x = coordinates[(n * 3u + c) * pc.pixels + p];
        for (uint f = 0u; f < pc.frequencies; ++f) {
            const float angle = float(1u << f) * 3.14159265358979 * x;
            const uint channel = (c * pc.frequencies + f) * 2u;
            encoding[(n * channels + channel) * pc.pixels + p] = mask * sin(angle);
            encoding[(n * channels + channel + 1u) * pc.pixels + p] = mask * cos(angle);
        }
        encoding[(n * channels + 6u * pc.frequencies + c) * pc.pixels + p] = mask * x;
    }
}
