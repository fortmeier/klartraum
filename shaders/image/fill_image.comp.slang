// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Fills an image with a colour read from a buffer, so that the colour can change per submission of a recorded
// graph (e.g. a background that follows the time of day).

[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> image;
[[vk::binding(1, 0)]] StructuredBuffer<float4> colour; // one element

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint width, height;
    image.GetDimensions(width, height);
    if (globalId.x < width && globalId.y < height) image[globalId.xy] = colour[0];
}
