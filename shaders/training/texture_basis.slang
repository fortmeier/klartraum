// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Shared parts of the code-mixed feature texture (texture_basis.comp.slang and its backward pass): per face of the
// object a stack of layers [layers][channels][resolution][resolution], layer 0 the base and layer k + 1 scaled by the
// pixel's latent k; sampled bilinearly at (u, v) (texel centres at (i + 0.5) / resolution, clamped at the edges).

struct TextureBasisConstants {
    uint batch;
    uint pixels;
    uint faces;      // faces of the object; the face is floor(id x faces) of the third coordinate
    uint layers;     // 1 + the latents that mix in
    uint channels;   // features per layer
    uint resolution; // texels per side
    uint classCount;
    uint outputChannels; // channels of the output (the generator input)
    uint4 classes;       // the textured classes, the first classCount of them
    uint outputOffset;   // where the features go in the output
    uint accumulate;     // 1: add to the output (or the latent gradient) instead of writing it
};

// The pixel's face, -1 off the textured classes; its four texels (offsets into one channel plane) and their weights.
struct TexelQuad {
    int face;
    uint index[4];
    float weight[4];
};

TexelQuad texelQuad(TextureBasisConstants c, float u, float v, float id, float pixelClass) {
    TexelQuad quad;
    bool textured = false;
    for (uint k = 0u; k < c.classCount; ++k)
        textured = textured || c.classes[k] == uint(pixelClass + 0.5);
    quad.face = textured ? int(clamp(floor(id * float(c.faces)), 0.0, float(c.faces - 1u))) : -1;
    const float last = float(c.resolution - 1u);
    const float x = clamp(u * float(c.resolution) - 0.5, 0.0, last), y = clamp(v * float(c.resolution) - 0.5, 0.0, last);
    const uint x0 = uint(floor(x)), y0 = uint(floor(y));
    const uint x1 = min(x0 + 1u, c.resolution - 1u), y1 = min(y0 + 1u, c.resolution - 1u);
    const float ax = x - float(x0), ay = y - float(y0);
    quad.index[0] = y0 * c.resolution + x0;
    quad.index[1] = y0 * c.resolution + x1;
    quad.index[2] = y1 * c.resolution + x0;
    quad.index[3] = y1 * c.resolution + x1;
    quad.weight[0] = (1.0 - ax) * (1.0 - ay);
    quad.weight[1] = ax * (1.0 - ay);
    quad.weight[2] = (1.0 - ax) * ay;
    quad.weight[3] = ax * ay;
    return quad;
}

// Offset of channel `channel` of layer `layer` of face `face` in the table.
uint planeOf(TextureBasisConstants c, uint face, uint layer, uint channel) {
    return ((face * c.layers + layer) * c.channels + channel) * c.resolution * c.resolution;
}
