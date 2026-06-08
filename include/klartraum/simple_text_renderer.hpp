#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace klartraum {

// Vertex format consumed by the text shaders: a 2D screen-space position,
// an atlas UV coordinate, and an RGBA color (replicated per-vertex so the
// fragment shader can simply multiply the sampled glyph alpha by it).
struct TextVertex {
    float x, y;
    float u, v;
    float r, g, b, a;
};

// Builds (and rebuilds, on demand) the quad-per-glyph mesh for a text
// string. The caller manages line breaks and alignment; this class only
// lays out a single line, advancing one fixed glyph cell per character.
//
// Each glyph becomes a quad of 4 vertices (top-left, top-right,
// bottom-right, bottom-left) and 6 indices (two triangles), sized to the
// glyph's pixel rectangle (GLYPH_PIXEL_WIDTH x GLYPH_PIXEL_HEIGHT) scaled by
// `scale`, with consecutive glyphs advanced by GLYPH_CELL_WIDTH x `scale`.
class SimpleTextRenderer {
public:
    // Regenerates the mesh for `text`, anchored with its top-left corner at
    // screen-space position (x, y), scaled by `scale`, and colored (r, g, b, a).
    void setText(const std::string& text, float x, float y, float scale, float r, float g, float b, float a);

    const std::vector<TextVertex>& getVertices() const { return vertices; }
    const std::vector<uint32_t>& getIndices() const { return indices; }

private:
    std::vector<TextVertex> vertices;
    std::vector<uint32_t> indices;
};

}  // namespace klartraum
