#include "klartraum/simple_text_renderer.hpp"

#include "klartraum/glyph_atlas.hpp"

namespace klartraum {

void SimpleTextRenderer::setText(const std::string& text, float x, float y, float scale, float r, float g, float b,
                                 float a) {
    vertices.clear();
    indices.clear();
    vertices.reserve(text.size() * 4);
    indices.reserve(text.size() * 6);

    const float glyphWidth = static_cast<float>(GLYPH_PIXEL_WIDTH) * scale;
    const float glyphHeight = static_cast<float>(GLYPH_PIXEL_HEIGHT) * scale;
    const float advance = static_cast<float>(GLYPH_CELL_WIDTH) * scale;

    for (size_t i = 0; i < text.size(); ++i) {
        const float originX = x + static_cast<float>(i) * advance;
        const GlyphUVRect uv = glyphUVRect(static_cast<unsigned char>(text[i]));

        const uint32_t base = static_cast<uint32_t>(vertices.size());

        // top-left, top-right, bottom-right, bottom-left
        vertices.push_back({originX, y, uv.u0, uv.v0, r, g, b, a});
        vertices.push_back({originX + glyphWidth, y, uv.u1, uv.v0, r, g, b, a});
        vertices.push_back({originX + glyphWidth, y + glyphHeight, uv.u1, uv.v1, r, g, b, a});
        vertices.push_back({originX, y + glyphHeight, uv.u0, uv.v1, r, g, b, a});

        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
        indices.push_back(base + 0);
    }
}

}  // namespace klartraum
