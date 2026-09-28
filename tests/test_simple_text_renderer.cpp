/**
 * TESTS:
 * - emptyStringProducesEmptyMesh: an empty string yields no vertices and no indices
 * - singleCharacterProducesOneQuad: a single character yields exactly 4 vertices and 6 indices forming two triangles
 * - vertexCountsScaleWithStringLength: vertex/index counts are 4N and 6N for an N-character string
 * - quadPositionsAdvanceByGlyphCellWidth: consecutive glyph quads are offset by GLYPH_CELL_WIDTH * scale
 * - quadSizeMatchesGlyphPixelDimensions: each quad spans GLYPH_PIXEL_WIDTH x GLYPH_PIXEL_HEIGHT scaled
 * - uvCoordinatesMatchGlyphAtlasLookup: each quad's UVs match GlyphAtlas::glyphUVRect for its character
 * - colorIsPropagatedToAllVertices: the requested RGBA color is written to all four vertices of every quad
 * - outOfRangeCharacterUsesMissingGlyphUv: characters outside ASCII [32, 126] use the missing-glyph UV rect
 **/

#include <gtest/gtest.h>

#include "klartraum/glyph_atlas.hpp"
#include "klartraum/simple_text_renderer.hpp"

using namespace klartraum;

namespace {

// Checks that `quad` (4 vertices starting at `base`) is a screen-space
// rectangle with top-left at (x, y) and the given width/height.
void expectQuadGeometry(const std::vector<TextVertex>& vertices, size_t base, float x, float y, float width,
                        float height) {
    ASSERT_GE(vertices.size(), base + 4);

    const TextVertex& topLeft = vertices[base + 0];
    const TextVertex& topRight = vertices[base + 1];
    const TextVertex& bottomRight = vertices[base + 2];
    const TextVertex& bottomLeft = vertices[base + 3];

    EXPECT_FLOAT_EQ(topLeft.x, x);
    EXPECT_FLOAT_EQ(topLeft.y, y);

    EXPECT_FLOAT_EQ(topRight.x, x + width);
    EXPECT_FLOAT_EQ(topRight.y, y);

    EXPECT_FLOAT_EQ(bottomRight.x, x + width);
    EXPECT_FLOAT_EQ(bottomRight.y, y + height);

    EXPECT_FLOAT_EQ(bottomLeft.x, x);
    EXPECT_FLOAT_EQ(bottomLeft.y, y + height);
}

void expectQuadUv(const std::vector<TextVertex>& vertices, size_t base, const GlyphUVRect& uv) {
    ASSERT_GE(vertices.size(), base + 4);

    EXPECT_FLOAT_EQ(vertices[base + 0].u, uv.u0);
    EXPECT_FLOAT_EQ(vertices[base + 0].v, uv.v0);

    EXPECT_FLOAT_EQ(vertices[base + 1].u, uv.u1);
    EXPECT_FLOAT_EQ(vertices[base + 1].v, uv.v0);

    EXPECT_FLOAT_EQ(vertices[base + 2].u, uv.u1);
    EXPECT_FLOAT_EQ(vertices[base + 2].v, uv.v1);

    EXPECT_FLOAT_EQ(vertices[base + 3].u, uv.u0);
    EXPECT_FLOAT_EQ(vertices[base + 3].v, uv.v1);
}

void expectQuadColor(const std::vector<TextVertex>& vertices, size_t base, float r, float g, float b, float a) {
    ASSERT_GE(vertices.size(), base + 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(vertices[base + i].r, r);
        EXPECT_FLOAT_EQ(vertices[base + i].g, g);
        EXPECT_FLOAT_EQ(vertices[base + i].b, b);
        EXPECT_FLOAT_EQ(vertices[base + i].a, a);
    }
}

}  // namespace

TEST(SimpleTextRendererTest, emptyStringProducesEmptyMesh) {
    SimpleTextRenderer renderer;
    renderer.setText("", 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    EXPECT_TRUE(renderer.getVertices().empty());
    EXPECT_TRUE(renderer.getIndices().empty());
}

TEST(SimpleTextRendererTest, singleCharacterProducesOneQuad) {
    SimpleTextRenderer renderer;
    renderer.setText("A", 10.0f, 20.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    ASSERT_EQ(renderer.getVertices().size(), 4u);
    ASSERT_EQ(renderer.getIndices().size(), 6u);

    const std::vector<uint32_t> expectedIndices = {0, 1, 2, 2, 3, 0};
    EXPECT_EQ(renderer.getIndices(), expectedIndices);
}

TEST(SimpleTextRendererTest, vertexCountsScaleWithStringLength) {
    SimpleTextRenderer renderer;
    const std::string text = "Hello!";
    renderer.setText(text, 0.0f, 0.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    EXPECT_EQ(renderer.getVertices().size(), text.size() * 4);
    EXPECT_EQ(renderer.getIndices().size(), text.size() * 6);

    // Every glyph's quad must form two triangles relative to its own 4-vertex base.
    for (size_t i = 0; i < text.size(); ++i) {
        const uint32_t base = static_cast<uint32_t>(i * 4);
        const size_t indexBase = i * 6;
        EXPECT_EQ(renderer.getIndices()[indexBase + 0], base + 0);
        EXPECT_EQ(renderer.getIndices()[indexBase + 1], base + 1);
        EXPECT_EQ(renderer.getIndices()[indexBase + 2], base + 2);
        EXPECT_EQ(renderer.getIndices()[indexBase + 3], base + 2);
        EXPECT_EQ(renderer.getIndices()[indexBase + 4], base + 3);
        EXPECT_EQ(renderer.getIndices()[indexBase + 5], base + 0);
    }
}

TEST(SimpleTextRendererTest, quadPositionsAdvanceByGlyphCellWidth) {
    SimpleTextRenderer renderer;
    const float scale = 3.0f;
    const float startX = 5.0f;
    const float startY = 7.0f;
    renderer.setText("AB", startX, startY, scale, 1.0f, 1.0f, 1.0f, 1.0f);

    const float advance = static_cast<float>(GLYPH_CELL_WIDTH) * scale;
    const float width = static_cast<float>(GLYPH_PIXEL_WIDTH) * scale;
    const float height = static_cast<float>(GLYPH_PIXEL_HEIGHT) * scale;

    expectQuadGeometry(renderer.getVertices(), 0, startX, startY, width, height);
    expectQuadGeometry(renderer.getVertices(), 4, startX + advance, startY, width, height);
}

TEST(SimpleTextRendererTest, quadSizeMatchesGlyphPixelDimensions) {
    SimpleTextRenderer renderer;
    const float scale = 4.0f;
    renderer.setText("X", 0.0f, 0.0f, scale, 1.0f, 1.0f, 1.0f, 1.0f);

    const float width = static_cast<float>(GLYPH_PIXEL_WIDTH) * scale;
    const float height = static_cast<float>(GLYPH_PIXEL_HEIGHT) * scale;
    expectQuadGeometry(renderer.getVertices(), 0, 0.0f, 0.0f, width, height);
}

TEST(SimpleTextRendererTest, uvCoordinatesMatchGlyphAtlasLookup) {
    SimpleTextRenderer renderer;
    const std::string text = "Ab1";
    renderer.setText(text, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    for (size_t i = 0; i < text.size(); ++i) {
        const GlyphUVRect uv = glyphUVRect(static_cast<unsigned char>(text[i]));
        expectQuadUv(renderer.getVertices(), i * 4, uv);
    }
}

TEST(SimpleTextRendererTest, colorIsPropagatedToAllVertices) {
    SimpleTextRenderer renderer;
    renderer.setText("Hi", 0.0f, 0.0f, 1.0f, 0.25f, 0.5f, 0.75f, 0.9f);

    expectQuadColor(renderer.getVertices(), 0, 0.25f, 0.5f, 0.75f, 0.9f);
    expectQuadColor(renderer.getVertices(), 4, 0.25f, 0.5f, 0.75f, 0.9f);
}

TEST(SimpleTextRendererTest, outOfRangeCharacterUsesMissingGlyphUv) {
    SimpleTextRenderer renderer;
    const std::string text = "\x01";  // control character, outside [32, 126]
    renderer.setText(text, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    const GlyphUVRect missingUv = glyphUVRect(0x01);
    expectQuadUv(renderer.getVertices(), 0, missingUv);

    // Sanity: the missing-glyph UV differs from a regular in-range glyph's.
    const GlyphUVRect letterUv = glyphUVRect('A');
    EXPECT_FALSE(missingUv.u0 == letterUv.u0 && missingUv.v0 == letterUv.v0);
}
