/**
 * TESTS:
 * - inRangeIndexMapping: ASCII codes [32, 126] map to sequential cell indices [0, 94]
 * - outOfRangeMapsToMissingGlyph: codes outside [32, 126] map to GLYPH_MISSING_INDEX
 * - uvRectMatchesGridCell: UV rect for a character matches its expected grid cell position and glyph size
 * - outOfRangeUvRectMatchesMissingGlyphCell: UV rect for an out-of-range code matches the missing-glyph cell
 * - atlasBitmapHasExpectedSize: generated atlas bitmap has the documented byte size
 * - atlasBitmapEncodesGlyphPixels: generated atlas bitmap encodes the 'A' glyph pattern at its expected cell
 * - atlasBitmapBackgroundIsEmpty: pixels outside any glyph's pixel region are background (zero)
 **/

#include <gtest/gtest.h>

#include "klartraum/glyph_atlas.hpp"

using namespace klartraum;

TEST(GlyphAtlasTest, inRangeIndexMapping) {
    EXPECT_EQ(glyphIndexForChar(' '), 0);
    EXPECT_EQ(glyphIndexForChar('!'), 1);
    EXPECT_EQ(glyphIndexForChar('A'), 'A' - GLYPH_FIRST_CHAR);
    EXPECT_EQ(glyphIndexForChar('~'), GLYPH_COUNT - 1);
}

TEST(GlyphAtlasTest, outOfRangeMapsToMissingGlyph) {
    EXPECT_EQ(glyphIndexForChar(0), GLYPH_MISSING_INDEX);
    EXPECT_EQ(glyphIndexForChar(31), GLYPH_MISSING_INDEX);
    EXPECT_EQ(glyphIndexForChar(127), GLYPH_MISSING_INDEX);
    EXPECT_EQ(glyphIndexForChar(-1), GLYPH_MISSING_INDEX);
}

TEST(GlyphAtlasTest, uvRectMatchesGridCell) {
    const int index = glyphIndexForChar('A');
    const int col = index % GLYPH_ATLAS_COLS;
    const int row = index / GLYPH_ATLAS_COLS;

    const GlyphUVRect rect = glyphUVRect('A');

    const float expectedU0 = static_cast<float>(col * GLYPH_CELL_WIDTH) / GLYPH_ATLAS_WIDTH;
    const float expectedV0 = static_cast<float>(row * GLYPH_CELL_HEIGHT) / GLYPH_ATLAS_HEIGHT;
    const float expectedU1 = static_cast<float>(col * GLYPH_CELL_WIDTH + GLYPH_PIXEL_WIDTH) / GLYPH_ATLAS_WIDTH;
    const float expectedV1 = static_cast<float>(row * GLYPH_CELL_HEIGHT + GLYPH_PIXEL_HEIGHT) / GLYPH_ATLAS_HEIGHT;

    EXPECT_FLOAT_EQ(rect.u0, expectedU0);
    EXPECT_FLOAT_EQ(rect.v0, expectedV0);
    EXPECT_FLOAT_EQ(rect.u1, expectedU1);
    EXPECT_FLOAT_EQ(rect.v1, expectedV1);

    // Sanity: the rectangle covers exactly the glyph's pixel area, not the full padded cell.
    EXPECT_FLOAT_EQ((rect.u1 - rect.u0) * GLYPH_ATLAS_WIDTH, static_cast<float>(GLYPH_PIXEL_WIDTH));
    EXPECT_FLOAT_EQ((rect.v1 - rect.v0) * GLYPH_ATLAS_HEIGHT, static_cast<float>(GLYPH_PIXEL_HEIGHT));
}

TEST(GlyphAtlasTest, outOfRangeUvRectMatchesMissingGlyphCell) {
    const GlyphUVRect missingRect = glyphUVRect(0);
    const int col = GLYPH_MISSING_INDEX % GLYPH_ATLAS_COLS;
    const int row = GLYPH_MISSING_INDEX / GLYPH_ATLAS_COLS;

    const float expectedU0 = static_cast<float>(col * GLYPH_CELL_WIDTH) / GLYPH_ATLAS_WIDTH;
    const float expectedV0 = static_cast<float>(row * GLYPH_CELL_HEIGHT) / GLYPH_ATLAS_HEIGHT;

    EXPECT_FLOAT_EQ(missingRect.u0, expectedU0);
    EXPECT_FLOAT_EQ(missingRect.v0, expectedV0);

    // Every out-of-range code must resolve to the same cell.
    const GlyphUVRect anotherMissingRect = glyphUVRect(200);
    EXPECT_FLOAT_EQ(missingRect.u0, anotherMissingRect.u0);
    EXPECT_FLOAT_EQ(missingRect.v0, anotherMissingRect.v0);
    EXPECT_FLOAT_EQ(missingRect.u1, anotherMissingRect.u1);
    EXPECT_FLOAT_EQ(missingRect.v1, anotherMissingRect.v1);
}

TEST(GlyphAtlasTest, atlasBitmapHasExpectedSize) {
    const std::vector<uint8_t> bitmap = generateGlyphAtlasBitmap();
    EXPECT_EQ(bitmap.size(), static_cast<size_t>(GLYPH_ATLAS_WIDTH) * GLYPH_ATLAS_HEIGHT);
}

TEST(GlyphAtlasTest, atlasBitmapEncodesGlyphPixels) {
    // The 'A' glyph (5x7) is expected to look like:
    //  ..X..
    //  .X.X.
    //  X...X
    //  X...X
    //  XXXXX
    //  X...X
    //  X...X
    static const char* const kExpectedA[GLYPH_PIXEL_HEIGHT] = {
        "..X..", ".X.X.", "X...X", "X...X", "XXXXX", "X...X", "X...X"};

    const std::vector<uint8_t> bitmap = generateGlyphAtlasBitmap();

    const int index = glyphIndexForChar('A');
    const int col = index % GLYPH_ATLAS_COLS;
    const int row = index / GLYPH_ATLAS_COLS;
    const int originX = col * GLYPH_CELL_WIDTH;
    const int originY = row * GLYPH_CELL_HEIGHT;

    for (int py = 0; py < GLYPH_PIXEL_HEIGHT; ++py) {
        for (int px = 0; px < GLYPH_PIXEL_WIDTH; ++px) {
            const size_t offset = static_cast<size_t>(originY + py) * GLYPH_ATLAS_WIDTH + (originX + px);
            const uint8_t expected = (kExpectedA[py][px] == 'X') ? 255 : 0;
            EXPECT_EQ(bitmap[offset], expected) << "mismatch at glyph-local (" << px << ", " << py << ")";
        }
    }
}

TEST(GlyphAtlasTest, atlasBitmapBackgroundIsEmpty) {
    const std::vector<uint8_t> bitmap = generateGlyphAtlasBitmap();

    // The padding column/row of the 'A' cell (outside the 5x7 glyph area, but
    // within its 6x8 cell) must remain background.
    const int index = glyphIndexForChar('A');
    const int col = index % GLYPH_ATLAS_COLS;
    const int row = index / GLYPH_ATLAS_COLS;
    const int originX = col * GLYPH_CELL_WIDTH;
    const int originY = row * GLYPH_CELL_HEIGHT;

    for (int py = 0; py < GLYPH_CELL_HEIGHT; ++py) {
        const size_t offset = static_cast<size_t>(originY + py) * GLYPH_ATLAS_WIDTH + (originX + GLYPH_PIXEL_WIDTH);
        EXPECT_EQ(bitmap[offset], 0) << "expected background in padding column at row " << py;
    }
    for (int px = 0; px < GLYPH_CELL_WIDTH; ++px) {
        const size_t offset = static_cast<size_t>(originY + GLYPH_PIXEL_HEIGHT) * GLYPH_ATLAS_WIDTH + (originX + px);
        EXPECT_EQ(bitmap[offset], 0) << "expected background in padding row at column " << px;
    }
}
