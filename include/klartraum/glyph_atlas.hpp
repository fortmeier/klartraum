#pragma once

#include <cstdint>
#include <vector>

namespace klartraum {

// Range of supported ASCII characters (printable, codes 32..126).
constexpr int GLYPH_FIRST_CHAR = 32;
constexpr int GLYPH_LAST_CHAR = 126;
constexpr int GLYPH_COUNT = GLYPH_LAST_CHAR - GLYPH_FIRST_CHAR + 1;  // 95

// Each glyph is rasterized into a fixed-size cell within the atlas. The
// glyph itself occupies GLYPH_PIXEL_WIDTH x GLYPH_PIXEL_HEIGHT pixels in the
// top-left corner of its cell; the remaining pixels are transparent padding
// used as inter-glyph spacing when rendering text.
constexpr int GLYPH_PIXEL_WIDTH = 5;
constexpr int GLYPH_PIXEL_HEIGHT = 7;
constexpr int GLYPH_CELL_WIDTH = 6;
constexpr int GLYPH_CELL_HEIGHT = 8;

// The atlas is laid out as a grid of cells, row by row. It holds one cell
// per supported character plus one extra "missing glyph" cell, so the grid
// is sized to exactly fit GLYPH_COUNT + 1 cells.
constexpr int GLYPH_ATLAS_COLS = 16;
constexpr int GLYPH_ATLAS_ROWS = (GLYPH_COUNT + 1 + GLYPH_ATLAS_COLS - 1) / GLYPH_ATLAS_COLS;
constexpr int GLYPH_ATLAS_WIDTH = GLYPH_ATLAS_COLS * GLYPH_CELL_WIDTH;
constexpr int GLYPH_ATLAS_HEIGHT = GLYPH_ATLAS_ROWS * GLYPH_CELL_HEIGHT;

// Cell index used for character codes outside [GLYPH_FIRST_CHAR, GLYPH_LAST_CHAR].
constexpr int GLYPH_MISSING_INDEX = GLYPH_COUNT;

// Atlas-space UV rectangle of a single glyph cell, in normalized [0, 1]
// texture coordinates. (u0, v0) is the top-left corner, (u1, v1) the
// bottom-right corner of the glyph's pixels (padding excluded).
struct GlyphUVRect {
    float u0;
    float v0;
    float u1;
    float v1;
};

// Maps a character code to its cell index within the atlas grid.
// Codes outside [GLYPH_FIRST_CHAR, GLYPH_LAST_CHAR] map to GLYPH_MISSING_INDEX.
int glyphIndexForChar(int charCode);

// Returns the atlas UV rectangle for a character code. Codes outside
// [GLYPH_FIRST_CHAR, GLYPH_LAST_CHAR] return the rectangle of the
// "missing glyph" cell.
GlyphUVRect glyphUVRect(int charCode);

// Generates the atlas bitmap as a single-channel (alpha) image, row-major,
// top row first, GLYPH_ATLAS_WIDTH * GLYPH_ATLAS_HEIGHT bytes in size.
// Each byte is 0 for background and 255 for glyph pixels.
std::vector<uint8_t> generateGlyphAtlasBitmap();

}  // namespace klartraum
