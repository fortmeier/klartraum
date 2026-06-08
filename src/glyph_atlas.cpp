#include "klartraum/glyph_atlas.hpp"

namespace klartraum {

namespace {

// Procedural 5x7 bitmap font, one entry per character code in
// [GLYPH_FIRST_CHAR, GLYPH_LAST_CHAR], in order. Each glyph is described by
// GLYPH_PIXEL_HEIGHT rows of GLYPH_PIXEL_WIDTH characters: 'X' is a filled
// (foreground) pixel, '.' is empty (background/transparent).
const char* const kGlyphPatterns[GLYPH_COUNT][GLYPH_PIXEL_HEIGHT] = {
    // 32 ' '
    {".....", ".....", ".....", ".....", ".....", ".....", "....."},
    // 33 '!'
    {"..X..", "..X..", "..X..", "..X..", "..X..", ".....", "..X.."},
    // 34 '"'
    {".X.X.", ".X.X.", ".....", ".....", ".....", ".....", "....."},
    // 35 '#'
    {".X.X.", ".X.X.", "XXXXX", ".X.X.", "XXXXX", ".X.X.", ".X.X."},
    // 36 '$'
    {"..X..", ".XXXX", "X.X..", ".XXX.", "..X.X", "XXXX.", "..X.."},
    // 37 '%'
    {"XX...", "XX..X", "...X.", "..X..", ".X...", "X..XX", "...XX"},
    // 38 '&'
    {".XX..", "X..X.", "X.X..", ".X...", "X.X.X", "X..X.", ".XX.X"},
    // 39 '\''
    {"..X..", "..X..", ".....", ".....", ".....", ".....", "....."},
    // 40 '('
    {"...X.", "..X..", ".X...", ".X...", ".X...", "..X..", "...X."},
    // 41 ')'
    {".X...", "..X..", "...X.", "...X.", "...X.", "..X..", ".X..."},
    // 42 '*'
    {".....", "X.X.X", ".XXX.", "XXXXX", ".XXX.", "X.X.X", "....."},
    // 43 '+'
    {".....", "..X..", "..X..", "XXXXX", "..X..", "..X..", "....."},
    // 44 ','
    {".....", ".....", ".....", ".....", "..XX.", "..X..", ".X..."},
    // 45 '-'
    {".....", ".....", ".....", "XXXXX", ".....", ".....", "....."},
    // 46 '.'
    {".....", ".....", ".....", ".....", ".....", ".XX..", ".XX.."},
    // 47 '/'
    {"....X", "...X.", "..X..", "..X..", "..X..", ".X...", "X...."},
    // 48 '0'
    {".XXX.", "X...X", "X..XX", "X.X.X", "XX..X", "X...X", ".XXX."},
    // 49 '1'
    {"..X..", ".XX..", "..X..", "..X..", "..X..", "..X..", ".XXX."},
    // 50 '2'
    {".XXX.", "X...X", "....X", "...X.", "..X..", ".X...", "XXXXX"},
    // 51 '3'
    {".XXX.", "X...X", "....X", "..XX.", "....X", "X...X", ".XXX."},
    // 52 '4'
    {"...X.", "..XX.", ".X.X.", "X..X.", "XXXXX", "...X.", "...X."},
    // 53 '5'
    {"XXXXX", "X....", "XXXX.", "....X", "....X", "X...X", ".XXX."},
    // 54 '6'
    {"..XX.", ".X...", "X....", "XXXX.", "X...X", "X...X", ".XXX."},
    // 55 '7'
    {"XXXXX", "....X", "...X.", "..X..", ".X...", ".X...", ".X..."},
    // 56 '8'
    {".XXX.", "X...X", "X...X", ".XXX.", "X...X", "X...X", ".XXX."},
    // 57 '9'
    {".XXX.", "X...X", "X...X", ".XXXX", "....X", "...X.", ".XX.."},
    // 58 ':'
    {".....", "..X..", ".....", ".....", ".....", "..X..", "....."},
    // 59 ';'
    {".....", "..X..", ".....", ".....", "..X..", "..X..", ".X..."},
    // 60 '<'
    {"...X.", "..X..", ".X...", "X....", ".X...", "..X..", "...X."},
    // 61 '='
    {".....", ".....", "XXXXX", ".....", "XXXXX", ".....", "....."},
    // 62 '>'
    {".X...", "..X..", "...X.", "....X", "...X.", "..X..", ".X..."},
    // 63 '?'
    {".XXX.", "X...X", "....X", "..XX.", "..X..", ".....", "..X.."},
    // 64 '@'
    {".XXX.", "X...X", "X.XXX", "X.X.X", "X.XX.", "X....", ".XXX."},
    // 65 'A'
    {"..X..", ".X.X.", "X...X", "X...X", "XXXXX", "X...X", "X...X"},
    // 66 'B'
    {"XXXX.", "X...X", "X...X", "XXXX.", "X...X", "X...X", "XXXX."},
    // 67 'C'
    {".XXX.", "X...X", "X....", "X....", "X....", "X...X", ".XXX."},
    // 68 'D'
    {"XXXX.", "X...X", "X...X", "X...X", "X...X", "X...X", "XXXX."},
    // 69 'E'
    {"XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "XXXXX"},
    // 70 'F'
    {"XXXXX", "X....", "X....", "XXXX.", "X....", "X....", "X...."},
    // 71 'G'
    {".XXX.", "X...X", "X....", "X.XXX", "X...X", "X...X", ".XXXX"},
    // 72 'H'
    {"X...X", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"},
    // 73 'I'
    {".XXX.", "..X..", "..X..", "..X..", "..X..", "..X..", ".XXX."},
    // 74 'J'
    {"..XXX", "...X.", "...X.", "...X.", "X..X.", "X..X.", ".XX.."},
    // 75 'K'
    {"X...X", "X..X.", "X.X..", "XX...", "X.X..", "X..X.", "X...X"},
    // 76 'L'
    {"X....", "X....", "X....", "X....", "X....", "X....", "XXXXX"},
    // 77 'M'
    {"X...X", "XX.XX", "X.X.X", "X.X.X", "X...X", "X...X", "X...X"},
    // 78 'N'
    {"X...X", "XX..X", "X.X.X", "X.X.X", "X..XX", "X...X", "X...X"},
    // 79 'O'
    {".XXX.", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."},
    // 80 'P'
    {"XXXX.", "X...X", "X...X", "XXXX.", "X....", "X....", "X...."},
    // 81 'Q'
    {".XXX.", "X...X", "X...X", "X...X", "X.X.X", "X..X.", ".XX.X"},
    // 82 'R'
    {"XXXX.", "X...X", "X...X", "XXXX.", "X.X..", "X..X.", "X...X"},
    // 83 'S'
    {".XXXX", "X....", "X....", ".XXX.", "....X", "....X", "XXXX."},
    // 84 'T'
    {"XXXXX", "..X..", "..X..", "..X..", "..X..", "..X..", "..X.."},
    // 85 'U'
    {"X...X", "X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."},
    // 86 'V'
    {"X...X", "X...X", "X...X", "X...X", "X...X", ".X.X.", "..X.."},
    // 87 'W'
    {"X...X", "X...X", "X...X", "X.X.X", "X.X.X", "XX.XX", "X...X"},
    // 88 'X'
    {"X...X", "X...X", ".X.X.", "..X..", ".X.X.", "X...X", "X...X"},
    // 89 'Y'
    {"X...X", "X...X", ".X.X.", "..X..", "..X..", "..X..", "..X.."},
    // 90 'Z'
    {"XXXXX", "....X", "...X.", "..X..", ".X...", "X....", "XXXXX"},
    // 91 '['
    {".XXX.", ".X...", ".X...", ".X...", ".X...", ".X...", ".XXX."},
    // 92 '\\'
    {"X....", ".X...", "..X..", "..X..", "..X..", "...X.", "....X"},
    // 93 ']'
    {".XXX.", "...X.", "...X.", "...X.", "...X.", "...X.", ".XXX."},
    // 94 '^'
    {"..X..", ".X.X.", "X...X", ".....", ".....", ".....", "....."},
    // 95 '_'
    {".....", ".....", ".....", ".....", ".....", ".....", "XXXXX"},
    // 96 '`'
    {".X...", "..X..", ".....", ".....", ".....", ".....", "....."},
    // 97 'a'
    {".....", ".....", ".XXX.", "....X", ".XXXX", "X...X", ".XXXX"},
    // 98 'b'
    {"X....", "X....", "XXXX.", "X...X", "X...X", "X...X", "XXXX."},
    // 99 'c'
    {".....", ".....", ".XXXX", "X....", "X....", "X....", ".XXXX"},
    // 100 'd'
    {"....X", "....X", ".XXXX", "X...X", "X...X", "X...X", ".XXXX"},
    // 101 'e'
    {".....", ".....", ".XXX.", "X...X", "XXXXX", "X....", ".XXX."},
    // 102 'f'
    {"..XX.", ".X...", "XXXX.", ".X...", ".X...", ".X...", ".X..."},
    // 103 'g'
    {".....", ".XXXX", "X...X", "X...X", ".XXXX", "....X", ".XXX."},
    // 104 'h'
    {"X....", "X....", "XXXX.", "X...X", "X...X", "X...X", "X...X"},
    // 105 'i'
    {"..X..", ".....", ".XX..", "..X..", "..X..", "..X..", ".XXX."},
    // 106 'j'
    {"...X.", ".....", "..XX.", "...X.", "...X.", "X..X.", ".XX.."},
    // 107 'k'
    {"X....", "X....", "X..X.", "X.X..", "XX...", "X.X..", "X..X."},
    // 108 'l'
    {".XX..", "..X..", "..X..", "..X..", "..X..", "..X..", ".XXX."},
    // 109 'm'
    {".....", ".....", "XX.X.", "X.X.X", "X.X.X", "X.X.X", "X...X"},
    // 110 'n'
    {".....", ".....", "XXXX.", "X...X", "X...X", "X...X", "X...X"},
    // 111 'o'
    {".....", ".....", ".XXX.", "X...X", "X...X", "X...X", ".XXX."},
    // 112 'p'
    {".....", "XXXX.", "X...X", "X...X", "XXXX.", "X....", "X...."},
    // 113 'q'
    {".....", ".XXXX", "X...X", "X...X", ".XXXX", "....X", "....X"},
    // 114 'r'
    {".....", ".....", "X.XXX", "XX...", "X....", "X....", "X...."},
    // 115 's'
    {".....", ".....", ".XXXX", "X....", ".XXX.", "....X", "XXXX."},
    // 116 't'
    {"..X..", ".XXX.", "..X..", "..X..", "..X..", "..X..", "...XX"},
    // 117 'u'
    {".....", ".....", "X...X", "X...X", "X...X", "X...X", ".XXXX"},
    // 118 'v'
    {".....", ".....", "X...X", "X...X", "X...X", ".X.X.", "..X.."},
    // 119 'w'
    {".....", ".....", "X...X", "X.X.X", "X.X.X", "X.X.X", ".X.X."},
    // 120 'x'
    {".....", ".....", "X...X", ".X.X.", "..X..", ".X.X.", "X...X"},
    // 121 'y'
    {".....", "X...X", "X...X", "X...X", ".XXXX", "....X", ".XXX."},
    // 122 'z'
    {".....", ".....", "XXXXX", "...X.", "..X..", ".X...", "XXXXX"},
    // 123 '{'
    {"...X.", "..X..", "..X..", ".X...", "..X..", "..X..", "...X."},
    // 124 '|'
    {"..X..", "..X..", "..X..", "..X..", "..X..", "..X..", "..X.."},
    // 125 '}'
    {".X...", "..X..", "..X..", "...X.", "..X..", "..X..", ".X..."},
    // 126 '~'
    {".....", ".....", ".X...", "X.X.X", "...X.", ".....", "....."},
};

// Pattern used for the "missing glyph" cell: a hollow box.
const char* const kMissingGlyphPattern[GLYPH_PIXEL_HEIGHT] = {
    "XXXXX", "X...X", "X...X", "X...X", "X...X", "X...X", "XXXXX"};

const char* const* patternForIndex(int index) {
    if (index >= 0 && index < GLYPH_COUNT) {
        return kGlyphPatterns[index];
    }
    return kMissingGlyphPattern;
}

}  // namespace

int glyphIndexForChar(int charCode) {
    if (charCode < GLYPH_FIRST_CHAR || charCode > GLYPH_LAST_CHAR) {
        return GLYPH_MISSING_INDEX;
    }
    return charCode - GLYPH_FIRST_CHAR;
}

GlyphUVRect glyphUVRect(int charCode) {
    const int index = glyphIndexForChar(charCode);
    const int col = index % GLYPH_ATLAS_COLS;
    const int row = index / GLYPH_ATLAS_COLS;

    GlyphUVRect rect;
    rect.u0 = static_cast<float>(col * GLYPH_CELL_WIDTH) / static_cast<float>(GLYPH_ATLAS_WIDTH);
    rect.v0 = static_cast<float>(row * GLYPH_CELL_HEIGHT) / static_cast<float>(GLYPH_ATLAS_HEIGHT);
    rect.u1 = static_cast<float>(col * GLYPH_CELL_WIDTH + GLYPH_PIXEL_WIDTH) /
              static_cast<float>(GLYPH_ATLAS_WIDTH);
    rect.v1 = static_cast<float>(row * GLYPH_CELL_HEIGHT + GLYPH_PIXEL_HEIGHT) /
              static_cast<float>(GLYPH_ATLAS_HEIGHT);
    return rect;
}

std::vector<uint8_t> generateGlyphAtlasBitmap() {
    std::vector<uint8_t> bitmap(static_cast<size_t>(GLYPH_ATLAS_WIDTH) * GLYPH_ATLAS_HEIGHT, 0);

    const int numCells = GLYPH_COUNT + 1;
    for (int index = 0; index < numCells; ++index) {
        const char* const* pattern = patternForIndex(index == GLYPH_COUNT ? GLYPH_MISSING_INDEX : index);
        const int col = index % GLYPH_ATLAS_COLS;
        const int row = index / GLYPH_ATLAS_COLS;
        const int originX = col * GLYPH_CELL_WIDTH;
        const int originY = row * GLYPH_CELL_HEIGHT;

        for (int py = 0; py < GLYPH_PIXEL_HEIGHT; ++py) {
            for (int px = 0; px < GLYPH_PIXEL_WIDTH; ++px) {
                if (pattern[py][px] == 'X') {
                    const int x = originX + px;
                    const int y = originY + py;
                    bitmap[static_cast<size_t>(y) * GLYPH_ATLAS_WIDTH + x] = 255;
                }
            }
        }
    }

    return bitmap;
}

}  // namespace klartraum
