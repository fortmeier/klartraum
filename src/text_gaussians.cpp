// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/text_gaussians.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_MULTIPLE_MASTERS_H

#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

namespace {

constexpr float kSh0 = 0.282095f; // SH degree-0 basis value used by the splatting shaders

// Em size, in pixels, at which lines are measured. Large enough that the
// advances FreeType reports carry no visible rounding.
constexpr float kMeasureEmPixels = 1024.0f;

FT_Library library() {
    // Created on first use and kept for the lifetime of the program; FreeType
    // releases it at exit.
    static FT_Library lib = [] {
        FT_Library l = nullptr;
        if (FT_Init_FreeType(&l) != 0) {
            throw std::runtime_error("FreeType could not be initialised");
        }
        return l;
    }();
    return lib;
}

FT_ULong axisTag(const std::string& tag) {
    if (tag.size() != 4) {
        return 0;
    }
    return FT_MAKE_TAG(tag[0], tag[1], tag[2], tag[3]);
}

FT_Face faceOf(const Font& font) { return static_cast<FT_Face>(font.getFace()); }

void setEmPixels(FT_Face face, float emPixels) {
    if (FT_Set_Char_Size(face, 0, FT_F26Dot6(std::lround(emPixels * 64.0f)), 72, 72) != 0) {
        throw std::runtime_error("FreeType could not set the font size");
    }
}

void applyWeight(const TextLine& line) {
    if (line.weight > 0.0f) {
        line.font->setAxis("wght", line.weight);
    }
}

// Pen advance after each character of `line`, in pixels at the face's current
// size, including kerning and letter spacing. No spacing follows the last
// character.
std::vector<float> advances(const TextLine& line, FT_Face face, float emPixels) {
    const std::vector<char32_t> codePoints = decodeUtf8(line.text);
    std::vector<float> result;
    result.reserve(codePoints.size());
    FT_UInt previous = 0;
    for (size_t i = 0; i < codePoints.size(); ++i) {
        const FT_UInt glyph = FT_Get_Char_Index(face, FT_ULong(codePoints[i]));
        if (FT_Load_Glyph(face, glyph, FT_LOAD_NO_HINTING) != 0) {
            throw std::runtime_error("FreeType could not load a glyph");
        }
        // linearHoriAdvance is 16.16 fixed point and not rounded to pixels.
        float advance = float(face->glyph->linearHoriAdvance) / 65536.0f;
        if (i + 1 < codePoints.size()) {
            advance += line.letterSpacing * emPixels;
        }
        if (previous != 0 && FT_HAS_KERNING(face)) {
            FT_Vector kerning;
            FT_Get_Kerning(face, previous, glyph, FT_KERNING_UNFITTED, &kerning);
            // The kerning shifts this glyph relative to the previous one.
            if (!result.empty()) {
                result.back() += float(kerning.x) / 64.0f;
            }
        }
        result.push_back(advance);
        previous = glyph;
    }
    return result;
}

float sumOf(const std::vector<float>& values) {
    float sum = 0.0f;
    for (float v : values) {
        sum += v;
    }
    return sum;
}

// A rendered glyph placed in layout pixel space (x right, y down, the first
// baseline at y = 0).
struct PlacedGlyph {
    int left = 0;
    int top = 0;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

// Deterministic, platform-independent random numbers in [-1, 1] for a grid
// cell (splitmix64 finaliser).
float hashUnit(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x ^= x >> 31;
    return float(x >> 40) / float(1ull << 24) * 2.0f - 1.0f;
}

// Whether the mask is covered at (x, y) and at the points `inset` away from it
// in eight directions, i.e. whether (x, y) lies at least `inset` inside the
// outline.
bool insideInset(const CoverageMask& mask, float x, float y, float inset, float threshold) {
    if (mask.sample(x, y) < threshold) {
        return false;
    }
    if (inset <= 0.0f) {
        return true;
    }
    const float d = inset * 0.70710678f;
    const float offsets[8][2] = {{inset, 0}, {-inset, 0}, {0, inset}, {0, -inset}, {d, d}, {d, -d}, {-d, d}, {-d, -d}};
    for (const auto& o : offsets) {
        if (mask.sample(x + o[0], y + o[1]) < threshold) {
            return false;
        }
    }
    return true;
}

} // namespace

Font::Font(const std::string& path)
    : data(readFile(path)) {
    FT_Face f = nullptr;
    if (FT_New_Memory_Face(library(), reinterpret_cast<const FT_Byte*>(data.data()), FT_Long(data.size()), 0, &f) !=
        0) {
        throw std::runtime_error("not a font FreeType can read: " + path);
    }
    face = f;

    FT_MM_Var* mm = nullptr;
    if (FT_HAS_MULTIPLE_MASTERS(f) && FT_Get_MM_Var(f, &mm) == 0) {
        for (FT_UInt i = 0; i < mm->num_axis; ++i) {
            axisValues.push_back(mm->axis[i].def);
        }
        FT_Done_MM_Var(library(), mm);
    }
}

Font::~Font() {
    if (face) {
        FT_Done_Face(static_cast<FT_Face>(face));
    }
}

std::string Font::getFamilyName() const {
    const FT_Face f = static_cast<FT_Face>(face);
    return f->family_name ? f->family_name : "";
}

bool Font::hasAxis(const std::string& tag) const {
    const FT_Face f = static_cast<FT_Face>(face);
    FT_MM_Var* mm = nullptr;
    if (axisValues.empty() || FT_Get_MM_Var(f, &mm) != 0) {
        return false;
    }
    bool found = false;
    for (FT_UInt i = 0; i < mm->num_axis && !found; ++i) {
        found = mm->axis[i].tag == axisTag(tag);
    }
    FT_Done_MM_Var(library(), mm);
    return found;
}

bool Font::setAxis(const std::string& tag, float value) {
    const FT_Face f = static_cast<FT_Face>(face);
    FT_MM_Var* mm = nullptr;
    if (axisValues.empty() || FT_Get_MM_Var(f, &mm) != 0) {
        return false;
    }
    bool found = false;
    for (FT_UInt i = 0; i < mm->num_axis; ++i) {
        if (mm->axis[i].tag == axisTag(tag)) {
            const FT_Fixed fixed = FT_Fixed(std::lround(double(value) * 65536.0));
            axisValues[i] = std::clamp(fixed, mm->axis[i].minimum, mm->axis[i].maximum);
            found = true;
        }
    }
    FT_Done_MM_Var(library(), mm);
    if (found) {
        FT_Set_Var_Design_Coordinates(f, FT_UInt(axisValues.size()), axisValues.data());
    }
    return found;
}

float CoverageMask::sample(float worldX, float worldY) const {
    const float fx = (worldX - originX) * pixelsPerUnit - 0.5f;
    const float fy = (originY - worldY) * pixelsPerUnit - 0.5f;
    const int x0 = int(std::floor(fx));
    const int y0 = int(std::floor(fy));
    const float tx = fx - float(x0);
    const float ty = fy - float(y0);
    auto value = [&](int x, int y) {
        return (x < 0 || y < 0 || x >= width || y >= height) ? 0.0f : float(at(x, y)) / 255.0f;
    };
    const float top = value(x0, y0) * (1.0f - tx) + value(x0 + 1, y0) * tx;
    const float bottom = value(x0, y0 + 1) * (1.0f - tx) + value(x0 + 1, y0 + 1) * tx;
    return top * (1.0f - ty) + bottom * ty;
}

glm::vec2 CoverageMask::pixelCentre(int x, int y) const {
    return {originX + (float(x) + 0.5f) / pixelsPerUnit, originY - (float(y) + 0.5f) / pixelsPerUnit};
}

std::vector<char32_t> decodeUtf8(const std::string& text) {
    std::vector<char32_t> result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        const int length = lead < 0x80           ? 1
                           : (lead >> 5) == 0x6  ? 2
                           : (lead >> 4) == 0xE  ? 3
                           : (lead >> 3) == 0x1E ? 4
                                                 : 0;
        if (length == 0 || i + length > text.size()) {
            result.push_back(0xFFFD);
            ++i;
            continue;
        }
        char32_t codePoint = length == 1 ? lead : lead & (0x7F >> length);
        bool valid = true;
        for (int k = 1; k < length; ++k) {
            const unsigned char next = static_cast<unsigned char>(text[i + k]);
            valid = valid && (next >> 6) == 0x2;
            codePoint = (codePoint << 6) | (next & 0x3F);
        }
        if (!valid) {
            result.push_back(0xFFFD);
            ++i;
            continue;
        }
        result.push_back(codePoint);
        i += length;
    }
    return result;
}

float measureLine(const TextLine& line) {
    if (!line.font || line.text.empty()) {
        return 0.0f;
    }
    applyWeight(line);
    const FT_Face face = faceOf(*line.font);
    setEmPixels(face, kMeasureEmPixels);
    return sumOf(advances(line, face, kMeasureEmPixels)) / kMeasureEmPixels * line.size;
}

CoverageMask rasterizeTitle(const TitleLayout& layout) {
    const float ppu = layout.pixelsPerUnit;
    std::vector<PlacedGlyph> glyphs;
    float baseline = 0.0f;
    for (size_t l = 0; l < layout.lines.size(); ++l) {
        const TextLine& line = layout.lines[l];
        if (!line.font) {
            throw std::invalid_argument("rasterizeTitle: a line has no font");
        }
        if (l > 0) {
            baseline += layout.lineSpacing * std::max(line.size, layout.lines[l - 1].size) * ppu;
        }
        applyWeight(line);
        const FT_Face face = faceOf(*line.font);
        const float emPixels = line.size * ppu;
        setEmPixels(face, emPixels);
        const std::vector<float> steps = advances(line, face, emPixels);

        float pen = layout.align == TextAlign::Left ? 0.0f : -0.5f * sumOf(steps);
        const int baselineRow = int(std::lround(baseline));
        const std::vector<char32_t> codePoints = decodeUtf8(line.text);
        for (size_t i = 0; i < codePoints.size(); ++i) {
            // The fractional part of the pen position goes into the outline,
            // so glyphs are placed with subpixel accuracy.
            const float whole = std::floor(pen);
            FT_Vector delta{FT_Pos(std::lround((pen - whole) * 64.0f)), 0};
            FT_Set_Transform(face, nullptr, &delta);
            const FT_UInt glyph = FT_Get_Char_Index(face, FT_ULong(codePoints[i]));
            const FT_Error error = FT_Load_Glyph(face, glyph, FT_LOAD_NO_HINTING | FT_LOAD_RENDER);
            FT_Set_Transform(face, nullptr, nullptr);
            if (error != 0) {
                throw std::runtime_error("FreeType could not render a glyph");
            }
            const FT_Bitmap& bitmap = face->glyph->bitmap;
            if (bitmap.width > 0 && bitmap.rows > 0) {
                if (bitmap.pixel_mode != FT_PIXEL_MODE_GRAY) {
                    throw std::runtime_error("FreeType rendered a glyph in an unexpected pixel mode");
                }
                PlacedGlyph placed;
                placed.left = int(whole) + face->glyph->bitmap_left;
                placed.top = baselineRow - face->glyph->bitmap_top;
                placed.width = int(bitmap.width);
                placed.height = int(bitmap.rows);
                placed.pixels.resize(size_t(placed.width) * placed.height);
                for (int y = 0; y < placed.height; ++y) {
                    const unsigned char* row = bitmap.buffer + std::ptrdiff_t(y) * bitmap.pitch;
                    std::copy(row, row + placed.width, placed.pixels.begin() + std::ptrdiff_t(y) * placed.width);
                }
                glyphs.push_back(std::move(placed));
            }
            pen += steps[i];
        }
    }

    CoverageMask mask;
    mask.pixelsPerUnit = ppu;
    if (glyphs.empty()) {
        mask.width = 1;
        mask.height = 1;
        mask.coverage.assign(1, 0);
        mask.originX = -0.5f / ppu;
        mask.originY = 0.5f / ppu;
        return mask;
    }

    int minX = glyphs[0].left, minY = glyphs[0].top;
    int maxX = minX + glyphs[0].width, maxY = minY + glyphs[0].height;
    for (const PlacedGlyph& g : glyphs) {
        minX = std::min(minX, g.left);
        minY = std::min(minY, g.top);
        maxX = std::max(maxX, g.left + g.width);
        maxY = std::max(maxY, g.top + g.height);
    }
    // A border of empty pixels keeps the bilinear lookup and the splats'
    // jitter away from the edge.
    const int border = 4;
    mask.width = maxX - minX + 2 * border;
    mask.height = maxY - minY + 2 * border;
    mask.coverage.assign(size_t(mask.width) * mask.height, 0);
    for (const PlacedGlyph& g : glyphs) {
        for (int y = 0; y < g.height; ++y) {
            for (int x = 0; x < g.width; ++x) {
                uint8_t& dst =
                    mask.coverage[size_t(g.top - minY + border + y) * mask.width + (g.left - minX + border + x)];
                dst = std::max(dst, g.pixels[size_t(y) * g.width + x]);
            }
        }
    }
    // The ink's bounding box, and so the mask, is centred on the origin.
    mask.originX = -0.5f * float(mask.width) / ppu;
    mask.originY = 0.5f * float(mask.height) / ppu;
    return mask;
}

std::vector<Gaussian3D> sampleTextGaussians(const CoverageMask& mask, const TextSplatParams& params) {
    if (!(params.spacing > 0.0f)) {
        throw std::invalid_argument("sampleTextGaussians: spacing must be positive");
    }
    const float extentX = float(mask.width) / mask.pixelsPerUnit;
    const float extentY = float(mask.height) / mask.pixelsPerUnit;
    const int columns = int(std::ceil(extentX / params.spacing));
    const int rows = int(std::ceil(extentY / params.spacing));
    const float sigma = params.splatScale * params.spacing;
    const std::array<float, 3> color = {colorToSh0(params.color.r), colorToSh0(params.color.g),
                                        colorToSh0(params.color.b)};

    std::vector<Gaussian3D> result;
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < columns; ++i) {
            const uint64_t cell = (uint64_t(params.seed) << 42) ^ (uint64_t(uint32_t(j)) << 21) ^ uint64_t(uint32_t(i));
            const float jx = hashUnit(cell * 3 + 0);
            const float jy = hashUnit(cell * 3 + 1);
            const float jz = hashUnit(cell * 3 + 2);
            const float x = mask.originX + (float(i) + 0.5f + 0.5f * params.jitter * jx) * params.spacing;
            const float y = mask.originY - (float(j) + 0.5f + 0.5f * params.jitter * jy) * params.spacing;
            if (!insideInset(mask, x, y, params.inset * params.spacing, params.threshold)) {
                continue;
            }
            Gaussian3D g{};
            g.position = {x, y, params.depthJitter * jz};
            g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
            g.scale = {sigma, sigma, sigma};
            g.color = color;
            g.alpha = params.opacity;
            result.push_back(g);
        }
    }
    return result;
}

std::vector<Gaussian3D> sampleTextGaussians(const CoverageMask& mask, TextSplatParams params, uint32_t count) {
    if (count == 0) {
        return {};
    }
    size_t coveredPixels = 0;
    for (uint8_t c : mask.coverage) {
        coveredPixels += float(c) / 255.0f >= params.threshold;
    }
    if (coveredPixels == 0) {
        throw std::invalid_argument("sampleTextGaussians: the mask covers nothing");
    }
    // A grid of spacing s puts about area / s^2 samples into the covered
    // area; the inset and the outline make it a few less, so the spacing
    // shrinks until there are enough.
    const float area = float(coveredPixels) / (mask.pixelsPerUnit * mask.pixelsPerUnit);
    params.spacing = std::sqrt(area / float(count));
    std::vector<Gaussian3D> samples = sampleTextGaussians(mask, params);
    for (int attempt = 0; samples.size() < count; ++attempt) {
        if (attempt == 32) {
            throw std::runtime_error("sampleTextGaussians: could not place enough splats");
        }
        params.spacing *= 0.99f * std::sqrt(float(std::max<size_t>(samples.size(), 1)) / float(count));
        samples = sampleTextGaussians(mask, params);
    }

    // Keep the `count` samples with the smallest random keys, in grid order.
    std::vector<std::pair<float, uint32_t>> keys(samples.size());
    for (uint32_t i = 0; i < samples.size(); ++i) {
        keys[i] = {hashUnit((uint64_t(params.seed) << 40) ^ (0xa5a5ull << 32) ^ i), i};
    }
    std::nth_element(keys.begin(), keys.begin() + count, keys.end());
    std::vector<uint32_t> kept(count);
    for (uint32_t i = 0; i < count; ++i) {
        kept[i] = keys[i].second;
    }
    std::sort(kept.begin(), kept.end());
    std::vector<Gaussian3D> result(count);
    for (uint32_t i = 0; i < count; ++i) {
        result[i] = samples[kept[i]];
    }
    return result;
}

float colorToSh0(float value) { return (value - 0.5f) / kSh0; }

} // namespace klartraum
