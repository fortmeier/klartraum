// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RENDERING_TEXT_GAUSSIANS_HPP
#define KLARTRAUM_RENDERING_TEXT_GAUSSIANS_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

// Built only with the CMake option KLARTRAUM_WITH_FREETYPE, which defines the
// macro of the same name for code linking klartraum_lib.

namespace klartraum {

/// A TrueType/OpenType font loaded with FreeType. Variable fonts can be set to
/// any point of their design space, e.g. a weight between their lightest and
/// boldest instance.
class Font {
public:
    /// `path` is resolved like readFile() does: relative to the working
    /// directory first, then to the asset root, so "fonts/SpaceGrotesk.ttf"
    /// finds the fonts the build downloads. Throws std::runtime_error if the
    /// file cannot be read or is not a font.
    explicit Font(const std::string& path);
    ~Font();
    Font(const Font&) = delete;
    Font& operator=(const Font&) = delete;

    std::string getFamilyName() const;

    /// Whether the font has the variation axis `tag` (e.g. "wght", "opsz").
    bool hasAxis(const std::string& tag) const;

    /// Sets the variation axis `tag` to `value`, clamped to the axis range; the
    /// other axes keep their values. Returns false if the font has no such axis.
    bool setAxis(const std::string& tag, float value);

    /// FreeType's FT_Face; kept opaque so FreeType's headers stay private.
    void* getFace() const { return face; }

private:
    std::vector<char> data; // FreeType reads the face from this memory
    void* face = nullptr;
    std::vector<long> axisValues; // current design coordinates, 16.16 fixed point
};

/// One line of the title.
struct TextLine {
    std::string text; ///< UTF-8
    std::shared_ptr<Font> font;
    float weight = 0.0f;        ///< value of the "wght" axis; 0 keeps the font's default
    float size = 1.0f;          ///< em size in world units
    float letterSpacing = 0.0f; ///< extra space between letters, in em
};

enum class TextAlign {
    Centre, ///< each line centred on the block
    Left,   ///< all lines start at the block's left edge
};

/// Layout of a text block.
struct TitleLayout {
    std::vector<TextLine> lines;
    TextAlign align = TextAlign::Centre;
    float lineSpacing = 1.2f;     ///< distance between baselines, in em of the larger of two neighbouring lines
    float pixelsPerUnit = 512.0f; ///< resolution of the coverage mask
};

/// Coverage of a rasterized text block. Row 0 is the top. The block is centred
/// on the world origin: pixel (x, y) covers the world rectangle starting at
/// (originX + x / pixelsPerUnit, originY - y / pixelsPerUnit) and extending
/// right and down.
struct CoverageMask {
    int width = 0;
    int height = 0;
    float pixelsPerUnit = 1.0f;
    float originX = 0.0f;          ///< world x of the left edge
    float originY = 0.0f;          ///< world y of the top edge
    std::vector<uint8_t> coverage; ///< width * height, 0..255

    uint8_t at(int x, int y) const { return coverage[size_t(y) * width + x]; }
    /// Bilinear coverage in [0, 1] at a world position; 0 outside the mask.
    float sample(float worldX, float worldY) const;
    /// World position of the centre of pixel (x, y).
    glm::vec2 pixelCentre(int x, int y) const;
};

/// The code points of UTF-8 text; a broken sequence gives U+FFFD.
std::vector<char32_t> decodeUtf8(const std::string& text);

/// Width of `line` in world units: the sum of its advances, kerning and letter
/// spacing (the extra space is not added after the last letter).
float measureLine(const TextLine& line);

/// Rasterizes the lines of `layout`, each centred horizontally, the block
/// centred vertically on the origin by its ink (ascender to descender of the
/// first and last line).
CoverageMask rasterizeTitle(const TitleLayout& layout);

/// How a coverage mask is turned into Gaussians.
struct TextSplatParams {
    float spacing = 0.004f;  ///< distance between neighbouring samples, world units
    float splatScale = 0.6f; ///< standard deviation of a splat, relative to `spacing`
    float jitter = 0.35f;    ///< random offset of a sample within its grid cell, relative to `spacing`
    float threshold = 0.5f;  ///< minimum coverage for a sample to become a splat
    /// Distance, relative to `spacing`, by which splats keep away from the
    /// outline. The splats' soft edges reach beyond their centres, so without
    /// an inset the rendered letters come out bolder than the font.
    float inset = 0.7f;
    float depthJitter = 0.0f; ///< random z offset, world units (0: all splats on the z = 0 plane)
    glm::vec3 color{1.0f};    ///< linear RGB in [0, 1]
    float opacity = 1.0f;
    uint32_t seed = 1;
};

/// Places round Gaussians on a jittered grid inside the covered area of `mask`,
/// on the plane z = 0 (y up). The same parameters always give the same result.
std::vector<Gaussian3D> sampleTextGaussians(const CoverageMask& mask, const TextSplatParams& params);

/// Like above, but with exactly `count` splats: the spacing is chosen from the
/// covered area so the grid gives at least `count` samples (params.spacing is
/// ignored), and the surplus is dropped at random positions.
std::vector<Gaussian3D> sampleTextGaussians(const CoverageMask& mask, TextSplatParams params, uint32_t count);

/// The SH degree-0 coefficient that the splatting shaders turn into `value`.
float colorToSh0(float value);

} // namespace klartraum

#endif // KLARTRAUM_RENDERING_TEXT_GAUSSIANS_HPP
