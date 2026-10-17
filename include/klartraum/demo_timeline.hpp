// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_DEMO_TIMELINE_HPP
#define KLARTRAUM_DEMO_TIMELINE_HPP

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/text_gaussians.hpp"

namespace klartraum {

// When a chapter's caption is shown: from `delay` seconds after the chapter
// starts until `endGap` seconds before it ends, fading in and out over `fade`
// seconds within that window.
struct CaptionWindow {
    int chapter = 0;
    float delay = 0.5f;
    float endGap = 0.5f;
    float fade = 0.5f;
};

// Timing of a video made of chapters in one continuous shot around a scene:
// the first chapter holds a title for titleSeconds and morphs it into the
// scene over morphSeconds; the camera starts to orbit with the morph, speeds
// up evenly until the morph ends and then keeps one turn per secondsPerTurn
// until slowChapter (if set), where it eases over slowTransitionSeconds to one
// turn per slowSecondsPerTurn for the rest of the video. One chapter may push
// the camera in from startDistance to endDistance.
struct DemoTimeline {
    float fps = 30.0f;
    std::vector<float> chapterSeconds;
    float titleSeconds = 2.0f;
    float morphSeconds = 1.5f;
    float secondsPerTurn = 15.0f;
    int slowChapter = -1; // -1: the orbit keeps its speed
    float slowSecondsPerTurn = 60.0f;
    float slowTransitionSeconds = 1.5f;
    float startDistance = 0.65f;
    float endDistance = 0.65f;
    int pushInChapter = -1; // -1: no push-in

    uint32_t frameCount() const;
    uint32_t chapterStart(int chapter) const;
    // The chapter frame `frame` belongs to.
    int chapterAt(uint32_t frame) const;
    // Seconds since the start of the frame's chapter.
    float timeInChapter(uint32_t frame) const;

    float morphProgress(uint32_t frame) const;
    // Camera angle around the scene, radians since the morph started.
    double orbitAngle(uint32_t frame) const;
    float cameraDistance(uint32_t frame) const;
    // Opacity in [0, 1] of the caption shown in `window` at `frame`.
    float captionOpacity(const CaptionWindow& window, uint32_t frame) const;
};

// Blends `colour` into the tightly packed BGRA image `bgra` (width x height)
// where `mask` covers it, with the mask's top-left pixel at (x, y): each
// pixel moves towards the colour by coverage x opacity. Parts of the mask
// outside the image are skipped; the alpha channel is left as it is.
void blendCaption(std::vector<uint8_t>& bgra, uint32_t width, uint32_t height, const CoverageMask& mask, int x, int y,
                  const glm::vec3& colour, float opacity);

} // namespace klartraum

#endif // KLARTRAUM_DEMO_TIMELINE_HPP
