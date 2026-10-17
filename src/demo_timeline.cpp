// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/demo_timeline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace klartraum {

namespace {

double easeInOutCubic(double s) { return s < 0.5 ? 4.0 * s * s * s : 1.0 - std::pow(-2.0 * s + 2.0, 3.0) / 2.0; }

} // namespace

uint32_t DemoTimeline::frameCount() const { return chapterStart(int(chapterSeconds.size())); }

uint32_t DemoTimeline::chapterStart(int chapter) const {
    if (chapter < 0 || chapter > int(chapterSeconds.size())) {
        throw std::out_of_range("DemoTimeline::chapterStart: no such chapter");
    }
    double seconds = 0.0;
    for (int c = 0; c < chapter; ++c) {
        seconds += chapterSeconds[c];
    }
    return uint32_t(std::lround(seconds * fps));
}

int DemoTimeline::chapterAt(uint32_t frame) const {
    int chapter = 0;
    while (chapter + 1 < int(chapterSeconds.size()) && frame >= chapterStart(chapter + 1)) {
        ++chapter;
    }
    return chapter;
}

float DemoTimeline::timeInChapter(uint32_t frame) const {
    return float(double(frame - chapterStart(chapterAt(frame))) / fps);
}

float DemoTimeline::morphProgress(uint32_t frame) const {
    const double t = double(frame) / fps - titleSeconds;
    if (morphSeconds <= 0.0f) {
        return t >= 0.0 ? 1.0f : 0.0f;
    }
    return float(std::clamp(t / morphSeconds, 0.0, 1.0));
}

double DemoTimeline::orbitAngle(uint32_t frame) const {
    const double t = double(frame) / fps - titleSeconds;
    if (t <= 0.0) {
        return 0.0;
    }
    // Constant acceleration during the morph reaches the orbit speed when
    // the morph ends.
    const double rate = 2.0 * 3.14159265358979323846 / secondsPerTurn;
    const double morph = morphSeconds;
    if (t < morph) {
        return rate * t * t / (2.0 * morph);
    }
    double angle = rate * (0.5 * morph + (t - morph));
    if (slowChapter >= 0 && slowChapter < int(chapterSeconds.size())) {
        // From the slow chapter's start the rate eases by smoothstep to the
        // slow rate; the angle adds the integral of the change, using
        // integral of 3u^2 - 2u^3 = u^3 - u^4 / 2.
        const double slowStart = double(chapterStart(slowChapter)) / fps - titleSeconds;
        const double transition = std::max(double(slowTransitionSeconds), 1e-9);
        const double change = 2.0 * 3.14159265358979323846 / slowSecondsPerTurn - rate;
        const double u = (t - std::max(slowStart, morph)) / transition;
        if (u > 0.0) {
            const double eased = u < 1.0 ? u * u * u - 0.5 * u * u * u * u : 0.5 + (u - 1.0);
            angle += change * transition * eased;
        }
    }
    return angle;
}

float DemoTimeline::cameraDistance(uint32_t frame) const {
    if (pushInChapter < 0 || pushInChapter >= int(chapterSeconds.size())) {
        return startDistance;
    }
    const uint32_t first = chapterStart(pushInChapter);
    const uint32_t last = chapterStart(pushInChapter + 1);
    if (frame <= first) {
        return startDistance;
    }
    if (frame >= last) {
        return endDistance;
    }
    const double s = easeInOutCubic(double(frame - first) / double(last - first));
    return float(startDistance + s * (endDistance - startDistance));
}

float DemoTimeline::captionOpacity(const CaptionWindow& window, uint32_t frame) const {
    if (window.chapter < 0 || window.chapter >= int(chapterSeconds.size()) || chapterAt(frame) != window.chapter) {
        return 0.0f;
    }
    const double t = double(frame - chapterStart(window.chapter)) / fps;
    const double begin = window.delay;
    const double end = double(chapterSeconds[window.chapter]) - window.endGap;
    if (t <= begin || t >= end) {
        return 0.0f;
    }
    if (window.fade <= 0.0f) {
        return 1.0f;
    }
    return float(std::clamp(std::min(t - begin, end - t) / window.fade, 0.0, 1.0));
}

void blendCaption(std::vector<uint8_t>& bgra, uint32_t width, uint32_t height, const CoverageMask& mask, int x, int y,
                  const glm::vec3& colour, float opacity) {
    if (bgra.size() < size_t(width) * height * 4) {
        throw std::invalid_argument("blendCaption: the image is smaller than width x height");
    }
    if (opacity <= 0.0f) {
        return;
    }
    const float target[3] = {colour.b * 255.0f, colour.g * 255.0f, colour.r * 255.0f};
    for (int my = 0; my < mask.height; ++my) {
        const int py = y + my;
        if (py < 0 || py >= int(height)) {
            continue;
        }
        for (int mx = 0; mx < mask.width; ++mx) {
            const int px = x + mx;
            const uint8_t coverage = mask.at(mx, my);
            if (px < 0 || px >= int(width) || coverage == 0) {
                continue;
            }
            const float a = float(coverage) / 255.0f * std::min(opacity, 1.0f);
            uint8_t* pixel = &bgra[(size_t(py) * width + px) * 4];
            for (int c = 0; c < 3; ++c) {
                const float value = float(pixel[c]) + (target[c] - float(pixel[c])) * a;
                pixel[c] = uint8_t(std::clamp(std::lround(value), 0l, 255l));
            }
        }
    }
}

} // namespace klartraum
