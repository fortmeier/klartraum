// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/sd15/closeup_inpainting.hpp"

#include <algorithm>

#include "klartraum/sd15/latent_inpainting.hpp"

namespace klartraum {

std::vector<float> bgraToTensor(const std::vector<uint8_t>& bgra, size_t pixels, float low, float high) {
    std::vector<float> tensor(3 * pixels);
    for (size_t i = 0; i < pixels; ++i) {
        for (size_t channel = 0; channel < 3; ++channel) {
            tensor[channel * pixels + i] = low + (high - low) * float(bgra[4 * i + 2 - channel]) / 255.0f;
        }
    }
    return tensor;
}

std::vector<uint8_t> redChannel(const std::vector<uint8_t>& bgra) {
    std::vector<uint8_t> result(bgra.size() / 4);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = bgra[4 * i + 2];
    }
    return result;
}

std::vector<float> featherMask(const std::vector<uint8_t>& mask, int width, int height) {
    std::vector<float> result(mask.size(), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float coverage = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    const int sx = std::clamp(x + dx, 0, width - 1), sy = std::clamp(y + dy, 0, height - 1);
                    coverage += mask[size_t(sy) * size_t(width) + size_t(sx)] != 0 ? 1.0f : 0.0f;
                }
            }
            result[size_t(y) * size_t(width) + size_t(x)] = coverage / 25.0f;
        }
    }
    return result;
}

float sampleChannel(const std::vector<uint8_t>& bgra, int width, int height, glm::vec2 position, size_t channel) {
    const float x = std::clamp(position.x, 0.0f, float(width - 1)), y = std::clamp(position.y, 0.0f, float(height - 1));
    const int x0 = int(x), y0 = int(y), x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const float fx = x - float(x0), fy = y - float(y0);
    auto at = [&](int px, int py) { return float(bgra[4 * (size_t(py) * size_t(width) + size_t(px)) + channel]); };
    return (1.0f - fy) * ((1.0f - fx) * at(x0, y0) + fx * at(x1, y0)) +
           fy * ((1.0f - fx) * at(x0, y1) + fx * at(x1, y1));
}

std::vector<float> closeUpLatentMask(const std::vector<uint8_t>& marginMask, const std::vector<uint8_t>& bodyMask,
                                     uint32_t width, uint32_t height, uint32_t factor, uint32_t dilation) {
    const auto body = latentMask(bodyMask, width, height, factor, dilation);
    auto result = latentMask(marginMask, width, height, factor, dilation);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = body[i] > 0.0f ? 1.0f : result[i] * 0.5f;
    }
    return result;
}

glm::mat4 cropProjection(const glm::mat4& proj, glm::vec2 centre, glm::vec2 halfExtent) {
    glm::mat4 crop(1.0f);
    crop[0][0] = 1.0f / halfExtent.x;
    crop[1][1] = 1.0f / halfExtent.y;
    crop[3][0] = -centre.x / halfExtent.x;
    crop[3][1] = -centre.y / halfExtent.y;
    return crop * proj;
}

void CloseUpWindow::update(const std::vector<glm::vec4>& joints, const CameraMVP& full,
                           const glm::mat4& characterToScene, float aspect) {
    glm::vec2 lower(1e9f), upper(-1e9f);
    for (const auto& joint : joints) {
        const glm::vec4 clip = full.proj * full.view * characterToScene * glm::vec4(glm::vec3(joint), 1.0f);
        const glm::vec2 ndc = glm::vec2(clip) / clip.w;
        lower = glm::min(lower, ndc);
        upper = glm::max(upper, ndc);
    }
    // In units of the NDC y extent; x spans `aspect` times as many pixels per unit.
    const float target = std::max(0.08f, 0.8f * std::max(upper.y - lower.y, (upper.x - lower.x) * aspect));
    centre = 0.5f * (lower + upper);
    halfY = initialized ? halfY + 0.35f * (target - halfY) : target;
    halfY = std::max(halfY, 0.9f * target);
    halfExtent = glm::vec2(halfY / aspect, halfY);
    initialized = true;
}

CameraMVP CloseUpWindow::sceneCamera(const CameraMVP& full) const {
    CameraMVP camera = full;
    camera.proj = cropProjection(full.proj, centre, halfExtent);
    return camera;
}

CameraMVP CloseUpWindow::skeletonCamera(const CameraMVP& full, const glm::mat4& characterToScene) const {
    CameraMVP camera = sceneCamera(full);
    camera.model = characterToScene;
    return camera;
}

PastedCloseUp pasteGenerated(const std::vector<uint8_t>& scene, const std::vector<float>& generated,
                             const std::vector<uint8_t>& bodyMask, int size) {
    const size_t pixels = size_t(size) * size_t(size);
    const auto alpha = featherMask(bodyMask, size, size);
    PastedCloseUp result{scene, std::vector<uint8_t>(4 * pixels, 0)};
    for (size_t i = 0; i < pixels; ++i) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const float value = (std::clamp(generated[channel * pixels + i], -1.0f, 1.0f) + 1.0f) * 127.5f;
            const float original = float(scene[4 * i + 2 - channel]);
            result.image[4 * i + 2 - channel] = uint8_t(alpha[i] * value + (1.0f - alpha[i]) * original + 0.5f);
        }
        result.alpha[4 * i] = uint8_t(alpha[i] * 255.0f + 0.5f);
    }
    return result;
}

std::vector<uint8_t> compositeCloseUp(const std::vector<uint8_t>& scene, const std::vector<uint8_t>& characterMask,
                                      const PastedCloseUp& closeUp, int size, int width, int height,
                                      const CloseUpWindow& window) {
    std::vector<uint8_t> result = scene;
    const auto region = featherMask(redChannel(characterMask), width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = size_t(y) * size_t(width) + size_t(x);
            if (region[i] == 0.0f) {
                continue;
            }
            // The pixel's NDC in the full view, then in the close-up, then its close-up pixel.
            const glm::vec2 ndc((float(x) + 0.5f) / float(width) * 2.0f - 1.0f,
                                (float(y) + 0.5f) / float(height) * 2.0f - 1.0f);
            const glm::vec2 closeUpNdc = (ndc - window.centre) / window.halfExtent;
            const glm::vec2 source = (closeUpNdc * 0.5f + 0.5f) * float(size) - 0.5f;
            const float alpha = sampleChannel(closeUp.alpha, size, size, source, 0) / 255.0f;
            for (size_t channel = 0; channel < 3; ++channel) {
                const float value = sampleChannel(closeUp.image, size, size, source, 2 - channel);
                result[4 * i + 2 - channel] =
                    uint8_t(alpha * value + (1.0f - alpha) * float(scene[4 * i + 2 - channel]) + 0.5f);
            }
        }
    }
    return result;
}

} // namespace klartraum
