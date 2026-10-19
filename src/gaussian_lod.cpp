// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gaussian_lod.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <glm/glm.hpp>

namespace klartraum {

namespace {

// The cell of each Gaussian in a grid of `resolution` cells along the longest extent.
std::vector<uint64_t> cellIds(const std::vector<Gaussian3D>& gaussians, glm::vec3 low, float size, int resolution) {
    std::vector<uint64_t> ids(gaussians.size());
    const uint64_t stride = uint64_t(resolution) + 1;
    for (size_t i = 0; i < gaussians.size(); ++i) {
        const glm::vec3 p(gaussians[i].position[0], gaussians[i].position[1], gaussians[i].position[2]);
        const glm::uvec3 c = glm::uvec3(glm::floor((p - low) / size * float(resolution)));
        ids[i] = c.x + c.y * stride + c.z * stride * stride;
    }
    return ids;
}

size_t distinct(std::vector<uint64_t> ids) {
    std::sort(ids.begin(), ids.end());
    return size_t(std::unique(ids.begin(), ids.end()) - ids.begin());
}

} // namespace

std::vector<Gaussian3D> gaussianSpatialReduce(const std::vector<Gaussian3D>& gaussians, uint32_t budget) {
    const size_t n = gaussians.size();
    if (n <= budget) {
        return gaussians;
    }
    glm::vec3 low(1e30f), high(-1e30f);
    for (const Gaussian3D& g : gaussians) {
        const glm::vec3 p(g.position[0], g.position[1], g.position[2]);
        low = glm::min(low, p);
        high = glm::max(high, p);
    }
    const float size = std::max(std::max(high.x - low.x, high.y - low.y), std::max(high.z - low.z, 1e-6f));

    // How much each Gaussian is seen: its optical depth over the area of its two broad axes.
    std::vector<float> weight(n);
    for (size_t i = 0; i < n; ++i) {
        std::array<float, 3> s = gaussians[i].scale;
        std::sort(s.begin(), s.end());
        weight[i] = -std::log1p(-std::min(gaussians[i].alpha, 0.999f)) * s[1] * s[2];
    }

    // The finest grid whose occupied cells fit the budget.
    int lo = 2, hi = 512;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (distinct(cellIds(gaussians, low, size, mid)) <= budget) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const std::vector<uint64_t> ids = cellIds(gaussians, low, size, lo);

    // By cell, the most visible first.
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return ids[a] != ids[b] ? ids[a] < ids[b] : weight[a] > weight[b]; });
    std::vector<size_t> chosen;
    std::vector<bool> taken(n, false);
    for (size_t k = 0; k < n; ++k) {
        if (k == 0 || ids[order[k]] != ids[order[k - 1]]) {
            chosen.push_back(order[k]);
            taken[order[k]] = true;
        }
    }
    // The rest of the budget evenly over the others.
    std::vector<size_t> candidates;
    for (size_t k : order) {
        if (!taken[k]) {
            candidates.push_back(k);
        }
    }
    const size_t extra = budget - chosen.size();
    for (size_t e = 0; e < extra && !candidates.empty(); ++e) {
        const size_t index = extra > 1 ? e * (candidates.size() - 1) / (extra - 1) : 0;
        chosen.push_back(candidates[index]);
    }

    // Each cell keeps its coverage: the kept Gaussians' broad axes grow by sqrt(cell / kept), at most 3 times.
    std::vector<std::pair<uint64_t, float>> mass, kept;
    auto sumByCell = [&](const std::vector<size_t>& indices) {
        std::vector<std::pair<uint64_t, float>> sums;
        std::vector<size_t> sorted = indices;
        std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) { return ids[a] < ids[b]; });
        for (size_t i : sorted) {
            if (sums.empty() || sums.back().first != ids[i]) {
                sums.push_back({ids[i], 0.0f});
            }
            sums.back().second += weight[i];
        }
        return sums;
    };
    std::vector<size_t> all(n);
    std::iota(all.begin(), all.end(), 0);
    mass = sumByCell(all);
    kept = sumByCell(chosen);
    auto lookup = [](const std::vector<std::pair<uint64_t, float>>& sums, uint64_t id) {
        return std::lower_bound(sums.begin(), sums.end(), std::make_pair(id, -1e30f))->second;
    };

    std::vector<Gaussian3D> result;
    result.reserve(chosen.size());
    for (size_t i : chosen) {
        Gaussian3D g = gaussians[i];
        const float factor =
            std::clamp(std::sqrt(lookup(mass, ids[i]) / std::max(lookup(kept, ids[i]), 1e-12f)), 1.0f, 3.0f);
        std::array<int, 3> axes{0, 1, 2};
        std::sort(axes.begin(), axes.end(), [&](int a, int b) { return g.scale[a] < g.scale[b]; });
        g.scale[axes[1]] *= factor;
        g.scale[axes[2]] *= factor;
        result.push_back(g);
    }
    return result;
}

GaussianLods buildGaussianLods(const std::vector<Gaussian3D>& full, uint32_t minimum) {
    GaussianLods lods;
    lods.levels[0] = full;
    const uint32_t n = uint32_t(full.size());
    for (int level = 1; level < 4; ++level) {
        const uint32_t budget = std::max(n >> (2 * level), minimum);
        lods.levels[level] = gaussianSpatialReduce(lods.levels[level - 1], budget);
    }
    return lods;
}

} // namespace klartraum
