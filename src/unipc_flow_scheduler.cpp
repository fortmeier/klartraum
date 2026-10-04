// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/cosmos3/unipc_flow_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace klartraum {

namespace {

/** Solves the small dense system A x = b by Gaussian elimination with partial pivoting. */
std::vector<double> solve(std::vector<std::vector<double>> a, std::vector<double> b) {
    const size_t n = b.size();
    for (size_t column = 0; column < n; ++column) {
        size_t pivot = column;
        for (size_t row = column + 1; row < n; ++row) {
            if (std::abs(a[row][column]) > std::abs(a[pivot][column]))
                pivot = row;
        }
        std::swap(a[column], a[pivot]);
        std::swap(b[column], b[pivot]);
        for (size_t row = column + 1; row < n; ++row) {
            const double factor = a[row][column] / a[column][column];
            for (size_t k = column; k < n; ++k)
                a[row][k] -= factor * a[column][k];
            b[row] -= factor * b[column];
        }
    }
    std::vector<double> x(n);
    for (size_t row = n; row-- > 0;) {
        double sum = b[row];
        for (size_t k = row + 1; k < n; ++k)
            sum -= a[row][k] * x[k];
        x[row] = sum / a[row][row];
    }
    return x;
}

} // namespace

UniPCFlowScheduler::UniPCFlowScheduler(size_t steps, double flowShift, uint32_t trainTimesteps, uint32_t order)
    : solverOrder(order) {
    if (steps == 0 || order == 0 || order > 3)
        throw std::runtime_error("Invalid UniPC configuration");
    const double first = 1.0 - 1.0 / trainTimesteps;
    for (size_t i = 0; i < steps; ++i) {
        // linspace(first, 0, steps + 1) without its last entry, then the static flow shift.
        const double sigma = first - first * double(i) / double(steps);
        float shifted = static_cast<float>(flowShift * sigma / (1.0 + (flowShift - 1.0) * sigma));
        if (std::abs(shifted - 1.0f) < 1e-6f)
            shifted -= 1e-6f;
        sigmaValues.push_back(shifted);
        timestepValues.push_back(std::floor(shifted * static_cast<float>(trainTimesteps)));
    }
    sigmaValues.push_back(0.0f);
}

double UniPCFlowScheduler::lambda(size_t index) const {
    // Flow matching: alpha = 1 - sigma. lambda = log(alpha / sigma), +inf at sigma = 0.
    const double sigma = sigmaValues[index];
    return std::log(1.0 - sigma) - std::log(sigma);
}

UniPCFlowScheduler::Coefficients UniPCFlowScheduler::coefficients(size_t target, size_t source, uint32_t order,
                                                                  bool corrector) const {
    const double sigmaT = sigmaValues[target];
    const double alphaT = 1.0 - sigmaT;
    const double lambdaS0 = lambda(source);
    const double h = lambda(target) - lambdaS0;

    Coefficients result;
    // The predictor's earlier outputs belong to steps source-1, source-2, ...;
    // the corrector's to one step further back, since its newest output is the current one.
    for (uint32_t k = 1; k < order; ++k) {
        const size_t index = source - k;
        result.ratios.push_back((lambda(index) - lambdaS0) / h);
    }
    std::vector<double> rks = result.ratios;
    rks.push_back(1.0);

    const double hh = -h;
    const double hPhi1 = std::expm1(hh);
    const double bH = std::expm1(hh);
    double hPhiK = hPhi1 / hh - 1.0;
    double factorial = 1.0;
    std::vector<std::vector<double>> r;
    std::vector<double> b;
    for (uint32_t i = 1; i <= order; ++i) {
        std::vector<double> row;
        for (double rk : rks)
            row.push_back(std::pow(rk, double(i - 1)));
        r.push_back(row);
        b.push_back(hPhiK * factorial / bH);
        factorial *= i + 1;
        hPhiK = hPhiK / hh - 1.0 / factorial;
    }

    if (corrector) {
        if (order == 1) {
            result.rhos = {0.5};
        } else {
            result.rhos = solve(r, b);
        }
    } else if (order == 2) {
        result.rhos = {0.5};
    } else if (order > 2) {
        std::vector<std::vector<double>> reduced(order - 1);
        for (uint32_t i = 0; i + 1 < order; ++i)
            reduced[i].assign(r[i].begin(), r[i].end() - 1);
        result.rhos = solve(reduced, std::vector<double>(b.begin(), b.end() - 1));
    }
    result.sampleScale = sigmaT / sigmaValues[source];
    result.outputScale = alphaT * hPhi1;
    result.differenceScale = alphaT * bH;
    return result;
}

void UniPCFlowScheduler::step(const std::vector<float>& velocity, std::vector<float>& sample) {
    const size_t steps = timestepValues.size();
    if (currentStep >= steps)
        throw std::runtime_error("UniPC scheduler has no steps left");
    if (velocity.size() != sample.size())
        throw std::runtime_error("UniPC velocity and sample sizes differ");
    const size_t n = sample.size();

    // Data prediction from the flow velocity: x0 = x - sigma * v.
    std::vector<float> x0(n);
    const float sigma = sigmaValues[currentStep];
    for (size_t i = 0; i < n; ++i)
        x0[i] = sample[i] - sigma * velocity[i];

    if (currentStep > 0 && !lastSample.empty()) {
        // UniC: correct the previous prediction using this step's model output.
        const auto c = coefficients(currentStep, currentStep - 1, lastOrder, true);
        const auto& m0 = modelOutputs.back();
        for (size_t i = 0; i < n; ++i) {
            double correction = c.rhos.back() * (double(x0[i]) - m0[i]);
            for (size_t k = 0; k < c.ratios.size(); ++k) {
                const auto& mk = modelOutputs[modelOutputs.size() - 2 - k];
                correction += c.rhos[k] * (double(mk[i]) - m0[i]) / c.ratios[k];
            }
            sample[i] = static_cast<float>(c.sampleScale * lastSample[i] - c.outputScale * m0[i] -
                                           c.differenceScale * correction);
        }
    }

    modelOutputs.push_back(std::move(x0));
    if (modelOutputs.size() > solverOrder)
        modelOutputs.erase(modelOutputs.begin());

    uint32_t order = std::min<uint32_t>(solverOrder, static_cast<uint32_t>(steps - currentStep));
    order = std::min(order, lowerOrderCount + 1);
    lastOrder = order;
    lastSample = sample;

    // UniP: predict the sample at the next noise level.
    const auto c = coefficients(currentStep + 1, currentStep, order, false);
    const auto& m0 = modelOutputs.back();
    for (size_t i = 0; i < n; ++i) {
        double prediction = 0.0;
        for (size_t k = 0; k < c.ratios.size(); ++k) {
            const auto& mk = modelOutputs[modelOutputs.size() - 2 - k];
            prediction += c.rhos[k] * (double(mk[i]) - m0[i]) / c.ratios[k];
        }
        sample[i] =
            static_cast<float>(c.sampleScale * lastSample[i] - c.outputScale * m0[i] - c.differenceScale * prediction);
    }

    if (lowerOrderCount < solverOrder)
        ++lowerOrderCount;
    ++currentStep;
}

} // namespace klartraum
