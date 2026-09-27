#include "klartraum/gaussian_transform.hpp"

#include <cmath>
#include <stdexcept>

namespace klartraum {

std::array<float, 15> shBasis(const glm::vec3& dir) {
    const float x = dir.x, y = dir.y, z = dir.z;
    const float x2 = x * x, y2 = y * y, z2 = z * z;
    return {
        -0.488603f * y,
        -0.488603f * z,
        -0.488603f * x,
        1.092548f * x * y,
        -1.092548f * y * z,
        0.315392f * (2.0f * z2 - x2 - y2),
        -1.092548f * x * z,
        0.546274f * (x2 - y2),
        -0.590044f * y * (3.0f * x2 - y2),
        2.890611f * x * y * z,
        -0.457046f * y * (4.0f * z2 - x2 - y2),
        0.373176f * z * (2.0f * z2 - 3.0f * x2 - 3.0f * y2),
        -0.457046f * x * (4.0f * z2 - x2 - y2),
        1.445306f * z * (x2 - y2),
        -0.590044f * x * (x2 - 3.0f * y2),
    };
}

namespace {

constexpr int kBandStart[] = {0, 3, 8};
constexpr int kBandSize[] = {3, 5, 7};
constexpr int kMaxBandSize = 7;

using BandMatrix = std::array<std::array<double, kMaxBandSize>, kMaxBandSize>;

// Solves A x = b for an n x n system in place, leaving x in b (Gaussian
// elimination with partial pivoting).
void solve(BandMatrix& a, std::array<double, kMaxBandSize>& b, int n) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) {
                pivot = row;
            }
        }
        std::swap(a[col], a[pivot]);
        std::swap(b[col], b[pivot]);
        for (int row = col + 1; row < n; ++row) {
            const double f = a[row][col] / a[col][col];
            for (int k = col; k < n; ++k) {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        for (int k = row + 1; k < n; ++k) {
            b[row] -= a[row][k] * b[k];
        }
        b[row] /= a[row][row];
    }
}

// The matrices M, one per band, with which rotated coefficients are M times
// the original ones: the rotated colour at direction d is the original one at
// R^T d. Each band is closed under rotation, so fitting M to the basis
// evaluated at enough directions (least squares over a Fibonacci sphere) is
// exact up to rounding.
std::array<BandMatrix, 3> shRotation(const glm::mat3& rotation) {
    constexpr int kDirections = 64;
    std::array<std::array<float, 15>, kDirections> basis{}, rotated{};
    const glm::mat3 inverse = glm::transpose(rotation);
    for (int i = 0; i < kDirections; ++i) {
        const double z = 1.0 - (2.0 * i + 1.0) / kDirections;
        const double r = std::sqrt(1.0 - z * z);
        const double phi = i * 2.399963229728653;  // golden angle
        const glm::vec3 d(static_cast<float>(r * std::cos(phi)), static_cast<float>(r * std::sin(phi)),
                          static_cast<float>(z));
        basis[i] = shBasis(d);
        rotated[i] = shBasis(inverse * d);
    }

    std::array<BandMatrix, 3> result{};
    for (int band = 0; band < 3; ++band) {
        const int start = kBandStart[band], n = kBandSize[band];
        // Normal equations: (A^T A) M = A^T B, one column of M at a time.
        BandMatrix ata{};
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int i = 0; i < kDirections; ++i) {
                    ata[j][k] += double(basis[i][start + j]) * basis[i][start + k];
                }
            }
        }
        for (int col = 0; col < n; ++col) {
            BandMatrix a = ata;
            std::array<double, kMaxBandSize> b{};
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < kDirections; ++i) {
                    b[j] += double(basis[i][start + j]) * rotated[i][start + col];
                }
            }
            solve(a, b, n);
            for (int row = 0; row < n; ++row) {
                result[band][row][col] = b[row];
            }
        }
    }
    return result;
}

void rotateCoefficients(std::array<float, 15>& c, const std::array<BandMatrix, 3>& m) {
    const std::array<float, 15> original = c;
    for (int band = 0; band < 3; ++band) {
        const int start = kBandStart[band], n = kBandSize[band];
        for (int row = 0; row < n; ++row) {
            double value = 0.0;
            for (int col = 0; col < n; ++col) {
                value += m[band][row][col] * original[start + col];
            }
            c[start + row] = static_cast<float>(value);
        }
    }
}

} // namespace

void transformGaussians(std::vector<Gaussian3D>& gaussians, const glm::quat& rotation, float scale,
                        const glm::vec3& translation) {
    if (!(scale > 0.0f)) {
        throw std::invalid_argument("transformGaussians: scale must be positive");
    }
    const glm::quat q = glm::normalize(rotation);
    const glm::mat3 r = glm::mat3_cast(q);
    const bool rotates = std::abs(q.w) < 1.0f - 1e-7f;
    const std::array<BandMatrix, 3> sh = rotates ? shRotation(r) : std::array<BandMatrix, 3>{};

    for (auto& g : gaussians) {
        const glm::vec3 p = scale * (r * glm::vec3(g.position[0], g.position[1], g.position[2])) + translation;
        g.position = {p.x, p.y, p.z};
        for (float& s : g.scale) {
            s *= scale;
        }
        if (!rotates) {
            continue;
        }
        // Stored as x, y, z, w; glm::quat takes w first.
        const glm::quat own(g.rotation[3], g.rotation[0], g.rotation[1], g.rotation[2]);
        const glm::quat turned = glm::normalize(q * own);
        g.rotation = {turned.x, turned.y, turned.z, turned.w};
        rotateCoefficients(g.shR, sh);
        rotateCoefficients(g.shG, sh);
        rotateCoefficients(g.shB, sh);
    }
}

} // namespace klartraum
