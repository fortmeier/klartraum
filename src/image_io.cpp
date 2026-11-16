// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/image_io.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include <zlib.h>

namespace klartraum {

namespace {

void appendBigEndian(std::vector<uint8_t>& out, uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
        out.push_back(uint8_t(value >> shift));
}

// A PNG chunk: length, type, data and the CRC of type and data.
void appendChunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    appendBigEndian(out, uint32_t(data.size()));
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    appendBigEndian(out, uint32_t(::crc32(0, out.data() + start, uInt(out.size() - start))));
}

uint8_t paeth(int a, int b, int c) {
    const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    return uint8_t(pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
}

// An 8-bit PNG (colour type grey, grey + alpha, RGB or RGBA by `channels`): every row takes the filter (None, Sub,
// Up, Average or Paeth) whose output has the smallest sum of absolute values (the heuristic of the PNG
// specification), the filtered rows are deflated by zlib.
std::vector<uint8_t> encodePng(const uint8_t* pixels, uint32_t width, uint32_t height, uint32_t channels) {
    const size_t stride = size_t(width) * channels;
    std::vector<uint8_t> filtered;
    filtered.reserve((stride + 1) * height);
    std::vector<uint8_t> candidate(stride), best(stride);
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* row = pixels + y * stride;
        const uint8_t* above = y > 0 ? row - stride : nullptr;
        uint64_t bestCost = UINT64_MAX;
        uint8_t bestFilter = 0;
        for (uint8_t filter = 0; filter < 5; ++filter) {
            uint64_t cost = 0;
            for (size_t i = 0; i < stride; ++i) {
                const int a = i >= channels ? row[i - channels] : 0, b = above ? above[i] : 0,
                          c = above && i >= channels ? above[i - channels] : 0;
                const uint8_t predicted = filter == 0   ? 0
                                          : filter == 1 ? uint8_t(a)
                                          : filter == 2 ? uint8_t(b)
                                          : filter == 3 ? uint8_t((a + b) / 2)
                                                        : paeth(a, b, c);
                candidate[i] = uint8_t(row[i] - predicted);
                cost += std::abs(int(int8_t(candidate[i])));
            }
            if (cost < bestCost) {
                bestCost = cost;
                bestFilter = filter;
                best.swap(candidate);
            }
        }
        filtered.push_back(bestFilter);
        filtered.insert(filtered.end(), best.begin(), best.end());
    }

    uLongf compressedSize = compressBound(uLong(filtered.size()));
    std::vector<uint8_t> compressed(compressedSize);
    if (compress2(compressed.data(), &compressedSize, filtered.data(), uLong(filtered.size()), 6) != Z_OK)
        throw std::runtime_error("writeImage: PNG compression failed");
    compressed.resize(compressedSize);

    std::vector<uint8_t> png{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    std::vector<uint8_t> header;
    appendBigEndian(header, width);
    appendBigEndian(header, height);
    const uint8_t colourType = channels == 1 ? 0 : channels == 2 ? 4 : channels == 3 ? 2 : 6;
    header.insert(header.end(), {8, colourType, 0, 0, 0}); // bit depth, colour type, deflate, adaptive filters, no
                                                           // interlace
    appendChunk(png, "IHDR", header);
    appendChunk(png, "IDAT", compressed);
    appendChunk(png, "IEND", {});
    return png;
}

} // namespace

void writeImage(const std::filesystem::path& path, const uint8_t* pixels, uint32_t width, uint32_t height,
                uint32_t channels) {
    const std::string extension = path.extension().string();
    if (extension == ".png") {
        if (channels < 1 || channels > 4)
            throw std::invalid_argument("writeImage: PNG takes 1 to 4 channels");
        const auto png = encodePng(pixels, width, height, channels);
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(png.data()), std::streamsize(png.size()));
        if (!file)
            throw std::runtime_error("writeImage: cannot write " + path.string());
        return;
    }
    const bool grey = extension == ".pgm";
    if (!grey && extension != ".ppm")
        throw std::invalid_argument("writeImage: unknown image format " + extension);
    if (channels != (grey ? 1u : 3u))
        throw std::invalid_argument("writeImage: " + extension + " takes " + (grey ? "1 channel" : "3 channels"));
    std::ofstream file(path, std::ios::binary);
    file << (grey ? "P5" : "P6") << "\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels), std::streamsize(size_t(width) * height * channels));
    if (!file)
        throw std::runtime_error("writeImage: cannot write " + path.string());
}

void writeImageBgra(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, uint32_t width,
                    uint32_t height) {
    std::vector<uint8_t> rgb(size_t(width) * height * 3);
    for (size_t p = 0; p < size_t(width) * height; ++p) {
        rgb[3 * p] = bgra[4 * p + 2];
        rgb[3 * p + 1] = bgra[4 * p + 1];
        rgb[3 * p + 2] = bgra[4 * p];
    }
    writeImage(path, rgb.data(), width, height, 3);
}

} // namespace klartraum
