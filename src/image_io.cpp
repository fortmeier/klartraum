// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/image_io.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

// stb_image_write (public domain / MIT, bundled with GLFW), with internal linkage so that programs may include their
// own copy.
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../3rdparty/glfw/deps/stb_image_write.h"

namespace klartraum {

void writeImage(const std::filesystem::path& path, const uint8_t* pixels, uint32_t width, uint32_t height,
                uint32_t channels) {
    const std::string extension = path.extension().string();
    if (extension == ".png") {
        if (channels < 1 || channels > 4)
            throw std::invalid_argument("writeImage: PNG takes 1 to 4 channels");
        if (!stbi_write_png(path.string().c_str(), int(width), int(height), int(channels), pixels,
                            int(width * channels)))
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
