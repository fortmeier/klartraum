// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_BACKEND_CONFIG_HPP
#define KLARTRAUM_BACKEND_CONFIG_HPP

#include <stdint.h>

namespace klartraum {

class BackendConfig {
public:
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t WIDTH = 512;
    static constexpr uint32_t HEIGHT = 384;

    static constexpr char* ENGINE_VERSION = "Klartraum Engine v0.0.1";
};

} // namespace klartraum

#endif // KLARTRAUM_BACKEND_CONFIG_HPP
