#ifndef BACKEND_CONFIG_HPP
#define BACKEND_CONFIG_HPP

#include <stdint.h>

namespace klartraum {

class BackendConfig {
public:
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t WIDTH = 512;
    static constexpr uint32_t HEIGHT = 512;

    static constexpr char* ENGINE_VERSION = "Klartraum Engine v0.0.1";
};

} // namespace klartraum

#endif // BACKEND_CONFIG_HPP