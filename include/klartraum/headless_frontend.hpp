// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_HEADLESS_FRONTEND_HPP
#define KLARTRAUM_HEADLESS_FRONTEND_HPP

#include "klartraum/klartraum_core.hpp"

namespace klartraum {

class HeadlessFrontend {
    /**
     * @brief User facing class
     *
     */
public:
    HeadlessFrontend();
    ~HeadlessFrontend();

    // void loop();

    KlartraumEngine& getKlartraumEngine();

private:
    void initialize();
    void shutdown();

    std::unique_ptr<KlartraumEngine> klartraumEngine;

    // VkSurfaceKHR surface;
};

} // namespace klartraum

#endif // KLARTRAUM_HEADLESS_FRONTEND_HPP
