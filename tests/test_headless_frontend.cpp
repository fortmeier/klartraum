#include <gtest/gtest.h>

#include "klartraum/headless_frontend.hpp"

using namespace klartraum;

TEST(HeadlessFrontend, Smoketest) {
    HeadlessFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    core.add(core.createRenderPass());
    core.step();


}