// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - compiledShadersAreSpirV: every .spv file the build writes to the asset root's shaders/ directory starts with the
 *   SPIR-V magic number
 * - referencedShadersExist: every literal "shaders/....spv" path in src/, include/, examples/ and tests/ resolves
 *   through readFile(), so the build produces every module the code loads
 * - compiledShadersCreateModules: the test device accepts every compiled module in vkCreateShaderModule
 **/

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_helpers.hpp"

using namespace klartraum;

namespace fs = std::filesystem;

namespace {

// Returns the paths, relative to the asset root, of all compiled shaders.
std::vector<std::string> compiledShaders() {
    const fs::path root = fs::path(getAssetRoot());
    std::vector<std::string> shaders;
    for (const auto& entry : fs::recursive_directory_iterator(root / "shaders")) {
        if (entry.is_regular_file() && entry.path().extension() == ".spv") {
            shaders.push_back(fs::relative(entry.path(), root).generic_string());
        }
    }
    return shaders;
}

// Returns the shader paths written as string literals in the C++ sources below
// the working directory's src/, include/, examples/ and tests/.
std::set<std::string> referencedShaders() {
    const std::regex literal("\"(shaders/[A-Za-z0-9_/]+\\.(comp|vert|frag|mesh)\\.spv)\"");
    std::set<std::string> shaders;
    for (const char* directory : {"src", "include", "examples", "tests"}) {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            const auto extension = entry.path().extension();
            if (!entry.is_regular_file() || (extension != ".cpp" && extension != ".hpp")) {
                continue;
            }
            std::ifstream in(entry.path(), std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            for (auto it = std::sregex_iterator(text.begin(), text.end(), literal); it != std::sregex_iterator();
                 ++it) {
                shaders.insert((*it)[1].str());
            }
        }
    }
    return shaders;
}

} // namespace

TEST(ShaderAssets, compiledShadersAreSpirV) {
    const auto shaders = compiledShaders();
    ASSERT_FALSE(shaders.empty()) << "no compiled shaders below " << getAssetRoot();
    for (const auto& shader : shaders) {
        const auto code = readFile(shader);
        ASSERT_GE(code.size(), sizeof(uint32_t)) << shader;
        uint32_t magic = 0;
        std::memcpy(&magic, code.data(), sizeof(magic));
        EXPECT_EQ(magic, 0x07230203u) << shader;
        EXPECT_EQ(code.size() % sizeof(uint32_t), 0u) << shader;
    }
}

TEST(ShaderAssets, referencedShadersExist) {
    ASSERT_TRUE(fs::is_directory("src")) << "run the tests from the repository root";
    const auto shaders = referencedShaders();
    ASSERT_GT(shaders.size(), 50u);
    for (const auto& shader : shaders) {
        EXPECT_NO_THROW(readFile(shader)) << shader;
    }
}

TEST(ShaderAssets, compiledShadersCreateModules) {
    auto frontend = std::make_unique<HeadlessFrontend>();
    const VkDevice device = frontend->getKlartraumEngine().getVulkanContext().getDevice();
    for (const auto& shader : compiledShaders()) {
        const auto code = readFile(shader);
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module = VK_NULL_HANDLE;
        EXPECT_EQ(vkCreateShaderModule(device, &createInfo, nullptr, &module), VK_SUCCESS) << shader;
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, module, nullptr);
        }
    }
}
