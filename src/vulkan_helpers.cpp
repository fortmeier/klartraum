#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

namespace {

std::mutex assetRootMutex;
std::string assetRoot;

} // namespace

void setAssetRoot(const std::string& directory) {
    std::lock_guard<std::mutex> lock(assetRootMutex);
    assetRoot = directory;
}

std::string getAssetRoot() {
    std::lock_guard<std::mutex> lock(assetRootMutex);
    if (!assetRoot.empty()) {
        return assetRoot;
    }
    const char* env = std::getenv("KLARTRAUM_ASSET_DIR");
    return env ? std::string(env) : std::string();
}

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    // The working directory takes precedence, so existing callers that run
    // from the repository root behave as before.
    const std::filesystem::path path(filename);
    if (!file.is_open() && path.is_relative()) {
        const std::string root = getAssetRoot();
        if (!root.empty()) {
            file.open(std::filesystem::path(root) / path, std::ios::ate | std::ios::binary);
        }
    }

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& code, const VkDevice& device) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

} // namespace klartraum