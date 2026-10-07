// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_VULKAN_HELPERS_HPP
#define KLARTRAUM_VULKAN_HELPERS_HPP

#include <vector>
#include <string>

namespace klartraum {

/**
 * @brief Reads a whole file into memory.
 *
 * A relative path that does not exist relative to the current working directory
 * is looked up relative to the asset root (see setAssetRoot()), so the library's
 * built-in shader paths such as `shaders/gsplat/...` resolve from any working
 * directory.
 */
std::vector<char> readFile(const std::string& filename);

/**
 * @brief Sets the directory that relative paths passed to readFile() fall back to.
 *
 * That is a directory with the compiled shaders in its `shaders/` subdirectory.
 * While it is empty (the default), the `KLARTRAUM_ASSET_DIR` environment variable
 * is used, and without that the build directory of Klartraum.
 */
void setAssetRoot(const std::string& directory);

/**
 * @brief The asset root set by setAssetRoot(), else `KLARTRAUM_ASSET_DIR`, else the build directory of Klartraum.
 */
std::string getAssetRoot();

VkShaderModule createShaderModule(const std::vector<char>& code, const VkDevice& device);

} // namespace klartraum

#endif // KLARTRAUM_VULKAN_HELPERS_HPP
