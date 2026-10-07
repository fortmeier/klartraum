/**
 * TESTS:
 * - readFileFromWorkingDirectory: a relative path is read relative to the working directory
 * - readFileMissingThrows: reading a file that exists nowhere throws
 * - readFileFallsBackToAssetRoot: a relative path missing from the working directory is read from the asset root
 * - readFileWorkingDirectoryTakesPrecedence: a file present in both locations is read from the working directory
 * - readFileAssetRootFromEnvironment: without an explicit asset root, KLARTRAUM_ASSET_DIR is used
 * - assetRootDefaultsToBuildDirectory: without an explicit root or KLARTRAUM_ASSET_DIR, the asset root is the build directory that holds the compiled shaders
 * - readFileShaderFromOtherWorkingDirectory: a built-in shader path resolves from another working directory without any setup
 **/

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/vulkan_helpers.hpp"

using namespace klartraum;

namespace fs = std::filesystem;

namespace {

void writeText(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << text;
}

std::string asString(const std::vector<char>& data) {
    return std::string(data.begin(), data.end());
}

void setEnv(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

void unsetEnv(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

} // namespace

class VulkanHelpersTest : public ::testing::Test {
protected:
    void SetUp() override {
        root = fs::absolute("build/TestingOutput/vulkan_helpers");
        fs::remove_all(root);
        fs::create_directories(root);
        setAssetRoot("");
        unsetEnv("KLARTRAUM_ASSET_DIR");
    }

    void TearDown() override {
        setAssetRoot("");
        unsetEnv("KLARTRAUM_ASSET_DIR");
        fs::remove_all(root);
    }

    fs::path root;
};

TEST_F(VulkanHelpersTest, readFileFromWorkingDirectory) {
    const std::string relative = "build/TestingOutput/vulkan_helpers/in_working_directory.txt";
    writeText(relative, "from working directory");
    EXPECT_EQ(asString(readFile(relative)), "from working directory");
}

TEST_F(VulkanHelpersTest, readFileMissingThrows) {
    EXPECT_THROW(readFile("does/not/exist.bin"), std::runtime_error);
}

TEST_F(VulkanHelpersTest, readFileFallsBackToAssetRoot) {
    writeText(root / "assets/only_in_root.txt", "from asset root");
    EXPECT_THROW(readFile("only_in_root.txt"), std::runtime_error);

    setAssetRoot((root / "assets").string());
    EXPECT_EQ(asString(readFile("only_in_root.txt")), "from asset root");
}

TEST_F(VulkanHelpersTest, readFileWorkingDirectoryTakesPrecedence) {
    const std::string relative = "build/TestingOutput/vulkan_helpers/both.txt";
    writeText(relative, "from working directory");
    writeText(root / "assets" / relative, "from asset root");

    setAssetRoot((root / "assets").string());
    EXPECT_EQ(asString(readFile(relative)), "from working directory");
}

TEST_F(VulkanHelpersTest, readFileAssetRootFromEnvironment) {
    writeText(root / "env/only_in_env.txt", "from environment");
    setEnv("KLARTRAUM_ASSET_DIR", (root / "env").string());

    EXPECT_EQ(getAssetRoot(), (root / "env").string());
    EXPECT_EQ(asString(readFile("only_in_env.txt")), "from environment");

    // An explicit asset root overrides the environment.
    writeText(root / "explicit/only_in_env.txt", "from explicit root");
    setAssetRoot((root / "explicit").string());
    EXPECT_EQ(asString(readFile("only_in_env.txt")), "from explicit root");
}

TEST_F(VulkanHelpersTest, assetRootDefaultsToBuildDirectory) {
    const fs::path defaultRoot = getAssetRoot();
    ASSERT_FALSE(defaultRoot.empty());
    EXPECT_TRUE(fs::exists(defaultRoot / "shaders/operator_double.comp.spv"));
}

TEST_F(VulkanHelpersTest, readFileShaderFromOtherWorkingDirectory) {
    const fs::path workingDirectory = fs::current_path();
    const fs::path shaderPath = "shaders/gsplat/gsplat_projection.comp.spv";
    const auto expected = readFile((fs::path(getAssetRoot()) / shaderPath).string());

    fs::current_path(root);
    std::vector<char> actual;
    EXPECT_NO_THROW(actual = readFile(shaderPath.string()));
    fs::current_path(workingDirectory);

    EXPECT_GT(actual.size(), 0u);
    EXPECT_EQ(actual, expected);
}
