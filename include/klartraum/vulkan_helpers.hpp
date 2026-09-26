#include <vector>
#include <string>

namespace klartraum {

// Reads a whole file into memory. A relative path that does not exist
// relative to the current working directory is looked up relative to the
// asset root (see setAssetRoot), so the library's built-in shader paths such
// as "shaders/gsplat/..." resolve from any working directory.
std::vector<char> readFile(const std::string& filename);

// Sets the directory that relative paths passed to readFile() fall back to,
// typically the klartraum source directory that holds the compiled shaders.
// While it is empty (the default), the KLARTRAUM_ASSET_DIR environment
// variable is used instead.
void setAssetRoot(const std::string& directory);

// The asset root set by setAssetRoot() or KLARTRAUM_ASSET_DIR; empty if none.
std::string getAssetRoot();

VkShaderModule createShaderModule(const std::vector<char>& code, const VkDevice& device);


} // namespace klartraum
