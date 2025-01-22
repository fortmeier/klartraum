#include <vector>
#include <string>

namespace klartraum {

std::vector<char> readFile(const std::string& filename);

VkShaderModule createShaderModule(const std::vector<char>& code, const VkDevice& device);


} // namespace klartraum