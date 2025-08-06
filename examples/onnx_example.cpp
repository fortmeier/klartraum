#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx_network.hpp"

int main() {
    const std::string model_path = "C:\\Users\\dfort\\Desktop\\workspace\\super_resolution\\super-resolution-10.onnx";

    klartraum::GlfwFrontend frontend;

    auto& engine = frontend.getKlartraumEngine();

    auto& vulkanContext = engine.getVulkanContext();

    auto onnxNetwork = vulkanContext.create<klartraum::OnnxNetwork>(model_path);

    onnxNetwork->printModelInfo();

    return 0;
}
