#include "klartraum/glfw_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/computegraph/computegraph.hpp"

int main() {
    const std::string model_path_encoder = "./data/onnx/simple_encoder.onnx";

    klartraum::GlfwFrontend frontend;

    auto& engine = frontend.getKlartraumEngine();

    auto& vulkanContext = engine.getVulkanContext();

    auto onnxNetwork = vulkanContext.create<klartraum::OnnxNetwork>(model_path_encoder);

    onnxNetwork->printModelInfo();

    klartraum::ComputeGraph graph(vulkanContext, 1);

    graph.compileFrom(onnxNetwork);

    return 0;
}
