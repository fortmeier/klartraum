/**
 * TESTS:
 * - ONNX initializers stored in a bounded external-data range are loaded and executed.
 * - External tensor paths cannot escape the ONNX model directory.
 **/

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"

using namespace klartraum;

namespace {

void setTensorInfo(onnx::ValueInfoProto* value, const std::string& name) {
    value->set_name(name);
    auto* tensorType = value->mutable_type()->mutable_tensor_type();
    tensorType->set_elem_type(onnx::TensorProto::FLOAT);
    tensorType->mutable_shape()->add_dim()->set_dim_value(4);
}

std::filesystem::path writeExternalReluModel(const std::string& modelName, const std::string& location) {
    const auto directory = std::filesystem::path("build/TestingOutput/onnx_external_data");
    std::filesystem::create_directories(directory);
    const auto modelPath = directory / modelName;
    const auto dataPath = directory / "weights.bin";
    const std::vector<float> values{-2.0f, -1.0f, 3.0f, 4.0f};
    {
        std::ofstream data(dataPath, std::ios::binary);
        const uint32_t prefix = 0x12345678;
        data.write(reinterpret_cast<const char*>(&prefix), sizeof(prefix));
        data.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    }

    onnx::ModelProto model;
    model.set_ir_version(8);
    model.add_opset_import()->set_version(17);
    auto* graph = model.mutable_graph();
    graph->set_name(modelName);
    setTensorInfo(graph->add_input(), "input");
    setTensorInfo(graph->add_output(), "output");
    auto* initializer = graph->add_initializer();
    initializer->set_name("input");
    initializer->set_data_type(onnx::TensorProto::FLOAT);
    initializer->add_dims(4);
    initializer->set_data_location(onnx::TensorProto::EXTERNAL);
    auto* locationEntry = initializer->add_external_data();
    locationEntry->set_key("location");
    locationEntry->set_value(location);
    auto* offsetEntry = initializer->add_external_data();
    offsetEntry->set_key("offset");
    offsetEntry->set_value("4");
    auto* lengthEntry = initializer->add_external_data();
    lengthEntry->set_key("length");
    lengthEntry->set_value(std::to_string(values.size() * sizeof(float)));
    auto* node = graph->add_node();
    node->set_name("relu");
    node->set_op_type("Relu");
    node->add_input("input");
    node->add_output("output");

    std::ofstream output(modelPath, std::ios::binary);
    if (!model.SerializeToOstream(&output)) throw std::runtime_error("Failed to write external-data test model");
    return modelPath;
}

} // namespace

TEST(OnnxExternalDataTest, LoadsBoundedInitializerRange) {
    const auto modelPath = writeExternalReluModel("external_relu.onnx", "weights.bin");
    HeadlessFrontend frontend;
    auto& context = frontend.getKlartraumEngine().getVulkanContext();
    auto network = context.create<OnnxNetwork>(modelPath.string());
    ComputeGraph graph(context, 1);
    graph.compileFrom(network);
    graph.submitAndWait(context.getGraphicsQueue(), 0);

    auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("output"));
    ASSERT_NE(output, nullptr);
    std::vector<float> actual(4);
    output->getDataBuffer(0).memcopyTo(actual);
    EXPECT_EQ(actual, (std::vector<float>{0.0f, 0.0f, 3.0f, 4.0f}));
    EXPECT_EQ(network->getFloatInitializerData("input"), (std::vector<float>{-2.0f, -1.0f, 3.0f, 4.0f}));
}

TEST(OnnxExternalDataTest, RejectsPathOutsideModelDirectory) {
    const auto modelPath = writeExternalReluModel("escaping_relu.onnx", "../weights.bin");
    HeadlessFrontend frontend;
    auto& context = frontend.getKlartraumEngine().getVulkanContext();
    auto network = context.create<OnnxNetwork>(modelPath.string());
    ComputeGraph graph(context, 1);
    EXPECT_THROW(graph.compileFrom(network), std::runtime_error);
}
