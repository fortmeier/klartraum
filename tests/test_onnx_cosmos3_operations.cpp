// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - ReduceMean averages the last axis with keepdims, as in an RMSNorm variance.
 * - ReduceMean averages a leading channel axis, as in the Wan VAE channel RMSNorm.
 * - ReduceMean averages several contiguous axes and drops them without keepdims.
 * - Mul broadcasts a per-channel rank-five scale over a video tensor.
 * - Add combines two equally shaped rank-six tensors.
 * - Div broadcasts a rank-five per-pixel denominator over the channel axis.
 * - Slice cuts frames out of a rank-five video tensor on its time axis.
 * - Add broadcasts a rank-zero scalar initializer, as the exporters write epsilon constants.
 * - Conv3d with a causal 3x3x3 kernel (two leading time pads) matches a CPU reference across tile boundaries.
 * - Conv3d downsamples spatially with stride two and trailing-only pads.
 * - Conv3d applies a causal 3x1x1 temporal kernel.
 * - Conv3d applies a pointwise 1x1x1 kernel with bias.
 * - Single-head rank-three attention with a 1024-wide head runs unfused and matches a CPU reference.
 **/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "onnx.pb.h"

using namespace klartraum;

namespace {

using Shape = std::vector<uint32_t>;

size_t elementCount(const Shape& shape) {
    size_t count = 1;
    for (auto dimension : shape)
        count *= dimension;
    return count;
}

std::vector<float> patternedValues(size_t count, uint32_t seed) {
    std::vector<float> values(count);
    uint32_t state = seed;
    for (auto& value : values) {
        state = state * 1664525u + 1013904223u;
        value = float(state >> 8) / float(1u << 24) - 0.5f;
    }
    return values;
}

/** Builds a fixed-shape single-graph ONNX model whose float inputs are bound at run time. */
class ModelBuilder {
public:
    explicit ModelBuilder(const std::string& name)
        : name(name) {
        model.set_ir_version(8);
        model.add_opset_import()->set_version(17);
        model.mutable_graph()->set_name(name);
    }

    void input(const std::string& tensor, const Shape& shape) {
        setInfo(model.mutable_graph()->add_input(), tensor, shape);
        inputs[tensor] = shape;
    }

    void output(const std::string& tensor, const Shape& shape) {
        setInfo(model.mutable_graph()->add_output(), tensor, shape);
    }

    onnx::NodeProto* node(const std::string& opType, const std::vector<std::string>& nodeInputs,
                          const std::vector<std::string>& nodeOutputs) {
        auto* node = model.mutable_graph()->add_node();
        node->set_op_type(opType);
        node->set_name(opType);
        for (const auto& tensor : nodeInputs)
            node->add_input(tensor);
        for (const auto& tensor : nodeOutputs)
            node->add_output(tensor);
        return node;
    }

    void int64Initializer(const std::string& tensor, const std::vector<int64_t>& values) {
        auto* initializer = model.mutable_graph()->add_initializer();
        initializer->set_name(tensor);
        initializer->set_data_type(onnx::TensorProto::INT64);
        initializer->add_dims(static_cast<int64_t>(values.size()));
        initializer->set_raw_data(values.data(), values.size() * sizeof(int64_t));
        // Klartraum reads every tensor's shape from value_info, as the exporters write it.
        auto* info = model.mutable_graph()->add_value_info();
        info->set_name(tensor);
        auto* type = info->mutable_type()->mutable_tensor_type();
        type->set_elem_type(onnx::TensorProto::INT64);
        type->mutable_shape()->add_dim()->set_dim_value(static_cast<int64_t>(values.size()));
    }

    void floatInitializer(const std::string& tensor, const Shape& shape, const std::vector<float>& values) {
        auto* initializer = model.mutable_graph()->add_initializer();
        initializer->set_name(tensor);
        initializer->set_data_type(onnx::TensorProto::FLOAT);
        for (auto dimension : shape)
            initializer->add_dims(dimension);
        initializer->set_raw_data(values.data(), values.size() * sizeof(float));
        setInfo(model.mutable_graph()->add_value_info(), tensor, shape);
    }

    static void ints(onnx::NodeProto* node, const std::string& attribute, const std::vector<int64_t>& values) {
        auto* proto = node->add_attribute();
        proto->set_name(attribute);
        proto->set_type(onnx::AttributeProto::INTS);
        for (auto value : values)
            proto->add_ints(value);
    }

    static void integer(onnx::NodeProto* node, const std::string& attribute, int64_t value) {
        auto* proto = node->add_attribute();
        proto->set_name(attribute);
        proto->set_type(onnx::AttributeProto::INT);
        proto->set_i(value);
    }

    /** Writes the model, binds @p values to its inputs, runs it, and reads @p outputName. */
    std::vector<float> run(const std::map<std::string, std::vector<float>>& values, const std::string& outputName) {
        const auto directory = std::filesystem::path("build/TestingOutput/onnx_cosmos3_operations");
        std::filesystem::create_directories(directory);
        const auto path = directory / (name + ".onnx");
        {
            std::ofstream file(path, std::ios::binary);
            if (!model.SerializeToOstream(&file))
                throw std::runtime_error("Failed to write " + path.string());
        }
        HeadlessFrontend frontend;
        auto& context = frontend.getKlartraumEngine().getVulkanContext();
        auto network = context.create<OnnxNetwork>(path.string());
        std::map<std::string, std::shared_ptr<TensorElement<float>>> bound;
        for (const auto& [tensor, shape] : inputs) {
            constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bound[tensor] = context.create<TensorElement<float>>(shape, usage);
            network->setInputTensor(tensor, bound[tensor]);
        }
        ComputeGraph graph(context, 1);
        graph.compileFrom(network);
        for (const auto& [tensor, data] : values)
            bound.at(tensor)->setData(0, data);
        graph.submitAndWait(context.getGraphicsQueue(), 0);
        auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement(outputName));
        if (!output)
            throw std::runtime_error("Missing float output " + outputName);
        std::vector<float> result(output->getDataElementCount());
        output->getDataBuffer(0).memcopyTo(result);
        return result;
    }

private:
    static void setInfo(onnx::ValueInfoProto* value, const std::string& tensor, const Shape& shape) {
        value->set_name(tensor);
        auto* type = value->mutable_type()->mutable_tensor_type();
        type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* dims = type->mutable_shape();
        for (auto dimension : shape)
            dims->add_dim()->set_dim_value(dimension);
    }

    std::string name;
    onnx::ModelProto model;
    std::map<std::string, Shape> inputs;
};

/** Mean over axes [first, first + count) of a row-major tensor. */
std::vector<float> reduceMeanReference(const std::vector<float>& input, const Shape& shape, size_t first,
                                       size_t count) {
    size_t outer = 1, axis = 1, inner = 1;
    for (size_t i = 0; i < first; ++i)
        outer *= shape[i];
    for (size_t i = first; i < first + count; ++i)
        axis *= shape[i];
    for (size_t i = first + count; i < shape.size(); ++i)
        inner *= shape[i];
    std::vector<float> output(outer * inner);
    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            double sum = 0.0;
            for (size_t a = 0; a < axis; ++a)
                sum += input[(o * axis + a) * inner + i];
            output[o * inner + i] = float(sum / double(axis));
        }
    }
    return output;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_NEAR(actual[i], expected[i], tolerance) << "at element " << i;
    }
}

} // namespace

TEST(OnnxCosmos3OperationsTest, ReduceMeanLastAxisKeepDims) {
    const Shape shape{2, 5, 3, 128};
    ModelBuilder builder("reduce_mean_last_axis");
    builder.input("x", shape);
    builder.output("y", {2, 5, 3, 1});
    ModelBuilder::ints(builder.node("ReduceMean", {"x"}, {"y"}), "axes", {-1});
    const auto x = patternedValues(elementCount(shape), 1);
    expectNear(builder.run({{"x", x}}, "y"), reduceMeanReference(x, shape, 3, 1), 1e-6f);
}

TEST(OnnxCosmos3OperationsTest, ReduceMeanChannelAxis) {
    const Shape shape{1, 24, 3, 5, 7};
    ModelBuilder builder("reduce_mean_channel_axis");
    builder.input("x", shape);
    builder.output("y", {1, 1, 3, 5, 7});
    ModelBuilder::ints(builder.node("ReduceMean", {"x"}, {"y"}), "axes", {1});
    const auto x = patternedValues(elementCount(shape), 2);
    expectNear(builder.run({{"x", x}}, "y"), reduceMeanReference(x, shape, 1, 1), 1e-6f);
}

TEST(OnnxCosmos3OperationsTest, ReduceMeanContiguousAxesWithoutKeepDims) {
    const Shape shape{3, 4, 6, 5};
    ModelBuilder builder("reduce_mean_contiguous_axes");
    builder.input("x", shape);
    builder.output("y", {3, 5});
    auto* node = builder.node("ReduceMean", {"x"}, {"y"});
    ModelBuilder::ints(node, "axes", {2, 1});
    ModelBuilder::integer(node, "keepdims", 0);
    const auto x = patternedValues(elementCount(shape), 3);
    expectNear(builder.run({{"x", x}}, "y"), reduceMeanReference(x, shape, 1, 2), 1e-6f);
}

namespace {

/** Elementwise reference with ONNX right-aligned broadcasting. */
template <typename Op>
std::vector<float> broadcastReference(const std::vector<float>& lhs, const Shape& lhsShape,
                                      const std::vector<float>& rhs, const Shape& rhsShape, const Shape& output,
                                      Op op) {
    std::vector<float> result(elementCount(output));
    auto index = [&output](const Shape& shape, size_t linear) {
        size_t offset = 0, stride = 1;
        for (size_t axis = output.size(); axis-- > 0;) {
            const size_t coordinate = linear % output[axis];
            linear /= output[axis];
            const size_t shift = output.size() - shape.size();
            if (axis < shift)
                continue;
            const size_t extent = shape[axis - shift];
            offset += (extent == 1 ? 0 : coordinate) * stride;
            stride *= extent;
        }
        return offset;
    };
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = op(lhs[index(lhsShape, i)], rhs[index(rhsShape, i)]);
    }
    return result;
}

} // namespace

TEST(OnnxCosmos3OperationsTest, MulBroadcastsChannelScaleOverRankFiveVideo) {
    const Shape video{1, 6, 3, 4, 5}, scale{1, 6, 1, 1, 1};
    ModelBuilder builder("mul_rank5_channel_scale");
    builder.input("x", video);
    builder.input("s", scale);
    builder.output("y", video);
    builder.node("Mul", {"x", "s"}, {"y"});
    const auto x = patternedValues(elementCount(video), 4);
    const auto s = patternedValues(elementCount(scale), 5);
    expectNear(builder.run({{"x", x}, {"s", s}}, "y"),
               broadcastReference(x, video, s, scale, video, [](float a, float b) { return a * b; }), 1e-6f);
}

TEST(OnnxCosmos3OperationsTest, AddCombinesEquallyShapedRankSixTensors) {
    const Shape shape{2, 1, 3, 2, 4, 5};
    ModelBuilder builder("add_rank6");
    builder.input("a", shape);
    builder.input("b", shape);
    builder.output("y", shape);
    builder.node("Add", {"a", "b"}, {"y"});
    const auto a = patternedValues(elementCount(shape), 6);
    const auto b = patternedValues(elementCount(shape), 7);
    expectNear(builder.run({{"a", a}, {"b", b}}, "y"),
               broadcastReference(a, shape, b, shape, shape, [](float x, float y) { return x + y; }), 1e-6f);
}

TEST(OnnxCosmos3OperationsTest, DivBroadcastsPerPixelDenominatorOverChannels) {
    const Shape video{1, 4, 2, 3, 5}, norm{1, 1, 2, 3, 5};
    ModelBuilder builder("div_rank5_per_pixel");
    builder.input("x", video);
    builder.input("n", norm);
    builder.output("y", video);
    builder.node("Div", {"x", "n"}, {"y"});
    const auto x = patternedValues(elementCount(video), 8);
    auto n = patternedValues(elementCount(norm), 9);
    for (auto& value : n)
        value += 1.0f;
    expectNear(builder.run({{"x", x}, {"n", n}}, "y"),
               broadcastReference(x, video, n, norm, video, [](float a, float b) { return a / b; }), 1e-6f);
}

TEST(OnnxCosmos3OperationsTest, SliceCutsFramesFromRankFiveVideo) {
    const Shape video{1, 3, 5, 4, 2}, frames{1, 3, 3, 4, 2};
    ModelBuilder builder("slice_rank5_time");
    builder.input("x", video);
    builder.output("y", frames);
    builder.int64Initializer("starts", {1});
    builder.int64Initializer("ends", {4});
    builder.int64Initializer("axes", {2});
    builder.node("Slice", {"x", "starts", "ends", "axes"}, {"y"});
    const auto x = patternedValues(elementCount(video), 10);
    std::vector<float> expected;
    for (uint32_t c = 0; c < 3; ++c) {
        for (uint32_t t = 1; t < 4; ++t) {
            for (uint32_t i = 0; i < 8; ++i)
                expected.push_back(x[(c * 5 + t) * 8 + i]);
        }
    }
    expectNear(builder.run({{"x", x}}, "y"), expected, 0.0f);
}

TEST(OnnxCosmos3OperationsTest, AddBroadcastsRankZeroScalarInitializer) {
    const Shape shape{2, 3, 1};
    ModelBuilder builder("add_rank0_scalar");
    builder.input("x", shape);
    builder.output("y", shape);
    builder.floatInitializer("epsilon", {}, {0.25f});
    builder.node("Add", {"x", "epsilon"}, {"y"});
    const auto x = patternedValues(elementCount(shape), 11);
    std::vector<float> expected(x);
    for (auto& value : expected)
        value += 0.25f;
    expectNear(builder.run({{"x", x}}, "y"), expected, 0.0f);
}

namespace {

struct Conv3dCase {
    Shape input;                  // N, C, D, H, W
    Shape weights;                // O, C, kD, kH, kW
    std::vector<int64_t> strides; // D, H, W
    std::vector<int64_t> pads;    // D, H, W begins, then ends
};

Shape conv3dOutputShape(const Conv3dCase& c) {
    Shape output{c.input[0], c.weights[0], 0, 0, 0};
    for (size_t axis = 0; axis < 3; ++axis) {
        const int64_t padded = c.input[2 + axis] + c.pads[axis] + c.pads[3 + axis];
        output[2 + axis] = static_cast<uint32_t>((padded - c.weights[2 + axis]) / c.strides[axis] + 1);
    }
    return output;
}

std::vector<float> conv3dReference(const Conv3dCase& c, const std::vector<float>& x, const std::vector<float>& w,
                                   const std::vector<float>& b) {
    const Shape out = conv3dOutputShape(c);
    std::vector<float> y(elementCount(out));
    const auto& in = c.input;
    const auto& k = c.weights;
    for (uint32_t n = 0; n < out[0]; ++n)
        for (uint32_t o = 0; o < out[1]; ++o)
            for (uint32_t d = 0; d < out[2]; ++d)
                for (uint32_t h = 0; h < out[3]; ++h)
                    for (uint32_t wo = 0; wo < out[4]; ++wo) {
                        double sum = b[o];
                        for (uint32_t ci = 0; ci < in[1]; ++ci)
                            for (uint32_t kd = 0; kd < k[2]; ++kd)
                                for (uint32_t kh = 0; kh < k[3]; ++kh)
                                    for (uint32_t kw = 0; kw < k[4]; ++kw) {
                                        const int64_t id = int64_t(d) * c.strides[0] + kd - c.pads[0];
                                        const int64_t ih = int64_t(h) * c.strides[1] + kh - c.pads[1];
                                        const int64_t iw = int64_t(wo) * c.strides[2] + kw - c.pads[2];
                                        if (id < 0 || ih < 0 || iw < 0 || id >= in[2] || ih >= in[3] || iw >= in[4])
                                            continue;
                                        sum += double(x[(((size_t(n) * in[1] + ci) * in[2] + id) * in[3] + ih) * in[4] +
                                                        iw]) *
                                               w[(((size_t(o) * k[1] + ci) * k[2] + kd) * k[3] + kh) * k[4] + kw];
                                    }
                        y[(((size_t(n) * out[1] + o) * out[2] + d) * out[3] + h) * out[4] + wo] = float(sum);
                    }
    return y;
}

void runConv3dCase(const std::string& name, const Conv3dCase& c, uint32_t seed) {
    const Shape output = conv3dOutputShape(c);
    ModelBuilder builder(name);
    builder.input("x", c.input);
    builder.output("y", output);
    const auto x = patternedValues(elementCount(c.input), seed);
    const auto w = patternedValues(elementCount(c.weights), seed + 1);
    const auto b = patternedValues(c.weights[0], seed + 2);
    builder.floatInitializer("w", c.weights, w);
    builder.floatInitializer("b", {c.weights[0]}, b);
    auto* node = builder.node("Conv", {"x", "w", "b"}, {"y"});
    ModelBuilder::ints(node, "kernel_shape", {c.weights[2], c.weights[3], c.weights[4]});
    ModelBuilder::ints(node, "strides", c.strides);
    ModelBuilder::ints(node, "pads", c.pads);
    ModelBuilder::ints(node, "dilations", {1, 1, 1});
    ModelBuilder::integer(node, "group", 1);
    expectNear(builder.run({{"x", x}}, "y"), conv3dReference(c, x, w, b), 2e-5f);
}

} // namespace

TEST(OnnxCosmos3OperationsTest, Conv3dCausalKernelAcrossTiles) {
    runConv3dCase("conv3d_causal_3x3x3", {{1, 5, 4, 9, 11}, {70, 5, 3, 3, 3}, {1, 1, 1}, {2, 1, 1, 0, 1, 1}}, 20);
}

TEST(OnnxCosmos3OperationsTest, Conv3dStridedSpatialDownsampling) {
    runConv3dCase("conv3d_stride2", {{1, 6, 2, 10, 12}, {8, 6, 1, 3, 3}, {1, 2, 2}, {0, 0, 0, 0, 1, 1}}, 30);
}

TEST(OnnxCosmos3OperationsTest, Conv3dCausalTemporalKernel) {
    runConv3dCase("conv3d_temporal", {{1, 7, 5, 4, 6}, {14, 7, 3, 1, 1}, {1, 1, 1}, {2, 0, 0, 0, 0, 0}}, 40);
}

TEST(OnnxCosmos3OperationsTest, Conv3dPointwiseWithBias) {
    runConv3dCase("conv3d_pointwise", {{1, 48, 3, 4, 4}, {48, 48, 1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0, 0, 0}}, 50);
}

TEST(OnnxCosmos3OperationsTest, WideRankThreeAttentionRunsUnfused) {
    // Wan VAE mid-block attention: [frames, tokens, channels] with one 1024-wide head.
    const uint32_t frames = 2, tokens = 5, width = 1024;
    ModelBuilder builder("attention_rank3_wide");
    builder.input("q", {frames, tokens, width});
    builder.input("kt", {frames, width, tokens});
    builder.input("v", {frames, tokens, width});
    builder.output("y", {frames, tokens, width});
    // Intermediate value_info, as the exporters write it.
    builder.output("scores", {frames, tokens, tokens});
    builder.output("probabilities", {frames, tokens, tokens});
    builder.node("MatMul", {"q", "kt"}, {"scores"});
    builder.node("Softmax", {"scores"}, {"probabilities"});
    builder.node("MatMul", {"probabilities", "v"}, {"y"});
    const auto q = patternedValues(size_t(frames) * tokens * width, 60);
    const auto kt = patternedValues(size_t(frames) * width * tokens, 61);
    const auto v = patternedValues(size_t(frames) * tokens * width, 62);
    std::vector<float> expected(size_t(frames) * tokens * width);
    for (uint32_t f = 0; f < frames; ++f) {
        for (uint32_t i = 0; i < tokens; ++i) {
            std::vector<double> scores(tokens);
            double maximum = -1e300, sum = 0.0;
            for (uint32_t j = 0; j < tokens; ++j) {
                double score = 0.0;
                for (uint32_t d = 0; d < width; ++d) {
                    score += double(q[(size_t(f) * tokens + i) * width + d]) * kt[(size_t(f) * width + d) * tokens + j];
                }
                scores[j] = score;
                maximum = std::max(maximum, score);
            }
            for (auto& score : scores)
                sum += (score = std::exp(score - maximum));
            for (uint32_t d = 0; d < width; ++d) {
                double value = 0.0;
                for (uint32_t j = 0; j < tokens; ++j)
                    value += scores[j] * v[(size_t(f) * tokens + j) * width + d];
                expected[(size_t(f) * tokens + i) * width + d] = float(value / sum);
            }
        }
    }
    expectNear(builder.run({{"q", q}, {"kt", kt}, {"v", v}}, "y"), expected, 1e-5f);
}
