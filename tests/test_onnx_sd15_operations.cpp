// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - Add and Mul apply ONNX right-aligned broadcasting.
 * - Sigmoid evaluates every tensor element.
 * - InstanceNormalization normalizes each N,C spatial slice.
 * - MatMul multiplies batched row-major matrices.
 * - Softmax normalizes rows along the last axis.
 * - Split writes three contiguous slices along the last axis.
 * - Slice extracts a contiguous range on an arbitrary tensor axis.
 * - Resize performs nearest-neighbor spatial upsampling.
 * - Transpose supports the rank-three permutation used by SD 1.5 attention.
 **/

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_push_constants.hpp"

using namespace klartraum;

namespace {

template <typename PushConstants>
std::vector<float> runUnary(VulkanContext& context, const char* shader, const std::vector<uint32_t>& dimensions,
                            const std::vector<float>& input, const PushConstants& pushConstants, uint32_t groups) {
    auto source = context.create<TensorElement<float>>(dimensions);
    auto destination = context.create<TensorElement<float>>(dimensions);
    auto computation = context.create<GeneralComputation<PushConstants>>(shader);
    computation->setPushConstants({pushConstants});
    computation->setGroupCount(groups, 1, 1);
    computation->setInput(source, 0);
    computation->setInput(destination, 1);
    ComputeGraph graph(context, 1);
    graph.compileFrom(computation);
    source->setData(0, input);
    graph.submitAndWait(context.getGraphicsQueue(), 0);
    std::vector<float> result(input.size());
    destination->getDataBuffer(0).memcopyTo(result);
    return result;
}

class OnnxSd15OperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        context = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* context = nullptr;
};

} // namespace

TEST_F(OnnxSd15OperationsTest, AddAndMulBroadcast) {
    auto lhs = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 2, 2});
    auto rhs = context->create<TensorElement<float>>(std::vector<uint32_t>{2, 1, 1});
    auto sum = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 2, 2});
    auto product = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 2, 2});
    BinaryBroadcastPushConstants constants{};
    constants.elementCount = 8;
    constants.rank = 4;
    constants.lhsDims[0] = 1;
    constants.lhsDims[1] = 2;
    constants.lhsDims[2] = 2;
    constants.lhsDims[3] = 2;
    constants.rhsDims[0] = 1;
    constants.rhsDims[1] = 2;
    constants.rhsDims[2] = 1;
    constants.rhsDims[3] = 1;
    constants.outputDims[0] = 1;
    constants.outputDims[1] = 2;
    constants.outputDims[2] = 2;
    constants.outputDims[3] = 2;

    auto add = context->create<GeneralComputation<BinaryBroadcastPushConstants>>("shaders/onnx/add.comp.spv");
    add->setPushConstants({constants});
    add->setGroupCount(1, 1, 1);
    add->setInput(lhs, 0);
    add->setInput(rhs, 1);
    add->setInput(sum, 2);
    auto mul = context->create<GeneralComputation<BinaryBroadcastPushConstants>>("shaders/onnx/mul.comp.spv");
    mul->setPushConstants({constants});
    mul->setGroupCount(1, 1, 1);
    mul->setInput(add, 0, 2);
    mul->setInput(rhs, 1);
    mul->setInput(product, 2);
    ComputeGraph graph(*context, 1);
    graph.compileFrom(mul);
    lhs->setData(0, {1, 2, 3, 4, 5, 6, 7, 8});
    rhs->setData(0, {10, 20});
    graph.submitAndWait(context->getGraphicsQueue(), 0);
    std::vector<float> values(8);
    product->getDataBuffer(0).memcopyTo(values);
    EXPECT_EQ(values, (std::vector<float>{110, 120, 130, 140, 500, 520, 540, 560}));
}

TEST_F(OnnxSd15OperationsTest, Sigmoid) {
    UnaryPushConstants constants{3};
    const auto result = runUnary(*context, "shaders/onnx/sigmoid.comp.spv", {3}, {-2, 0, 2}, constants, 1);
    EXPECT_NEAR(result[0], 1.0f / (1.0f + std::exp(2.0f)), 1e-6f);
    EXPECT_NEAR(result[1], 0.5f, 1e-6f);
    EXPECT_NEAR(result[2], 1.0f / (1.0f + std::exp(-2.0f)), 1e-6f);
}

TEST_F(OnnxSd15OperationsTest, InstanceNormalization) {
    auto input = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 4});
    auto scale = context->create<TensorElement<float>>(std::vector<uint32_t>{2});
    auto bias = context->create<TensorElement<float>>(std::vector<uint32_t>{2});
    auto output = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 4});
    InstanceNormalizationPushConstants constants{1, 2, 4, 1e-5f};
    auto computation = context->create<GeneralComputation<InstanceNormalizationPushConstants>>(
        "shaders/onnx/instance_normalization.comp.spv");
    computation->setPushConstants({constants});
    computation->setGroupCount(1, 1, 1);
    computation->setInput(input, 0);
    computation->setInput(scale, 1);
    computation->setInput(bias, 2);
    computation->setInput(output, 3);
    ComputeGraph graph(*context, 1);
    graph.compileFrom(computation);
    input->setData(0, {1, 2, 3, 4, 2, 2, 2, 2});
    scale->setData(0, {2, 3});
    bias->setData(0, {1, -1});
    graph.submitAndWait(context->getGraphicsQueue(), 0);
    std::vector<float> result(8);
    output->getDataBuffer(0).memcopyTo(result);
    EXPECT_NEAR(result[0], -1.683270f, 2e-5f);
    EXPECT_NEAR(result[3], 3.683270f, 2e-5f);
    for (size_t i = 4; i < 8; ++i)
        EXPECT_NEAR(result[i], -1.0f, 1e-6f);
}

TEST_F(OnnxSd15OperationsTest, MatMul) {
    auto lhs = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 3});
    auto rhs = context->create<TensorElement<float>>(std::vector<uint32_t>{3, 2});
    auto output = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 2});
    MatMulPushConstants constants{1, 2, 2, 3, 1, 1};
    auto computation = context->create<GeneralComputation<MatMulPushConstants>>("shaders/onnx/matmul.comp.spv");
    computation->setPushConstants({constants});
    computation->setGroupCount(1, 1, 1);
    computation->setInput(lhs, 0);
    computation->setInput(rhs, 1);
    computation->setInput(output, 2);
    ComputeGraph graph(*context, 1);
    graph.compileFrom(computation);
    lhs->setData(0, {1, 2, 3, 4, 5, 6});
    rhs->setData(0, {7, 8, 9, 10, 11, 12});
    graph.submitAndWait(context->getGraphicsQueue(), 0);
    std::vector<float> result(4);
    output->getDataBuffer(0).memcopyTo(result);
    EXPECT_EQ(result, (std::vector<float>{58, 64, 139, 154}));
}

TEST_F(OnnxSd15OperationsTest, Softmax) {
    SoftmaxPushConstants constants{2, 3};
    const auto result = runUnary(*context, "shaders/onnx/softmax.comp.spv", {2, 3}, {1, 2, 3, 0, 0, 0}, constants, 1);
    EXPECT_NEAR(result[0] + result[1] + result[2], 1.0f, 1e-6f);
    EXPECT_NEAR(result[0], 0.0900306f, 1e-6f);
    EXPECT_NEAR(result[3], 1.0f / 3.0f, 1e-6f);
}

TEST_F(OnnxSd15OperationsTest, SplitThree) {
    auto input = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 6});
    auto sizes = context->create<TensorElement<int64_t>>(std::vector<uint32_t>{3});
    auto a = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 1});
    auto b = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 2});
    auto c = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 3});
    SplitPushConstants constants{2, 6, 1, 1, 2};
    auto computation = context->create<GeneralComputation<SplitPushConstants>>("shaders/onnx/split3.comp.spv");
    computation->setPushConstants({constants});
    computation->setGroupCount(1, 1, 1);
    computation->setInput(input, 0);
    computation->setInput(sizes, 1);
    computation->setInput(a, 2);
    computation->setInput(b, 3);
    computation->setInput(c, 4);
    ComputeGraph graph(*context, 1);
    graph.compileFrom(computation);
    input->setData(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    sizes->setData(0, {1, 2, 3});
    graph.submitAndWait(context->getGraphicsQueue(), 0);
    std::vector<float> av(2), bv(4), cv(6);
    a->getDataBuffer(0).memcopyTo(av);
    b->getDataBuffer(0).memcopyTo(bv);
    c->getDataBuffer(0).memcopyTo(cv);
    EXPECT_EQ(av, (std::vector<float>{0, 6}));
    EXPECT_EQ(bv, (std::vector<float>{1, 2, 7, 8}));
    EXPECT_EQ(cv, (std::vector<float>{3, 4, 5, 9, 10, 11}));
}

TEST_F(OnnxSd15OperationsTest, SliceChannelRange) {
    auto input = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 4, 2});
    auto starts = context->create<TensorElement<int64_t>>(std::vector<uint32_t>{1});
    auto ends = context->create<TensorElement<int64_t>>(std::vector<uint32_t>{1});
    auto axes = context->create<TensorElement<int64_t>>(std::vector<uint32_t>{1});
    auto output = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 2});
    SlicePushConstants constants{};
    constants.rank = 3;
    constants.axis = 1;
    constants.start = 1;
    constants.elementCount = 4;
    constants.inputDims[0] = 1;
    constants.inputDims[1] = 4;
    constants.inputDims[2] = 2;
    constants.outputDims[0] = 1;
    constants.outputDims[1] = 2;
    constants.outputDims[2] = 2;
    auto computation = context->create<GeneralComputation<SlicePushConstants>>("shaders/onnx/slice.comp.spv");
    computation->setPushConstants({constants});
    computation->setGroupCount(1, 1, 1);
    computation->setInput(input, 0);
    computation->setInput(starts, 1);
    computation->setInput(ends, 2);
    computation->setInput(axes, 3);
    computation->setInput(output, 4);
    ComputeGraph graph(*context, 1);
    graph.compileFrom(computation);
    input->setData(0, {0, 1, 2, 3, 4, 5, 6, 7});
    starts->setData(0, {1});
    ends->setData(0, {3});
    axes->setData(0, {1});
    graph.submitAndWait(context->getGraphicsQueue(), 0);
    std::vector<float> result(4);
    output->getDataBuffer(0).memcopyTo(result);
    EXPECT_EQ(result, (std::vector<float>{2, 3, 4, 5}));
}

TEST_F(OnnxSd15OperationsTest, ResizeNearest) {
    auto input = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 1, 2, 2});
    auto scales = context->create<TensorElement<float>>(std::vector<uint32_t>{4});
    auto output = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 1, 4, 4});
    ResizePushConstants constants{1, 1, 2, 2, 4, 4};
    auto computation = context->create<GeneralComputation<ResizePushConstants>>("shaders/onnx/resize_nearest.comp.spv");
    computation->setPushConstants({constants});
    computation->setGroupCount(1, 1, 1);
    computation->setInput(input, 0);
    computation->setInput(scales, 1);
    computation->setInput(output, 2);
    ComputeGraph graph(*context, 1);
    graph.compileFrom(computation);
    input->setData(0, {1, 2, 3, 4});
    scales->setData(0, {1, 1, 2, 2});
    graph.submitAndWait(context->getGraphicsQueue(), 0);
    std::vector<float> result(16);
    output->getDataBuffer(0).memcopyTo(result);
    EXPECT_EQ(result, (std::vector<float>{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));
}

TEST_F(OnnxSd15OperationsTest, TransposeRankThree) {
    auto input = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 2, 3});
    auto output = context->create<TensorElement<float>>(std::vector<uint32_t>{1, 3, 2});
    TransposePushConstants constants{};
    constants.dims = 3;
    constants.dimInput[0] = 1;
    constants.dimInput[1] = 2;
    constants.dimInput[2] = 3;
    constants.dimOutput[0] = 1;
    constants.dimOutput[1] = 3;
    constants.dimOutput[2] = 2;
    constants.dimPerm[0] = 0;
    constants.dimPerm[1] = 2;
    constants.dimPerm[2] = 1;
    auto computation = context->create<GeneralComputation<TransposePushConstants>>("shaders/onnx/transpose.comp.spv");
    computation->setPushConstants({constants});
    computation->setGroupCount(1, 1, 1);
    computation->setInput(input, 0);
    computation->setInput(output, 1);
    ComputeGraph graph(*context, 1);
    graph.compileFrom(computation);
    input->setData(0, {1, 2, 3, 4, 5, 6});
    graph.submitAndWait(context->getGraphicsQueue(), 0);
    std::vector<float> result(6);
    output->getDataBuffer(0).memcopyTo(result);
    EXPECT_EQ(result, (std::vector<float>{1, 4, 2, 5, 3, 6}));
}
