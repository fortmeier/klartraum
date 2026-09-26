/**
 * TESTS:
 * - nearestUpscale: 2x2 -> 4x4 with nearest filtering repeats every pixel twice per axis
 * - bilinearUpscale: 2x1 -> 4x1 with bilinear filtering interpolates between pixel centres
 *   and clamps at the edges
 * - bilinearDownscale: 4x2 -> 2x1 averages each 2x2 block
 **/

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/imageresample.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"

using namespace klartraum;

namespace {

struct ImageSize {
    uint32_t width;
    uint32_t height;
};

class ImageResampleTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
    }

    void TearDown() override {
        frontend.reset();
    }

    VulkanContext& vc() { return frontend->getKlartraumEngine().getVulkanContext(); }

    // Uploads `planes` (1x3xHxW, values in [0, 1]) into an image of `src`
    // size, resamples it to `dst` and returns the result as 1x3xHxW.
    std::vector<float> resample(const std::vector<float>& planes, VkExtent2D src, VkExtent2D dst,
                                ResampleFilter filter) {
        auto input = vc().create<TensorElement<float>>(std::vector<uint32_t>{1, 3, src.height, src.width});
        auto srcImage = std::make_shared<OffscreenTarget>(vc(), src, 1u);
        auto upload = vc().create<GeneralComputation<ImageSize>>("shaders/onnx/tensor_to_image.comp.spv");
        upload->setPushConstants({{src.width, src.height}});
        upload->setGroupCount((src.width + 7) / 8, (src.height + 7) / 8, 1);
        upload->setInput(input, 0);
        upload->setInput(srcImage, 1);

        auto dstImage = std::make_shared<OffscreenTarget>(vc(), dst, 1u);
        auto resample = vc().create<ImageResample>(src, dst, filter);
        resample->setInput(upload, 0, 1);
        resample->setInput(dstImage, 1);

        auto output = vc().create<TensorElement<float>>(std::vector<uint32_t>{1, 3, dst.height, dst.width});
        auto download = vc().create<GeneralComputation<ImageSize>>("shaders/onnx/image_to_tensor.comp.spv");
        download->setPushConstants({{dst.width, dst.height}});
        download->setGroupCount((dst.width + 7) / 8, (dst.height + 7) / 8, 1);
        download->setInput(resample, 0, 1);
        download->setInput(output, 1);
        download->setImageLayoutTransition(0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        ComputeGraph graph(vc(), 1);
        graph.compileFrom(download);
        input->getDataBuffer(0).memcopyFrom(planes);
        graph.submitAndWait(vc().getGraphicsQueue(), 0);

        std::vector<float> result(output->getDataElementCount());
        output->getDataBuffer(0).memcopyTo(result);
        return result;
    }

    std::unique_ptr<HeadlessFrontend> frontend;
};

// One plane per channel; green and blue are copies of red scaled by 0.5 and 0.25.
std::vector<float> rgb(const std::vector<float>& red) {
    std::vector<float> planes = red;
    for (float v : red) planes.push_back(0.5f * v);
    for (float v : red) planes.push_back(0.25f * v);
    return planes;
}

// 8-bit images quantize every value.
constexpr float kTolerance = 1.0f / 255.0f + 1e-4f;

void expectPlanes(const std::vector<float>& actual, const std::vector<float>& expectedRed) {
    const std::vector<float> expected = rgb(expectedRed);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], kTolerance) << "element " << i;
    }
}

} // namespace

TEST_F(ImageResampleTest, nearestUpscale) {
    const auto result = resample(rgb({0.2f, 0.4f,
                                      0.6f, 0.8f}),
                                 {2, 2}, {4, 4}, ResampleFilter::Nearest);
    expectPlanes(result, {0.2f, 0.2f, 0.4f, 0.4f,
                          0.2f, 0.2f, 0.4f, 0.4f,
                          0.6f, 0.6f, 0.8f, 0.8f,
                          0.6f, 0.6f, 0.8f, 0.8f});
}

TEST_F(ImageResampleTest, bilinearUpscale) {
    const auto result = resample(rgb({0.0f, 1.0f}), {2, 1}, {4, 1}, ResampleFilter::Bilinear);
    expectPlanes(result, {0.0f, 0.25f, 0.75f, 1.0f});
}

TEST_F(ImageResampleTest, bilinearDownscale) {
    const auto result = resample(rgb({0.0f, 0.4f, 0.8f, 0.8f,
                                      0.4f, 0.8f, 0.0f, 0.4f}),
                                 {4, 2}, {2, 1}, ResampleFilter::Bilinear);
    expectPlanes(result, {0.4f, 0.5f});
}
