#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#ifdef _WIN32
#include <crtdbg.h>
#include <windows.h>
#endif

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace {

constexpr uint32_t kNetworkWidth = 128;
constexpr uint32_t kNetworkHeight = 128;

struct ImageTensorPushConstants {
    uint32_t width;
    uint32_t height;
};

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    int maxFrames = -1;
    std::string spzPath = "./3rdparty/spz/samples/racoonfamily.spz";
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--frames" && i + 1 < argc) {
            try {
                maxFrames = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid value for --frames" << std::endl;
                return 1;
            }
        } else if (argument == "--spz" && i + 1 < argc) {
            spzPath = argv[++i];
        }
    }

    const std::string encoderPath = "./data/onnx/simple_encoder.onnx";
    const std::string decoderPath = "./data/onnx/simple_decoder.onnx";
    for (const auto& path : {spzPath, encoderPath, decoderPath}) {
        if (!std::filesystem::exists(path)) {
            std::cerr << "Required file not found: " << path << std::endl;
            return 1;
        }
    }

    std::cout << "Gaussian autoencoder example: splat -> encoder -> decoder -> display";
    if (maxFrames > 0) {
        std::cout << " (closing after " << maxFrames << " frames)";
    }
    std::cout << std::endl;

    klartraum::GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vulkanContext = engine.getVulkanContext();
    const uint32_t pathCount = vulkanContext.getNumberOfSwapChainImages();

    auto splatTarget = std::make_shared<klartraum::OffscreenTarget>(
        vulkanContext, VkExtent2D{kNetworkWidth, kNetworkHeight}, pathCount);
    const auto windowExtent = vulkanContext.getSwapChainExtent();
    auto displayTarget = engine.getWindow().makeViewport(
        0, 0, windowExtent.width, windowExtent.height,
        kNetworkWidth, kNetworkHeight);

    auto cameraUbo = std::make_shared<klartraum::CameraUboType>();
    cameraUbo->setName("CameraUBO");
    auto model = std::make_shared<klartraum::GaussianDataStandard>(vulkanContext, spzPath);
    auto splatting = klartraum::createGaussianSplatting(
        vulkanContext, klartraum::GsplatBackend::Compute,
        splatTarget, cameraUbo, model);

    auto imageTensor = vulkanContext.create<klartraum::TensorElement<float>>(
        std::vector<uint32_t>{1, 3, kNetworkHeight, kNetworkWidth});
    auto imageToTensor = vulkanContext.create<
        klartraum::GeneralComputation<ImageTensorPushConstants>>(
            "shaders/onnx/image_to_tensor.comp.spv");
    imageToTensor->setName("SplatImageToTensor");
    imageToTensor->setPushConstants({{kNetworkWidth, kNetworkHeight}});
    imageToTensor->setGroupCount(
        (kNetworkWidth + 7) / 8, (kNetworkHeight + 7) / 8, 1);
    imageToTensor->setInput(splatting, 0, 0);
    imageToTensor->setInput(imageTensor, 1);
    imageToTensor->setImageLayoutTransition(
        0,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_SHADER_READ_BIT);

    auto encoder = vulkanContext.create<klartraum::OnnxNetwork>(encoderPath);
    encoder->setName("Encoder");
    encoder->setInputTensor("input", imageToTensor, 1);

    auto decoder = vulkanContext.create<klartraum::OnnxNetwork>(decoderPath);
    decoder->setName("Decoder");
    decoder->setInputTensor("input", encoder, 0);

    auto tensorToImage = vulkanContext.create<
        klartraum::GeneralComputation<ImageTensorPushConstants>>(
            "shaders/onnx/tensor_to_image.comp.spv");
    tensorToImage->setName("DecodedTensorToImage");
    tensorToImage->setPushConstants({{kNetworkWidth, kNetworkHeight}});
    tensorToImage->setGroupCount(
        (kNetworkWidth + 7) / 8, (kNetworkHeight + 7) / 8, 1);
    tensorToImage->setInput(decoder, 0, 0);
    tensorToImage->setInput(displayTarget, 1);
    tensorToImage->setImageLayoutTransition(
        1,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, VK_ACCESS_SHADER_WRITE_BIT);

    auto readyForDisplay = std::make_shared<klartraum::ImageViewSrcTransition>(
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    readyForDisplay->setName("DecodedImageReadyForDisplay");
    readyForDisplay->setInput(tensorToImage, 0, 1);
    engine.add(readyForDisplay);

    auto camera = std::make_shared<klartraum::InterfaceCameraOrbit>(
        klartraum::InterfaceCameraOrbit::UpDirection::Y);
    camera->initialize(vulkanContext);
    camera->setProjectionAspectRatio(1.0f);
    camera->setAzimuth(0.9f);
    camera->setElevation(-0.5f);
    camera->setPosition({-0.5f, 0.0f, 0.5f});
    camera->setDistance(1.0f);
    engine.setInterfaceCamera(camera);
    engine.setCameraUBO(cameraUbo);

    frontend.loop(maxFrames);

    if (maxFrames > 0) {
        auto decoded = std::dynamic_pointer_cast<klartraum::TensorElement<float>>(
            decoder->getOutputElement("output"));
        std::vector<float> values(decoded->getDataElementCount());
        decoded->getDataBuffer(0).memcopyTo(values);
        const auto range = std::minmax_element(values.begin(), values.end());
        std::cout << "Decoded output range: [" << *range.first
                  << ", " << *range.second << "]" << std::endl;
    }
    return 0;
}
