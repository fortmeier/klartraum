// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <zlib.h>

#ifdef _WIN32
#include <windows.h>
#include <crtdbg.h>
#endif

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"

namespace {
using namespace klartraum;
#ifdef NDEBUG
constexpr bool kValidation = false;
#else
constexpr bool kValidation = true;
#endif

struct Options {
    GsplatBackend backend = GsplatBackend::Raster;
    bool gsFirst = false;
    DepthState gs{}, raster{};
    GaussianDepthMode depthMode = GaussianDepthMode::BackendDefault;
    bool matrix = false;
    bool profile = true;
    VkExtent2D extent{512, 512};
    uint32_t warmup = 10, samples = 30, runs = 3;
    std::filesystem::path output = "build/TestingOutput/gaussian-depth-composition";
};

bool boolean(const std::string& value) {
    if (value == "on")
        return true;
    if (value == "off")
        return false;
    throw std::invalid_argument("expected on or off, got " + value);
}

uint32_t positive(const std::string& value) {
    size_t used = 0;
    const auto result = std::stoul(value, &used);
    if (used != value.size() || value.front() == '-' || !result || result > 4096)
        throw std::invalid_argument("expected an integer in [1,4096], got " + value);
    return uint32_t(result);
}

const char* backendName(GsplatBackend backend) { return backend == GsplatBackend::Raster ? "raster" : "compute"; }
const char* orderName(const Options& o) { return o.gsFirst ? "gs-first" : "raster-first"; }
const char* modeName(const Options& o) {
    if (o.depthMode == GaussianDepthMode::NearestOpaqueSplat)
        return "nearest";
    if (o.depthMode == GaussianDepthMode::AccumulatedOpacity)
        return "accumulated";
    return o.backend == GsplatBackend::Raster ? "nearest" : "accumulated";
}

std::string id(const Options& o) {
    return std::string(backendName(o.backend)) + "_" + orderName(o) + "_gs-t" + char('0' + o.gs.test) + "w" +
           char('0' + o.gs.write) + "_rp-t" + char('0' + o.raster.test) + "w" + char('0' + o.raster.write);
}

Options arguments(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Headless lantern and two rasterized boxes.\n"
                         "--backend raster|compute  --order raster-first|gs-first\n"
                         "--gs-test on|off  --gs-write on|off  --raster-test on|off  --raster-write on|off\n"
                         "--depth-mode default|nearest|accumulated  --profile on|off\n"
                         "--matrix  --out-dir PATH  --width N  --height N  --warmup N  --samples N  --runs N\n";
            std::exit(0);
        }
        if (arg == "--matrix") {
            o.matrix = true;
            continue;
        }
        if (i + 1 == argc)
            throw std::invalid_argument("missing value after " + arg);
        const std::string value = argv[++i];
        if (arg == "--backend") {
            if (value == "raster")
                o.backend = GsplatBackend::Raster;
            else if (value == "compute")
                o.backend = GsplatBackend::Compute;
            else
                throw std::invalid_argument("unknown backend: " + value);
        } else if (arg == "--order") {
            if (value == "gs-first")
                o.gsFirst = true;
            else if (value == "raster-first")
                o.gsFirst = false;
            else
                throw std::invalid_argument("unknown order: " + value);
        } else if (arg == "--gs-test")
            o.gs.test = boolean(value);
        else if (arg == "--gs-write")
            o.gs.write = boolean(value);
        else if (arg == "--raster-test")
            o.raster.test = boolean(value);
        else if (arg == "--raster-write")
            o.raster.write = boolean(value);
        else if (arg == "--profile")
            o.profile = boolean(value);
        else if (arg == "--depth-mode") {
            if (value == "default")
                o.depthMode = GaussianDepthMode::BackendDefault;
            else if (value == "nearest")
                o.depthMode = GaussianDepthMode::NearestOpaqueSplat;
            else if (value == "accumulated")
                o.depthMode = GaussianDepthMode::AccumulatedOpacity;
            else
                throw std::invalid_argument("unknown depth mode: " + value);
        } else if (arg == "--width")
            o.extent.width = positive(value);
        else if (arg == "--height")
            o.extent.height = positive(value);
        else if (arg == "--warmup")
            o.warmup = positive(value);
        else if (arg == "--samples")
            o.samples = positive(value);
        else if (arg == "--runs")
            o.runs = positive(value);
        else if (arg == "--out-dir")
            o.output = value;
        else
            throw std::invalid_argument("unknown option: " + arg);
    }
    return o;
}

// Two ordinary triangle meshes. Their camera and depth state are exactly those of the render pass.
class Boxes : public DrawComponent {
public:
    explicit Boxes(VkExtent2D extent)
        : extent(extent) {}
    ~Boxes() {
        if (vulkanContext) {
            vkDestroyPipeline(vulkanContext->getDevice(), pipeline, nullptr);
            vkDestroyPipelineLayout(vulkanContext->getDevice(), layout, nullptr);
        }
    }
    void initialize(VulkanContext& vc, VkRenderPass& pass, std::shared_ptr<CameraUboType> camera) override {
        DrawComponent::initialize(vc, pass, camera);
        const auto vertex =
            createShaderModule(readFile("shaders/examples/gaussian_depth_boxes.vert.spv"), vc.getDevice());
        const auto fragment =
            createShaderModule(readFile("shaders/examples/gaussian_depth_boxes.frag.spv"), vc.getDevice());
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (auto& stage : stages) {
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.pName = "main";
        }
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertex;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragment;
        VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Parameters)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        auto result = vkCreatePipelineLayout(vc.getDevice(), &layoutInfo, nullptr, &layout);
        VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport viewport{0, 0, float(extent.width), float(extent.height), 0, 1};
        VkRect2D scissor{{0, 0}, extent};
        VkPipelineViewportStateCreateInfo viewportInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportInfo.viewportCount = viewportInfo.scissorCount = 1;
        viewportInfo.pViewports = &viewport;
        viewportInfo.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_BACK_BIT;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState color{};
        color.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &color;
        auto depth = pipelineDepthState();
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vertices;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewportInfo;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &samples;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.layout = layout;
        info.renderPass = pass;
        if (result == VK_SUCCESS)
            result = vkCreateGraphicsPipelines(vc.getDevice(), VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
        vkDestroyShaderModule(vc.getDevice(), vertex, nullptr);
        vkDestroyShaderModule(vc.getDevice(), fragment, nullptr);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Boxes: failed to create graphics pipeline");
    }
    void recordCommandBuffer(VkCommandBuffer cmd, VkFramebuffer, uint32_t) override {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        // Place the boxes relative to the fixed view: orange nearer on the left, blue farther on the right.
        const auto inverseView = glm::inverse(cameraUBO->ubo.view);
        const auto right = glm::vec3(inverseView[0]);
        const auto up = glm::vec3(inverseView[1]);
        const auto towardCamera = glm::vec3(inverseView[2]);
        const std::array<glm::vec3, 2> centers{right * -0.16f + up * -0.04f + towardCamera * 0.11f,
                                               right * 0.16f + up * 0.045f - towardCamera * 0.13f};
        const std::array<glm::vec4, 2> colors{glm::vec4(0.98f, 0.33f, 0.065f, 1), glm::vec4(0.06f, 0.55f, 0.95f, 1)};
        for (size_t i = 0; i < centers.size(); ++i) {
            auto model = glm::translate(glm::mat4(1), centers[i]);
            model = glm::rotate(model, glm::radians(i ? -18.0f : 16.0f), glm::vec3(0, 1, 0));
            model = glm::scale(model, glm::vec3(0.095f, 0.15f, 0.095f));
            const Parameters parameters{cameraUBO->ubo.proj * cameraUBO->ubo.view * model, colors[i]};
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(parameters), &parameters);
            vkCmdDraw(cmd, 36, 1, 0, 0);
        }
    }

private:
    struct Parameters {
        glm::mat4 mvp;
        glm::vec4 color;
    };
    VkExtent2D extent;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

void bigEndian(std::vector<uint8_t>& bytes, uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
        bytes.push_back(uint8_t(value >> shift));
}

void chunk(std::ofstream& stream, const char* type, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> bytes;
    bigEndian(bytes, uint32_t(data.size()));
    bytes.insert(bytes.end(), type, type + 4);
    bytes.insert(bytes.end(), data.begin(), data.end());
    const auto checksum = crc32(0, bytes.data() + 4, uInt(bytes.size() - 4));
    bigEndian(bytes, uint32_t(checksum));
    stream.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

void image(VulkanContext& vc, OffscreenTarget& target, const std::filesystem::path& path) {
    const auto extent = target.extent();
    VulkanBuffer<uint8_t> staging(vc, extent.width * extent.height * 4);
    vc.submitImmediate([&](VkCommandBuffer cmd) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                             nullptr, 0, nullptr);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(cmd, target.getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.getBuffer(), 1,
                               &region);
    });
    std::vector<uint8_t> bgra(extent.width * extent.height * 4);
    staging.memcopyTo(bgra);
    std::vector<uint8_t> scanlines;
    for (uint32_t y = 0; y < extent.height; ++y) {
        scanlines.push_back(0); // PNG filter: none.
        for (uint32_t x = 0; x < extent.width; ++x) {
            const auto offset = (size_t(y) * extent.width + x) * 4;
            scanlines.insert(scanlines.end(), {bgra[offset + 2], bgra[offset + 1], bgra[offset]});
        }
    }
    uLongf size = compressBound(uLong(scanlines.size()));
    std::vector<uint8_t> compressed(size);
    if (compress2(compressed.data(), &size, scanlines.data(), uLong(scanlines.size()), Z_BEST_SPEED) != Z_OK)
        throw std::runtime_error("failed to compress PNG");
    compressed.resize(size);
    std::ofstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    const std::array<uint8_t, 8> signature{137, 80, 78, 71, 13, 10, 26, 10};
    stream.write(reinterpret_cast<const char*>(signature.data()), signature.size());
    std::vector<uint8_t> header;
    bigEndian(header, extent.width);
    bigEndian(header, extent.height);
    header.insert(header.end(), {8, 2, 0, 0, 0});
    chunk(stream, "IHDR", header);
    chunk(stream, "IDAT", compressed);
    chunk(stream, "IEND", {});
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
}

void render(VulkanContext& vc, const std::shared_ptr<GaussianDataStandard>& model, const Options& o,
            std::ofstream& summary, std::ofstream& elements, std::ofstream& runs) {
    auto target = std::make_shared<OffscreenTarget>(vc, o.extent, 1);
    target->setName("Color clear");
    auto camera = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    const glm::vec3 eye(0.55f, 0.48f, 0.69f);
    const auto distance = glm::length(eye);
    orbit.setPosition({0, 0, 0});
    orbit.setDistance(distance);
    orbit.setAzimuth(std::atan2(eye.z, eye.x));
    orbit.setElevation(-std::asin(eye.y / distance));
    orbit.setProjectionAspectRatio(float(o.extent.width) / float(o.extent.height));
    orbit.update(camera->ubo);
    auto depth = std::make_shared<DepthAttachment>(vc, o.extent, 1);
    depth->setName("Shared depth clear");
    auto pass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), o.extent);
    pass->setName("Boxes render pass");
    pass->setInput(camera, 1);
    pass->setDepthAttachment(depth, o.raster);
    pass->setLoadExisting(true);
    pass->addDrawComponent(std::make_shared<Boxes>(o.extent));
    GsplatConfig config;
    config.depthAttachment = depth;
    config.depthState = o.gs;
    config.depthMode = o.depthMode;
    std::shared_ptr<ImageViewSrc> source = target;
    if (!o.gsFirst) {
        pass->setInput(source, 0);
        source = pass;
    }
    auto splats = createGaussianSplatting(vc, o.backend, source, camera, model, config);
    ComputeGraphElementPtr output = splats;
    if (o.gsFirst) {
        pass->setInput(splats, 0);
        output = pass;
    }
    ComputeGraph graph(vc, 1);
    if (o.profile)
        graph.enableProfiling();
    graph.compileFrom(output);
    camera->update(0);
    std::vector<double> spans;
    std::vector<std::vector<double>> timings(graph.getElements().size());
    for (uint32_t run = 0; run < o.runs; ++run) {
        for (uint32_t i = 0; i < o.warmup; ++i)
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
        graph.resetProfiling();
        for (uint32_t i = 0; i < o.samples; ++i)
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
        if (o.profile) {
            spans.push_back(graph.getGpuSpanMs());
            runs << id(o) << ',' << run << ',' << spans.back() << '\n';
            const auto times = graph.getElementTimesMs();
            for (size_t i = 0; i < times.size(); ++i)
                timings[i].push_back(times[i]);
        }
    }
    image(vc, *target, o.output / (id(o) + ".png"));
    summary << id(o) << ',' << backendName(o.backend) << ',' << orderName(o) << ',' << o.gs.test << ',' << o.gs.write
            << ',' << o.raster.test << ',' << o.raster.write << ',' << modeName(o) << ',';
    if (o.profile)
        summary << median(spans) << ',' << *std::min_element(spans.begin(), spans.end()) << ','
                << *std::max_element(spans.begin(), spans.end());
    else
        summary << ",,";
    summary << '\n';
    summary.flush();
    if (o.profile) {
        const auto& nodes = graph.getElements();
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto* name = nodes[i]->getName();
            elements << id(o) << ',' << i << ',' << (name[0] ? name : nodes[i]->getType()) << ',' << median(timings[i])
                     << '\n';
        }
    }
    std::cout << "RESULT " << id(o);
    if (o.profile)
        std::cout << ' ' << median(spans) << " GPU ms";
    std::cout << std::endl;
}
} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    try {
        const auto options = arguments(argc, argv);
        std::filesystem::create_directories(options.output);
        HeadlessFrontend frontend;
        auto& vc = frontend.getKlartraumEngine().getVulkanContext();
        auto model = std::make_shared<GaussianDataStandard>(vc, "data/lantern.spz", true);
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(vc.getPhysicalDevice(), &properties);
        std::ofstream metadata(options.output / "metadata.json");
        metadata << "{\n  \"device\": " << std::quoted(properties.deviceName)
                 << ",\n  \"driver_version\": " << properties.driverVersion
                 << ",\n  \"width\": " << options.extent.width << ",\n  \"height\": " << options.extent.height
                 << ",\n  \"gaussians\": " << model->count() << ",\n  \"warmup_per_run\": " << options.warmup
                 << ",\n  \"samples_per_run\": " << options.samples << ",\n  \"runs\": " << options.runs
                 << ",\n  \"profiling\": " << (options.profile ? "true" : "false")
                 << ",\n  \"validation\": " << (kValidation ? "true" : "false") << "\n}\n";
        std::ofstream summary(options.output / "timings.csv"), elements(options.output / "elements.csv"),
            runs(options.output / "runs.csv");
        for (auto* stream : {&metadata, &summary, &elements, &runs}) {
            stream->exceptions(std::ios::badbit | std::ios::failbit);
            stream->imbue(std::locale::classic());
            *stream << std::setprecision(9);
        }
        summary
            << "case,backend,order,gs_test,gs_write,raster_test,raster_write,depth_mode,gpu_ms,min_run_ms,max_run_ms\n";
        elements << "case,index,element,gpu_ms\n";
        runs << "case,run,gpu_ms\n";
        std::vector<Options> cases;
        if (options.matrix) {
            for (auto backend : {GsplatBackend::Raster, GsplatBackend::Compute})
                for (bool first : {false, true})
                    for (int flags = 0; flags < 16; ++flags) {
                        auto current = options;
                        current.backend = backend;
                        current.gsFirst = first;
                        current.gs = {bool(flags & 1), bool(flags & 2)};
                        current.raster = {bool(flags & 4), bool(flags & 8)};
                        cases.push_back(current);
                    }
            std::mt19937 generator(7);
            std::shuffle(cases.begin(), cases.end(), generator);
        } else
            cases.push_back(options);
        for (const auto& current : cases)
            render(vc, model, current, summary, elements, runs);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gaussian_depth_composition_example: " << e.what() << '\n';
        return 1;
    }
}
