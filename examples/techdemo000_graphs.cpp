// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Builds the compute graphs of the tech demo's chapters as techdemo000_example
// builds them, measures them and writes them as JSON, for the technical
// chapter's graph pictures:
//   gaussian_splatting  the scene splatted (chapter 2)
//   dynamic_lighting    the scene relit by four lights, then splatted (chapter 3)
//   neural_animation    the controller walking a skeleton of Gaussians next to the lantern, merged with the
//                       scene and splatted (chapter 4)
// The SD1.5 chain of chapter 6 is written by techdemo000_example --dump-graphs.
//
//   techdemo000_graphs [--graph NAME] [--backend raster|compute] [--frames 50]
//                      [--out-dir build/TestingOutput/techdemo000_graphs] [--file data/lantern.spz]
//
// Each graph is compiled with timestamp profiling and run for a few warm-up
// frames, then for --frames frames while the camera circles the scene as in
// the demo. It goes to <out-dir>/<name>.json (klartraum::writeGraphJson()):
// the elements with their mean GPU time, and the edges. Every element's
// commands lie between a pair of timestamps; run with the api_dump layer to
// count them (scripts/techdemo000/graph_dispatches.py). <name>.wall.json holds
// the mean times per frame, in ms, of the profiled run and of a second run
// without profiling (its timestamps cost time):
//   wall_ms        from the submission until its fence (submit_ms + wait_ms)
//   submit_ms      the vkQueueSubmit call, with the graph's per-frame updates
//                  (MoltenVK encodes the Metal commands in it)
//   wait_ms        from the submission until the fence
//   gpu_span_ms    from the first element's start to the last element's end
//   unprofiled_*   wall_ms, submit_ms and wait_ms without profiling

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/character_controller.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/graphchain.hpp"
#include "klartraum/computegraph/gaussianmerge.hpp"
#include "klartraum/computegraph/graph_export.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/skeletongaussians.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_lighting.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/scene_placement.hpp"

namespace {

using klartraum::ComputeGraphElementPtr;

constexpr VkExtent2D kExtent{1920, 1080};
constexpr uint32_t kWarmUpFrames = 5;
constexpr float kCharacterHeight = 1.75f; // metres, the controller's skeleton

// A graph to compile: its sink, its groups and what changes per frame.
struct GraphSpec {
    ComputeGraphElementPtr sink;
    std::vector<klartraum::ExportGroup> groups;
    std::function<void(uint32_t frame)> perFrame;
    std::vector<std::shared_ptr<void>> keep;
};

std::vector<ComputeGraphElementPtr> buffers(const klartraum::GaussianSoABuffers& soa) {
    std::vector<ComputeGraphElementPtr> result;
    for (const auto* ref : soa.all()) {
        result.push_back(ref->element);
    }
    return result;
}

// Per-axis median of the Gaussian centres, as the demo's orbit centre.
glm::vec3 sceneCentre(const std::vector<klartraum::Gaussian3D>& gaussians) {
    glm::vec3 centre(0.0f);
    std::vector<float> values(gaussians.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (size_t i = 0; i < gaussians.size(); ++i) {
            values[i] = gaussians[i].position[axis];
        }
        auto middle = values.begin() + values.size() / 2;
        std::nth_element(values.begin(), middle, values.end());
        centre[axis] = *middle;
    }
    return centre;
}

class Builder {
public:
    Builder(klartraum::VulkanContext& vc, const std::string& spzFile, klartraum::GsplatBackend backend)
        : vc(vc),
          backend(backend),
          gaussians(klartraum::loadGaussiansSpz(spzFile, true)),
          lantern(klartraum::measureUprightObject(gaussians, 0.04f)),
          orbit(klartraum::InterfaceCameraOrbit::UpDirection::Y) {
        // The demo's camera: around the scene centre, one turn per 15 s at 30 fps.
        const glm::vec3 centre = sceneCentre(gaussians);
        const glm::vec3 cameraPosition(0.55f, 0.48f, 0.69f);
        startAzimuth = std::atan2(cameraPosition.z, cameraPosition.x);
        orbit.initialize(vc);
        orbit.setPosition(-centre);
        orbit.setElevation(-std::asin(cameraPosition.y / glm::length(cameraPosition)));
        orbit.setDistance(0.6);
        orbit.setProjectionAspectRatio(float(kExtent.width) / float(kExtent.height));
    }

    GraphSpec build(const std::string& name) {
        if (name == "gaussian_splatting") {
            return splatting(scene()->buffers(), {}, nullptr);
        }
        if (name == "dynamic_lighting") {
            auto data = scene();
            auto values = std::make_shared<klartraum::HostValues<klartraum::LightingFrame>>(
                vc, std::vector<klartraum::LightingFrame>{klartraum::LightingFrame{}});
            values->setName("LightingFrame");
            const auto lit = klartraum::createGaussianLighting(
                vc, data->buffers(), klartraum::bakeSplatMaterials(gaussians), klartraum::BufferRef{values});
            lit.element->setName("GaussianLighting");
            // Dusk, and a red, a green and a blue light circling the lantern at half its height.
            auto lights = [this, values](uint32_t frame) {
                klartraum::LightingFrame lighting;
                lighting.camera = glm::vec4(glm::vec3(cameraUBO->ubo.cameraWorldPos), 0.0f);
                lighting.control.x = 1.0f;
                lighting.ambient = glm::vec4(0.2f, 0.22f, 0.3f, 1.0f);
                const glm::vec3 colours[3] = {{1.0f, 0.12f, 0.08f}, {0.1f, 1.0f, 0.15f}, {0.12f, 0.25f, 1.0f}};
                const float radius = 0.9f * lantern.groundRadius;
                for (int k = 0; k < 3; ++k) {
                    const float angle = 0.05f * float(frame) + float(k) * 2.0f * 3.14159f / 3.0f;
                    const glm::vec3 position = lantern.base + glm::vec3(radius * std::cos(angle), 0.5f * lantern.height,
                                                                        radius * std::sin(angle));
                    lighting.lights[k] = {glm::vec4(position, 0.0f), glm::vec4(colours[k], 2.0f),
                                          glm::vec4(radius, 0.1f * radius, 0.0f, 0.0f)};
                }
                lighting.counts.x = 3;
                values->set(0, lighting);
            };
            auto spec = splatting(lit.output, {}, lights);
            spec.keep.push_back(data);
            return spec;
        }
        if (name == "neural_animation") {
            auto data = scene();
            auto controller = std::make_shared<klartraum::CharacterController>(vc, "data/character");
            controller->setName("CharacterController");
            // Half as tall as the lantern, on the moss beside it.
            const float sceneHeight = 0.5f * lantern.height;
            const float pathRadius = lantern.footprintRadius + 0.5f * (lantern.groundRadius - lantern.footprintRadius);
            klartraum::SkeletonGaussianParams params;
            params.toScene = glm::scale(glm::translate(glm::mat4(1.0f), lantern.base + glm::vec3(pathRadius, 0, 0)),
                                        glm::vec3(sceneHeight / kCharacterHeight));
            params.colourOpacity = glm::vec4(0.95f, 0.95f, 0.95f, 1.0f);
            params.shape = glm::vec4(0.004f * lantern.height, 1.5f, 0.0f, 0.0f);
            auto values = std::make_shared<klartraum::HostValues<klartraum::SkeletonGaussianParams>>(
                vc, std::vector<klartraum::SkeletonGaussianParams>{params});
            values->setName("SkeletonParams");
            const auto skeleton =
                klartraum::createSkeletonGaussians(vc, klartraum::BufferRef{controller->getJoints()},
                                                   klartraum::characterSkeletonBones(), klartraum::BufferRef{values});
            skeleton.element->setName("SkeletonGaussians");
            skeleton.element->addDependency(controller);
            const auto merged = klartraum::createGaussianMerge(vc, data->buffers(), skeleton.output);
            merged.element->setName("GaussianMerge");
            std::vector<klartraum::ExportGroup> groups = {
                {"Character controller", controller, {}},
                {"Mixture-of-experts network",
                 controller->getNetwork(),
                 {controller->getMotionInput(), controller->getGatingInput()}},
            };
            auto walk = [controller](uint32_t) {
                klartraum::CharacterControl control;
                control.velocity = glm::vec2(0.0f, 1.2f);
                controller->setControl(control);
            };
            auto spec = splatting(merged.output, groups, walk);
            spec.keep.push_back(data);
            spec.keep.push_back(controller);
            return spec;
        }
        throw std::runtime_error("unknown graph " + name);
    }

    // The camera for frame `frame`: the orbit turning as in the demo.
    void moveCamera(uint32_t frame) {
        orbit.setAzimuth(startAzimuth + 2.0 * 3.14159265 * double(frame) / (15.0 * 30.0));
        orbit.update(cameraUBO->ubo);
        cameraUBO->update(0);
    }

private:
    std::shared_ptr<klartraum::GaussianDataStandard> scene() {
        return std::make_shared<klartraum::GaussianDataStandard>(vc, gaussians);
    }

    GraphSpec splatting(const klartraum::GaussianSoABuffers& input, std::vector<klartraum::ExportGroup> groups,
                        std::function<void(uint32_t)> perFrame) {
        auto target = std::make_shared<klartraum::OffscreenTarget>(vc, kExtent, 1u);
        target->setName("Target");
        cameraUBO = std::make_shared<klartraum::CameraUboType>();
        cameraUBO->setName("Camera");
        klartraum::GsplatConfig config;
        config.outputDepth = true;
        auto splat = klartraum::createGaussianSplatting(vc, backend, target, cameraUBO, input, config);
        auto boundary = buffers(input);
        boundary.push_back(target);
        boundary.push_back(cameraUBO);
        groups.push_back({"Gaussian splatting", splat, boundary});
        GraphSpec spec;
        spec.sink = splat;
        spec.groups = groups;
        spec.perFrame = [this, perFrame](uint32_t frame) {
            moveCamera(frame);
            if (perFrame) {
                perFrame(frame);
            }
        };
        spec.keep = {target};
        return spec;
    }

    klartraum::VulkanContext& vc;
    klartraum::GsplatBackend backend;
    std::vector<klartraum::Gaussian3D> gaussians;
    klartraum::UprightObjectBounds lantern;
    klartraum::InterfaceCameraOrbit orbit;
    double startAzimuth = 0.0;
    std::shared_ptr<klartraum::CameraUboType> cameraUBO;
};

struct Times {
    double submitMs = 0.0;
    double waitMs = 0.0;
};

/// Runs the graph for warm-up frames, then `frames` frames, and returns the mean times of the submission and of the
/// wait for its fence; the graph's profiling covers the measured frames.
Times run(klartraum::VulkanContext& vc, klartraum::ComputeGraph& graph, GraphSpec& spec, uint32_t frames) {
    klartraum::GraphChain chain(vc);
    chain.add(graph);
    Times times;
    for (uint32_t frame = 0; frame < kWarmUpFrames + frames; ++frame) {
        if (frame == kWarmUpFrames) {
            graph.resetProfiling();
        }
        spec.perFrame(frame);
        const auto start = std::chrono::steady_clock::now();
        chain.submit(vc.getGraphicsQueue());
        const auto submitted = std::chrono::steady_clock::now();
        chain.wait();
        const auto finished = std::chrono::steady_clock::now();
        graph.readProfiling();
        if (frame >= kWarmUpFrames) {
            times.submitMs += std::chrono::duration<double, std::milli>(submitted - start).count();
            times.waitMs += std::chrono::duration<double, std::milli>(finished - submitted).count();
        }
    }
    times.submitMs /= std::max(1u, frames);
    times.waitMs /= std::max(1u, frames);
    return times;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> names = {"gaussian_splatting", "dynamic_lighting", "neural_animation"};
    std::filesystem::path outDir = "build/TestingOutput/techdemo000_graphs";
    std::string spzFile = "./data/lantern.spz";
    klartraum::GsplatBackend backend = klartraum::GsplatBackend::Raster;
    uint32_t frames = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--graph" && hasValue) {
            names = {argv[++i]};
        } else if (arg == "--out-dir" && hasValue) {
            outDir = argv[++i];
        } else if (arg == "--file" && hasValue) {
            spzFile = argv[++i];
        } else if (arg == "--frames" && hasValue) {
            frames = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--backend" && hasValue) {
            const std::string value = argv[++i];
            if (value != "raster" && value != "compute") {
                std::cerr << "--backend must be raster or compute" << std::endl;
                return 1;
            }
            backend = value == "raster" ? klartraum::GsplatBackend::Raster : klartraum::GsplatBackend::Compute;
        } else {
            std::cerr << "unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }
    std::filesystem::create_directories(outDir);

    klartraum::HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    Builder builder(vc, spzFile, backend);
    for (const auto& name : names) {
        GraphSpec spec = builder.build(name);
        klartraum::ComputeGraph graph(vc, 1);
        graph.enableProfiling();
        graph.compileFrom(spec.sink);
        const Times profiled = run(vc, graph, spec, frames);
        const auto path = outDir / (name + ".json");
        klartraum::writeGraphJson(path, name, graph, spec.groups);

        // The same graph without timestamps.
        GraphSpec plainSpec = builder.build(name);
        klartraum::ComputeGraph plain(vc, 1);
        plain.compileFrom(plainSpec.sink);
        const Times unprofiled = run(vc, plain, plainSpec, frames);

        std::ofstream(outDir / (name + ".wall.json"))
            << "{\"wall_ms\": " << profiled.submitMs + profiled.waitMs << ", \"submit_ms\": " << profiled.submitMs
            << ", \"wait_ms\": " << profiled.waitMs << ", \"gpu_span_ms\": " << graph.getGpuSpanMs()
            << ", \"unprofiled_wall_ms\": " << unprofiled.submitMs + unprofiled.waitMs
            << ", \"unprofiled_submit_ms\": " << unprofiled.submitMs
            << ", \"unprofiled_wait_ms\": " << unprofiled.waitMs << "}\n";
        double total = 0.0;
        for (double time : graph.getElementTimesMs()) {
            total += time;
        }
        std::cout << name << ": " << graph.getElements().size() << " elements, " << total << " ms GPU per frame, "
                  << path.string() << std::endl;
    }
    return 0;
}
