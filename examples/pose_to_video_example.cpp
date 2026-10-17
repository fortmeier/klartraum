// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Neural character animation turned into a photorealistic video, headless.
//
// A mixture-of-experts character controller (CharacterController) walks a
// character along a scripted path. Every frame, its joints are drawn as a
// ControlNet OpenPose pose map (OpenPoseSkeletonRenderer) and converted into
// the pose tensor of a ControlNet-OpenPose + SD1.5 LCM denoiser, all on the
// GPU. TemporalLcmSampler samples the frames with shared noise as img2img
// variations of one keyframe, and the SD1.5 VAE decodes them
// (PoseVideoPipeline).
//
//   pose_to_video_example --size 512 --frames 50 --out-dir build/TestingOutput/pose_to_video
//
// Writes pose_XXXX.ppm and frame_XXXX.ppm per frame. The third-party SD1.5,
// ControlNet and LCM-LoRA models are exported by scripts/sd15_onnx
// (export_denoiser.py and export_controlnet.py); the controller comes from
// scripts/character_animation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/sd15/pose_video_pipeline.hpp"

namespace {

struct Options {
    // SD1.5 and the ControlNet were trained at 512 pixels; smaller images lose faces and backgrounds.
    uint32_t size = 512;
    uint32_t frames = 50;
    uint32_t controllerSteps = 2;
    klartraum::PoseVideoPipeline::Directories directories;
    std::filesystem::path outDirectory = "build/TestingOutput/pose_to_video";
    std::string prompt = "a photograph of a man walking down a city street, jeans and a dark jacket, empty hands, "
                         "natural daylight, photorealistic, detailed, sharp focus, 50mm lens";
    std::string style = "Neutral";
    float speed = 1.3f;
    klartraum::TemporalLcmSampler::Settings sampler;
};

// pixels: [3, size, size]; values map from [low, high] to [0, 255].
void writePpm(const std::filesystem::path& path, const std::vector<float>& pixels, uint32_t size, float low,
              float high) {
    std::ofstream output(path, std::ios::binary);
    output << "P6\n" << size << " " << size << "\n255\n";
    const size_t plane = size_t(size) * size;
    for (size_t pixel = 0; pixel < plane; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const float value = std::clamp((pixels[channel * plane + pixel] - low) / (high - low), 0.0f, 1.0f);
            const auto byte = static_cast<unsigned char>(value * 255.0f + 0.5f);
            output.write(reinterpret_cast<const char*>(&byte), 1);
        }
    }
}

// Stand for the first fifth, walk forward (+z), and from the middle on turn right towards +x.
klartraum::CharacterControl scriptedControl(uint32_t step, uint32_t steps, float speed, uint32_t style) {
    const uint32_t walkStart = steps / 5, turnStart = steps / 2;
    float heading = 0.0f;
    if (step >= turnStart) {
        heading = 0.5f * 3.14159265f *
                  std::min(1.0f, float(step - turnStart) / float(std::max<uint32_t>(1, steps - turnStart - 1)));
    }
    klartraum::CharacterControl control;
    control.facing = {std::sin(heading), std::cos(heading)};
    control.velocity = step >= walkStart ? control.facing * speed : glm::vec2(0.0f);
    control.style = style;
    return control;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + argument);
            }
            return argv[++i];
        };
        if (argument == "--size") {
            options.size = static_cast<uint32_t>(std::stoul(next()));
        } else if (argument == "--frames") {
            options.frames = static_cast<uint32_t>(std::stoul(next()));
        } else if (argument == "--controller-steps") {
            options.controllerSteps = static_cast<uint32_t>(std::stoul(next()));
        } else if (argument == "--character-dir") {
            options.directories.character = next();
        } else if (argument == "--sd-dir") {
            options.directories.sd = next();
        } else if (argument == "--control-dir") {
            options.directories.control = next();
        } else if (argument == "--out-dir") {
            options.outDirectory = next();
        } else if (argument == "--prompt") {
            options.prompt = next();
        } else if (argument == "--style") {
            options.style = next();
        } else if (argument == "--speed") {
            options.speed = std::stof(next());
        } else if (argument == "--start-step") {
            options.sampler.startStep = std::stoul(next());
        } else if (argument == "--anchor") {
            options.sampler.anchorWeight = std::stof(next());
        } else if (argument == "--independent") {
            options.sampler.independent = true;
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }
    if (options.frames == 0 || options.controllerSteps == 0) {
        throw std::runtime_error("--frames and --controller-steps must be positive");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const auto& styles = klartraum::CharacterController::kStyles;
        const auto styleIt = std::find(styles.begin(), styles.end(), options.style);
        if (styleIt == styles.end()) {
            throw std::runtime_error("Unknown --style " + options.style);
        }
        const auto style = static_cast<uint32_t>(styleIt - styles.begin());
        std::filesystem::create_directories(options.outDirectory);

        klartraum::HeadlessFrontend frontend;
        auto& vc = frontend.getKlartraumEngine().getVulkanContext();
        const auto started = std::chrono::steady_clock::now();
        klartraum::PoseVideoPipeline pipeline(vc, options.directories, options.size);
        pipeline.setPrompt(options.prompt);
        auto sampler = pipeline.makeSampler(options.sampler);

        const uint32_t controllerSteps = options.frames * options.controllerSteps;
        glm::vec3 cameraTarget(0.0f, 0.9f, 0.0f);
        std::vector<float> previous;
        double flicker = 0.0;
        for (uint32_t frame = 0; frame < options.frames; ++frame) {
            for (uint32_t step = 0; step < options.controllerSteps; ++step) {
                const uint32_t controllerStep = frame * options.controllerSteps + step;
                pipeline.stepCharacter(scriptedControl(controllerStep, controllerSteps, options.speed, style),
                                       klartraum::PoseVideoPipeline::lookAt(cameraTarget));
                // The camera follows the hips smoothly, one step behind.
                const glm::vec3 hips(pipeline.getController().readJoints()[0]);
                cameraTarget += 0.2f * (glm::vec3(hips.x, 0.9f, hips.z) - cameraTarget);
            }
            char name[32];
            std::snprintf(name, sizeof(name), "pose_%04u.ppm", frame);
            writePpm(options.outDirectory / name, pipeline.readPoseMap(), options.size, 0.0f, 1.0f);

            const auto image = pipeline.decode(pipeline.sampleFrame(sampler));
            std::snprintf(name, sizeof(name), "frame_%04u.ppm", frame);
            writePpm(options.outDirectory / name, image, options.size, -1.0f, 1.0f);
            if (!previous.empty()) {
                double sum = 0.0;
                for (size_t i = 0; i < image.size(); ++i) {
                    sum += std::abs(image[i] - previous[i]);
                }
                flicker += sum / double(image.size()) * 127.5;
            }
            previous = image;
            std::cout << "frame " << frame + 1 << "/" << options.frames << std::endl;
        }

        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "wrote " << options.frames << " frames of " << options.size << "x" << options.size << " to "
                  << options.outDirectory.string() << " in " << seconds << " s with " << pipeline.getEvaluations()
                  << " ControlNet + UNet evaluations" << std::endl;
        if (options.frames > 1) {
            std::cout << "mean absolute difference between consecutive frames: " << flicker / double(options.frames - 1)
                      << " (of 255)" << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << "pose_to_video_example: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
