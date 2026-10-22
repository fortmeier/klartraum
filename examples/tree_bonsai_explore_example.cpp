// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Walks or flies through the bonsai landscape in a window, with an ego camera:
//
//   tree_bonsai_explore_example [--renderer gaussian|classical] [--scale 1] [--seed 7] [--season 2]
//                               [--width 1280] [--height 720] [--render-scale 0.5] [--start forest|pot]
//                               [--speed M] [--frames N]
//
// Mouse: hold the left button and drag to look around, or press Tab to capture
// the mouse and look without a button. W, A, S, D move; Shift moves ten times
// faster and the scroll wheel scales the speed. F switches between walking
// (1.7 m above the ground) and flying, where Space or E rise and Q sinks. R
// goes back to the start; Esc releases the mouse, and quits when it is free.
// The window's title shows the frame rate, the mode, the speed and the height.
// The landscape is rendered at --render-scale times the window's pixels (on a
// high-density display twice its size) and scaled up into it.
// With --start forest (the default at scales from 0.6 on) the walk starts
// where tree_bonsai_example's fly-in ends, deep in a forest; with --start pot
// the camera floats outside the pot. --frames N closes the window after N
// frames.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

#include "klartraum/computegraph/imageresample.hpp"
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/bonsai_flight.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"

using namespace klartraum;

namespace {

// Where the camera is and where it looks: yaw in radians from -z towards +x, pitch up from the horizon.
struct Ego {
    glm::vec3 eye{0.0f};
    float yaw = 0.0f;
    float pitch = 0.0f;
    bool walking = true;

    glm::vec3 forward() const {
        return glm::vec3(std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch));
    }
};

// The keys held down that move the camera.
struct Held {
    bool forward = false, back = false, left = false, right = false, up = false, down = false, fast = false;
};

} // namespace

int main(int argc, char** argv) {
    std::string rendererName = "gaussian";
    std::string startName;
    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 1.0f;
    float season = 2.0f;
    uint32_t width = 1280, height = 720;
    float metersPerSecond = 0.0f; // 0: 4 m/s in the forest, 400 m/s outside the pot
    float renderScale = 0.5f;
    int maxFrames = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--renderer" && hasValue) {
            rendererName = argv[++i];
        } else if (arg == "--scale" && hasValue) {
            sceneConfig.scale.s = std::stof(argv[++i]);
        } else if (arg == "--seed" && hasValue) {
            sceneConfig.seed = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--season" && hasValue) {
            season = std::stof(argv[++i]);
        } else if (arg == "--width" && hasValue) {
            width = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--height" && hasValue) {
            height = uint32_t(std::stoul(argv[++i]));
        } else if (arg == "--render-scale" && hasValue) {
            renderScale = std::clamp(std::stof(argv[++i]), 0.1f, 1.0f);
        } else if (arg == "--start" && hasValue) {
            startName = argv[++i];
        } else if (arg == "--speed" && hasValue) {
            metersPerSecond = std::stof(argv[++i]);
        } else if (arg == "--frames" && hasValue) {
            maxFrames = std::stoi(argv[++i]);
        } else {
            std::cerr << "unknown option " << arg << std::endl;
            return 1;
        }
    }
    if (rendererName != "gaussian" && rendererName != "classical") {
        std::cerr << "--renderer must be gaussian or classical" << std::endl;
        return 1;
    }

    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const float m = 1.0f / scene.config.scale.metersPerUnit();
    if (startName.empty()) {
        startName = scene.placed ? "forest" : "pot";
    }
    if (startName != "forest" && startName != "pot") {
        std::cerr << "--start must be forest or pot" << std::endl;
        return 1;
    }

    // The start: deep in the forest where the fly-in ends, walking; or outside the pot looking at it, flying.
    Ego start;
    if (startName == "forest") {
        if (!scene.placed) {
            std::cerr << "--start forest needs a landscape (--scale 0.6 or more)" << std::endl;
            return 1;
        }
        std::cout << "finding a place in the forest ..." << std::endl;
        ForestApproachConfig flightConfig;
        flightConfig.season = season;
        const std::vector<ForestApproach> approaches = findForestApproach(scene, flightConfig);
        if (approaches.empty()) {
            std::cerr << "no forest found; try --start pot" << std::endl;
            return 1;
        }
        const ForestFlight flight(scene, scene.focus + glm::vec3(0.0f, 4.0f, 0.0f), scene.focus, approaches,
                                  flightConfig);
        const FlyInPose end = flight.isValid() ? flight.at(0.85f) : FlyInPose{};
        const glm::vec2 place = flight.isValid() ? glm::vec2(end.eye.x, end.eye.z) : approaches.front().end;
        start.eye = glm::vec3(place.x, scene.soilHeight(place.x, place.y) + 1.7f * m, place.y);
        const glm::vec3 look = flight.isValid() ? end.lookAt - end.eye : glm::vec3(0.0f, 0.0f, -1.0f);
        start.yaw = flight.isValid() ? std::atan2(look.x, -look.z) : approaches.front().heading;
        start.walking = true;
    } else {
        const float azimuth = 0.35f, elevation = 0.32f, distance = 4.2f;
        start.eye = scene.focus + distance * glm::vec3(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                                                       std::cos(azimuth) * std::cos(elevation));
        const glm::vec3 look = glm::normalize(scene.focus - start.eye);
        start.yaw = std::atan2(look.x, -look.z);
        start.pitch = std::asin(look.y);
        start.walking = false;
    }

    GlfwFrontend frontend;
    KlartraumEngine& engine = frontend.getKlartraumEngine();
    VulkanContext& vc = engine.getVulkanContext();
    GLFWwindow* window = frontend.getGlfwWindow();
    // The window at the asked size before the renderer is built for its swapchain.
    glfwSetWindowSize(window, int(width), int(height));
    glfwPollEvents();
    vc.recreateSwapChain();

    BonsaiRenderConfig config;
    config.mode = rendererName == "gaussian" ? BonsaiRenderMode::Gaussian : BonsaiRenderMode::Classical;
    auto renderer = std::make_shared<std::unique_ptr<BonsaiRenderer>>();
    auto extent = std::make_shared<VkExtent2D>();
    // Everything tied to the swapchain: built now and again after each resize of the window.
    engine.setGraphBuilder([&scene, config, season, renderScale, renderer, extent](KlartraumEngine& e) {
        VulkanContext& context = e.getVulkanContext();
        const uint32_t images = context.getNumberOfSwapChainImages();
        std::vector<VkImageView> views(images);
        std::vector<VkImage> swapImages(images);
        std::vector<VkExtent2D> extents(images, context.getSwapChainExtent());
        for (uint32_t i = 0; i < images; ++i) {
            views[i] = context.getImageView(i);
            swapImages[i] = context.getSwapChainImage(i);
        }
        auto target = std::make_shared<ImageViewSrc>(views, swapImages, extents);
        for (uint32_t i = 0; i < images; ++i) {
            target->setWaitFor(i, context.imageAvailableSemaphoresPerImage[i]);
        }
        const VkExtent2D window = context.getSwapChainExtent();
        *extent = {std::max(1u, uint32_t(float(window.width) * renderScale)),
                   std::max(1u, uint32_t(float(window.height) * renderScale))};
        std::cout << "building the renderer for " << extent->width << " x " << extent->height << " ..." << std::endl;
        renderer->reset();
        BonsaiRenderConfig c = config;
        c.extent = *extent;
        c.numberPaths = images;
        // Rendered offscreen at the smaller size and scaled up into the window.
        auto offscreen = std::make_shared<OffscreenTarget>(context, *extent, images);
        *renderer = std::make_unique<BonsaiRenderer>(context, scene, offscreen, c);
        auto resample = std::make_shared<ImageResample>(context, *extent, window, ResampleFilter::Bilinear,
                                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        resample->setInput((*renderer)->getOutput(), 0, 0);
        resample->setInput(target, 1);
        auto present = std::make_shared<ImageToFinalLayout>();
        present->setInput(resample, 0, 1);
        e.add(present);
        (*renderer)->initialize(images);
        (*renderer)->setSeason(season);
    });

    if (metersPerSecond <= 0.0f) {
        metersPerSecond = startName == "forest" ? 4.0f : 400.0f;
    }
    Ego ego = start;
    Held held;
    bool looking = false, captured = false;
    const auto began = std::chrono::steady_clock::now();
    auto last = began;
    auto titled = began;
    int framesSinceTitle = 0;
    engine.setFrameUpdate([&](uint32_t path) {
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.1f);
        last = now;
        const float seconds = std::chrono::duration<float>(now - began).count();

        // The input since the last frame.
        auto& events = engine.getEventQueue();
        while (!events.empty()) {
            const std::unique_ptr<Event> event = std::move(events.front());
            events.pop();
            if (const auto* key = dynamic_cast<EventKey*>(event.get())) {
                const bool down = key->action == EventKey::Action::Press;
                switch (key->key) {
                case EventKey::Key::W:
                    held.forward = down;
                    break;
                case EventKey::Key::S:
                    held.back = down;
                    break;
                case EventKey::Key::A:
                    held.left = down;
                    break;
                case EventKey::Key::D:
                    held.right = down;
                    break;
                case EventKey::Key::Space:
                case EventKey::Key::E:
                    held.up = down;
                    break;
                case EventKey::Key::Q:
                    held.down = down;
                    break;
                case EventKey::Key::LeftShift:
                case EventKey::Key::RightShift:
                    held.fast = down;
                    break;
                case EventKey::Key::F:
                    if (down) {
                        ego.walking = !ego.walking;
                    }
                    break;
                case EventKey::Key::R:
                    if (down) {
                        ego = start;
                    }
                    break;
                case EventKey::Key::Tab:
                    if (down) {
                        captured = !captured;
                        glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
                    }
                    break;
                case EventKey::Key::Escape:
                    if (down) {
                        if (captured) {
                            captured = false;
                            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                        } else {
                            glfwSetWindowShouldClose(window, GLFW_TRUE);
                        }
                    }
                    break;
                default:
                    break;
                }
            } else if (const auto* button = dynamic_cast<EventMouseButton*>(event.get())) {
                if (button->button == EventMouseButton::Button::Left) {
                    looking = button->action == EventMouseButton::Action::Press;
                }
            } else if (const auto* move = dynamic_cast<EventMouseMove*>(event.get())) {
                if (looking || captured) {
                    ego.yaw += 0.003f * float(move->dx);
                    ego.pitch =
                        std::clamp(ego.pitch - 0.003f * float(move->dy), glm::radians(-85.0f), glm::radians(85.0f));
                }
            } else if (const auto* scroll = dynamic_cast<EventMouseScroll*>(event.get())) {
                metersPerSecond = std::clamp(metersPerSecond * std::pow(1.25f, float(scroll->y)), 0.25f, 5000.0f);
            }
        }

        // Moving: along the ground when walking, along the view when flying.
        const glm::vec3 flat = glm::normalize(glm::vec3(std::sin(ego.yaw), 0.0f, -std::cos(ego.yaw)));
        const glm::vec3 side(std::cos(ego.yaw), 0.0f, std::sin(ego.yaw));
        const glm::vec3 ahead = ego.walking ? flat : ego.forward();
        glm::vec3 direction = float(held.forward - held.back) * ahead + float(held.right - held.left) * side;
        if (!ego.walking) {
            direction += float(held.up - held.down) * glm::vec3(0.0f, 1.0f, 0.0f);
        }
        if (glm::length(direction) > 1e-6f) {
            const float speed = metersPerSecond * (held.fast ? 10.0f : 1.0f) * m;
            const glm::vec3 next = ego.eye + speed * dt * glm::normalize(direction);
            // Within the pot.
            if (tree::treeTerrainRadius(scene.terrain, next.x, next.z) < 0.97f) {
                ego.eye = next;
            }
        }
        const float soil = scene.soilHeight(ego.eye.x, ego.eye.z);
        ego.eye.y = ego.walking ? soil + 1.7f * m : std::max(ego.eye.y, soil + 0.3f * m);

        BonsaiRenderer& r = **renderer;
        CameraMVP& ubo = r.getCamera()->ubo;
        const float nearPlane = std::clamp(std::min(0.5f * (ego.eye.y - soil), 0.3f * m), 1e-5f, 0.05f);
        ubo.model = glm::mat4(1.0f);
        ubo.view = glm::lookAt(ego.eye, ego.eye + ego.forward(), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj =
            glm::perspective(glm::radians(50.0f), float(extent->width) / float(extent->height), nearPlane, 50.0f);
        ubo.proj[1][1] *= -1.0f;
        ubo.cameraWorldPos = glm::vec4(ego.eye, 1.0f);
        r.getCamera()->update(path);
        r.setView(ego.eye);
        r.setTime(seconds, 0.5f, season, glm::vec4(glm::normalize(glm::vec3(1.0f, 0.0f, 0.35f)), 0.5f));

        ++framesSinceTitle;
        const float sinceTitle = std::chrono::duration<float>(now - titled).count();
        if (sinceTitle > 0.5f) {
            char title[160];
            std::snprintf(title, sizeof(title), "bonsai: %.1f fps, %s, %.1f m/s, %.1f m above the ground",
                          float(framesSinceTitle) / sinceTitle, ego.walking ? "walking" : "flying",
                          metersPerSecond * (held.fast ? 10.0f : 1.0f), (ego.eye.y - soil) / m);
            glfwSetWindowTitle(window, title);
            std::printf("%s\n", title);
            titled = now;
            framesSinceTitle = 0;
        }
    });

    frontend.loop(maxFrames);
    vc.queueWaitIdle(vc.getGraphicsQueue());
    engine.setFrameUpdate(nullptr);
    engine.setGraphBuilder(nullptr);
    renderer->reset();
    return 0;
}
