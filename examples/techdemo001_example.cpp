// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Renders "klartraum tech demo 001", a promo video of the differentiable rendering and the procedural trees,
// headlessly. Chapters, one continuous camera:
//   0. Title and morph: on white, the black "klartraum" logo (Gaussians in Unbounded) and a large black sphere above
//   it,
//      which at once morphs into the pumpkin by differentiable mesh optimization (raster::MeshMorph): every frame
//      splats the captured pumpkin into offscreen pictures, turns them into targets and runs optimization steps
//      whose gradients flow from the pictures into the mesh's shape and texture, all in the frame's compute graph.
//      The jack-o'-lantern faces the camera.
//   1. Drop: the pumpkin falls onto the splatted bonsai pot with its soil and the lantern, shrinking to its size on
//      the soil and turning to face the camera; the camera follows.
//   2. Growth: the bonsai pine grows from a seedling, its trunk extending and branching out step by step.
//   3. Days: the summer afternoon turns into night, then four days, one season each; the background is white by
//      day and black by night, with a black sun by day and a white crescent moon by night where the light shines
//      from.
// Each chapter shows a caption with the work it builds on, lower left.
//
//   techdemo001_example [--pot FILE] [--pumpkin FILE] [--width 1280] [--height 720] [--fps 30]
//                       [--from 0] [--to 3] [--step 1] [--out-dir build/TestingOutput/techdemo001]
//
// --pot and --pumpkin are the object.spz files of splat_training_example for the bonsai pot and the pumpkin
// (defaults: build/TestingOutput/splat_training/{bonsai,pumpkin_closed_1}/object.spz). Frames go to
// <out-dir>/frames/frame_XXXX.ppm, numbered through the whole video; --from/--to select chapters, --step renders
// every n-th frame for previews.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/demo_sky_marks.hpp"
#include "klartraum/demo_timeline.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/raster/mesh_morph.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/rendering/text_gaussians.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_sky.hpp"

namespace {

using namespace klartraum;

constexpr float kFov = 40.0f; // vertical field of view of every camera, degrees

struct Chapter {
    float seconds;
    std::vector<std::string> caption; // first line large, the others small
};

const std::vector<Chapter> kChapters = {
    {5.0f, {"Differentiable rasterization", "Laine et al. 2020, automatic differentiation in Slang"}},
    {2.5f, {"Gaussian splatting", "Kerbl et al. 2023, trained on 38 photographs"}},
    {5.0f, {"GPU tree generation", "Kuth et al. 2025, after Weber and Penn 1995"}},
    {18.0f, {"Days and seasons", "one compute graph submission per frame"}},
};

// The frame's pixels as BGRA bytes, from an image in `layout`.
std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, layout, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return pixels;
}

void writePpm(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        file.write(rgb, 3);
    }
}

// A camera pose: where it stands and what it looks at.
struct Pose {
    glm::vec3 eye, target;
};

Pose mix(const Pose& a, const Pose& b, float t) { return {glm::mix(a.eye, b.eye, t), glm::mix(a.target, b.target, t)}; }

CameraMVP cameraOf(const Pose& pose, float aspect) {
    CameraMVP ubo{};
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(pose.eye, pose.target, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(kFov), aspect, 0.05f, 60.0f);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(pose.eye, 1.0f);
    return ubo;
}

float smooth(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// The background by day and night: white while the sun is up, black in the night, smooth in the twilight.
float backgroundAt(float timeOfDay, float season) {
    const float elevation = treeSkyAt(timeOfDay, season).sunElevation;
    return smooth((elevation + 0.2f) / 0.6f);
}

// A caption: its lines rasterized, drawn lower left in the colour opposite the background.
struct Caption {
    std::vector<CoverageMask> lines;
};

Caption makeCaption(const Chapter& chapter, const std::shared_ptr<Font>& title, const std::shared_ptr<Font>& mono,
                    uint32_t height) {
    Caption caption;
    for (size_t i = 0; i < chapter.caption.size(); ++i) {
        TitleLayout layout;
        layout.align = TextAlign::Left;
        layout.pixelsPerUnit = float(height);
        layout.lines.push_back(
            {chapter.caption[i], i == 0 ? title : mono, i == 0 ? 500.0f : 400.0f, i == 0 ? 0.034f : 0.021f, 0.0f});
        caption.lines.push_back(rasterizeTitle(layout));
    }
    return caption;
}

void drawCaption(std::vector<uint8_t>& frame, VkExtent2D extent, const Caption& caption, float opacity, float ink) {
    if (opacity <= 0.0f)
        return;
    int y = int(extent.height) - int(0.05f * float(extent.height));
    for (auto line = caption.lines.rbegin(); line != caption.lines.rend(); ++line) {
        y -= int(line->height);
        blendCaption(frame, extent.width, extent.height, *line, int(0.04f * float(extent.width)), y, glm::vec3(ink),
                     opacity);
        y -= int(0.01f * float(extent.height));
    }
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path potPath = "build/TestingOutput/splat_training/bonsai/object.spz";
    std::filesystem::path pumpkinPath = "build/TestingOutput/splat_training/pumpkin_closed_1/object.spz";
    std::filesystem::path outDir = "build/TestingOutput/techdemo001";
    VkExtent2D extent{1280, 720};
    float fps = 30.0f;
    int fromChapter = 0, toChapter = int(kChapters.size()) - 1;
    uint32_t step = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;
        if (arg == "--pot" && hasValue)
            potPath = argv[++i];
        else if (arg == "--pumpkin" && hasValue)
            pumpkinPath = argv[++i];
        else if (arg == "--width" && hasValue)
            extent.width = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--height" && hasValue)
            extent.height = uint32_t(std::stoul(argv[++i]));
        else if (arg == "--fps" && hasValue)
            fps = std::stof(argv[++i]);
        else if (arg == "--from" && hasValue)
            fromChapter = std::stoi(argv[++i]);
        else if (arg == "--to" && hasValue)
            toChapter = std::stoi(argv[++i]);
        else if (arg == "--step" && hasValue)
            step = std::max(1u, uint32_t(std::stoul(argv[++i])));
        else if (arg == "--out-dir" && hasValue)
            outDir = argv[++i];
    }
    std::filesystem::create_directories(outDir / "frames");
    const float aspect = float(extent.width) / float(extent.height);

    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();

    DemoTimeline timeline;
    timeline.fps = fps;
    for (const Chapter& chapter : kChapters)
        timeline.chapterSeconds.push_back(chapter.seconds);
    std::vector<Caption> captions;
    const auto unbounded = std::make_shared<Font>("fonts/Unbounded.ttf");
    const auto mono = std::make_shared<Font>("fonts/JetBrainsMono.ttf");
    for (const Chapter& chapter : kChapters)
        captions.push_back(makeCaption(chapter, unbounded, mono, extent.height));
    const auto frameName = [&](uint32_t frame) {
        std::ostringstream name;
        name << "frame_" << std::setw(4) << std::setfill('0') << frame << ".ppm";
        return outDir / "frames" / name.str();
    };
    // The sun as a black symbol by day and the moon as a white crescent by night, where they shine from, behind the
    // scene; each fades with the background and as it nears the horizon.
    const auto drawSky = [&](std::vector<uint8_t>& bgra, const glm::mat4& view, const TreeSky& sky, float background) {
        const float radius = 0.028f * float(extent.height);
        const SkyMark sun = skyMarkAt(view, sky.sunDirection, extent.width, extent.height);
        if (sun.visible)
            drawSunMark(bgra, extent.width, extent.height, sun.pixel, radius, glm::vec3(0.0f),
                        sun.opacity * background * smooth(sky.sunElevation / 0.08f), background);
        const SkyMark moon = skyMarkAt(view, sky.moonDirection, extent.width, extent.height);
        if (moon.visible)
            drawMoonMark(bgra, extent.width, extent.height, moon.pixel, radius,
                         litSideTowards(view, sky.moonDirection, sky.sunDirection), glm::vec3(1.0f),
                         moon.opacity * (1.0f - background) * smooth(sky.moonElevation / 0.08f), background);
    };
    const auto finishFrame = [&](uint32_t frame, std::vector<uint8_t> bgra, float background) {
        const int chapter = timeline.chapterAt(frame);
        const float opacity = timeline.captionOpacity({chapter, 0.6f, 0.4f, 0.5f}, frame);
        drawCaption(bgra, extent, captions[chapter], opacity, 1.0f - background);
        writePpm(frameName(frame), bgra, extent);
    };

    // The bonsai: the splatted pot with soil, lantern and a bonsai pine.
    BonsaiSceneConfig sceneConfig;
    sceneConfig.treeCount = 1;
    sceneConfig.species = BonsaiSpecies::Pine;
    sceneConfig.style = 1.0f;
    sceneConfig.pot.soilMound = 0.4f;
    sceneConfig.treeSize = 1.25f;
    sceneConfig.treeSpread = 0.65f;
    sceneConfig.potObject = std::make_shared<BonsaiObject>(loadBonsaiObject(potPath.string()));
    const BonsaiScene sceneWithoutPumpkin = makeBonsaiScene(sceneConfig);

    // The stage: the title hangs high above the pot, out of the pot's view; the camera comes down with the pumpkin.
    const glm::vec3 title = sceneWithoutPumpkin.focus + glm::vec3(0.0f, 5.0f, 0.0f);
    // Framed between the logo and the pumpkin above it, nearly level with the pumpkin.
    const Pose titlePose{title + glm::vec3(0.0f, 0.7f, 7.0f), title + glm::vec3(0.0f, 0.7f, 0.0f)};
    const float orbitElevation = 0.36f, orbitDistance = 7.6f, orbitStart = 0.35f;
    const auto orbitPose = [&](float angle) {
        // A little below the trees' focus, so that the pot stands in the frame.
        const glm::vec3 focus = sceneWithoutPumpkin.focus - glm::vec3(0.0f, 0.45f, 0.0f);
        return Pose{focus + orbitDistance * glm::vec3(std::cos(orbitElevation) * std::sin(angle),
                                                      std::sin(orbitElevation),
                                                      std::cos(orbitElevation) * std::cos(angle)),
                    focus};
    };
    const float morphScale = 0.35f; // the pumpkin's unit sphere in world units, at rest on the soil
    const float titleScale = 0.7f;  // the pumpkin's unit sphere in world units, above the title
    const glm::vec3 sphereCentre = title + glm::vec3(0.0f, 1.45f, 0.0f);
    // The rotation about +y that turns the jack-o'-lantern's face (-z in its own frame) towards @p eye.
    const auto facing = [](glm::vec3 at, glm::vec3 eye) { return std::atan2(at.x - eye.x, at.z - eye.z); };
    // The pumpkin turned by `yaw` about +y and tipped by `tilt` towards its face (its top towards -z), e.g. towards a
    // camera below it, so that its unseen underside stays out of view.
    const auto pumpkinModel = [](glm::vec3 at, float yaw, float scale, float tilt = 0.0f) {
        return glm::translate(glm::mat4(1.0f), at) * glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f)) *
               glm::rotate(glm::mat4(1.0f), -tilt, glm::vec3(1.0f, 0.0f, 0.0f)) *
               glm::scale(glm::mat4(1.0f), glm::vec3(scale));
    };
    const float titleTilt = 0.45f; // radians, above the title; upright on the soil, which the camera sees from above
    const glm::vec2 landing(-0.75f, 0.35f);
    const float secondsPerTurn = 40.0f;
    const auto growthOrbit = [&](float seconds) { return orbitStart + 2.0f * 3.14159265f * seconds / secondsPerTurn; };

    // The pumpkin's Gaussians: the targets of the morph.
    auto pumpkin = std::make_shared<GaussianDataStandard>(vc, loadGaussiansSpz(pumpkinPath.string(), false));
    raster::MeshMorphOptions morphOptions;
    for (const auto& [elevation, start] : {std::pair{-0.7f, 0.39f}, {-0.2f, 0.0f}, {0.3f, 0.39f}, {0.85f, 0.0f}})
        for (int k = 0; k < 4; ++k) {
            const float angle = start + float(k) * 1.5708f;
            const glm::vec3 eye = 3.5f * glm::vec3(std::cos(elevation) * std::cos(angle), std::sin(elevation),
                                                   std::cos(elevation) * std::sin(angle));
            CameraMVP view = cameraOf({eye, glm::vec3(0.0f)}, 1.0f);
            view.proj = raster::perspective(0.65f, 1.0f, 0.1f, 20.0f);
            morphOptions.views.push_back(view);
        }
    // More views of the carved face (towards -z), so that it comes out sharp.
    for (const auto& [elevation, angle] : {std::pair{0.05f, 0.0f}, {0.4f, 0.0f}, {0.2f, -0.45f}, {0.2f, 0.45f}}) {
        const glm::vec3 eye = 3.5f * glm::vec3(std::cos(elevation) * std::sin(angle), std::sin(elevation),
                                               -std::cos(elevation) * std::cos(angle));
        CameraMVP view = cameraOf({eye, glm::vec3(0.0f)}, 1.0f);
        view.proj = raster::perspective(0.65f, 1.0f, 0.1f, 20.0f);
        morphOptions.views.push_back(view);
    }
    morphOptions.training.stepsPerSubmit = 12;
    // Fine enough for the carved face.
    morphOptions.training.width = morphOptions.training.height = 256;
    morphOptions.training.textureWidth = 256;
    morphOptions.training.textureHeight = 128;
    morphOptions.frame = extent;

    // The trained pumpkin, carried from chapter to chapter.
    raster::TriangleMesh pumpkinMesh = raster::icosphere(4);
    std::vector<float> pumpkinTexture(
        4 * size_t(morphOptions.training.textureWidth) * morphOptions.training.textureHeight, 0.0f);
    for (size_t i = 3; i < pumpkinTexture.size(); i += 4)
        pumpkinTexture[i] = 1.0f; // black

    const auto chapterRange = [&](int chapter) {
        return std::pair{timeline.chapterStart(chapter), chapter + 1 < int(kChapters.size())
                                                             ? timeline.chapterStart(chapter + 1)
                                                             : timeline.frameCount()};
    };
    std::cout << "tech demo 001: " << timeline.frameCount() << " frames at " << fps << " fps" << std::endl;

    // The trees at `age`: from nothing (0) through a seedling branching out step by step to fully grown (1).
    const auto setTrees = [](BonsaiRenderer& bonsai, const BonsaiScene& scene, float age) {
        std::vector<TreeInstance> trees = scene.trees;
        for (TreeInstance& tree : trees)
            setTreeAge(tree, age);
        bonsai.getGenerator().getInstances().set(trees);
    };
    // The light volume reaches up to the pumpkin above the title, so that it is lit and shadowed all the way down.
    const float lightCeiling = sphereCentre.y + titleScale + 0.3f;

    // Chapter 0: the title and the morph, on white.
    if (fromChapter <= 0 && toChapter >= 0) {
        TitleLayout layout;
        const float size = 3.2f / measureLine({"klartraum", unbounded, 500.0f, 1.0f, 0.0f});
        layout.lines.push_back({"klartraum", unbounded, 500.0f, size, 0.0f});
        layout.lines.push_back({"tech demo 001", mono, 400.0f, 0.4f * size, 0.05f});
        layout.pixelsPerUnit = 0.5f * float(extent.height) / (7.0f * std::tan(glm::radians(0.5f * kFov))) * 2.0f;
        TextSplatParams params;
        params.color = glm::vec3(0.0f);
        params.spacing = 0.006f;
        std::vector<Gaussian3D> text = sampleTextGaussians(rasterizeTitle(layout), params);
        for (Gaussian3D& g : text) {
            g.position[0] += title.x;
            g.position[1] += title.y - 0.45f;
            g.position[2] += title.z;
        }
        auto textData = std::make_shared<GaussianDataStandard>(vc, text);
        std::vector<glm::vec4> textColours;
        for (const Gaussian3D& g : text)
            textColours.emplace_back(g.color[0], g.color[1], g.color[2], g.alpha);

        auto frame = std::make_shared<OffscreenTarget>(vc, extent, 1);
        frame->setClearColor({{1.0f, 1.0f, 1.0f, 1.0f}});
        auto camera = std::make_shared<CameraUboType>();
        auto splatting = createGaussianSplatting(vc, GsplatBackend::Raster, frame, camera, textData->buffers());
        // The bonsai's light, high up to the title, lights the morph from the first frame on as it will be lit
        // later; the bonsai itself is not drawn (only its light volume is in the graph). The shadows come from the
        // sphere the morph starts from, shrunk to lie within the pumpkin it becomes (which the unit sphere encloses),
        // so that the pumpkin's surface is not inside its own occluder.
        BonsaiRenderConfig lightConfig;
        lightConfig.extent = VkExtent2D{64, 64};
        lightConfig.numberPaths = 1;
        lightConfig.lightCeiling = lightCeiling;
        lightConfig.movingObjects.push_back(std::make_shared<BonsaiObject>(bonsaiObjectFromMesh(
            pumpkinMesh, pumpkinTexture, morphOptions.training.textureWidth, morphOptions.training.textureHeight)));
        BonsaiRenderer light(vc, sceneWithoutPumpkin, std::make_shared<OffscreenTarget>(vc, lightConfig.extent, 1),
                             lightConfig);
        raster::MeshMorphOptions litMorph = morphOptions;
        litMorph.lightSettings = light.getSettingsRef();
        litMorph.light = light.getLightVolume().getLight();
        raster::MeshMorph morph(vc, pumpkinMesh, pumpkin->buffers(), frame, splatting->getOutputLayout(), splatting,
                                litMorph);
        ComputeGraph graph(vc, 1);
        // Not light.initialize(): it fills the buffers of the bonsai's drawing, which is not in this graph.
        graph.compileFrom(morph.getOutput());
        morph.initialize();
        morph.getTraining().setTexture(pumpkinTexture);
        camera->ubo = cameraOf(titlePose, aspect);
        camera->update(0);
        morph.setCamera(camera->ubo.proj * camera->ubo.view);
        const float titleYaw = facing(sphereCentre, titlePose.eye);
        const glm::mat4 model = pumpkinModel(sphereCentre, titleYaw, titleScale, titleTilt);
        morph.setModel(model);
        light.setMovingObject(0, model * glm::scale(glm::mat4(1.0f), glm::vec3(0.7f)));
        light.setTime(0.0f, 0.5f, 2.0f, glm::vec4(0.0f));
        setTrees(light, sceneWithoutPumpkin, 0.0f);
        auto colours = std::dynamic_pointer_cast<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(
            textData->buffers().colAlpha.element);
        const auto [begin, end] = chapterRange(0);
        for (uint32_t f = begin; f < end; ++f) {
            // The title fades out over the chapter's last second.
            const float fade = smooth((timeline.timeInChapter(end - 1) - timeline.timeInChapter(f)) / 1.0f);
            std::vector<glm::vec4> faded = textColours;
            for (glm::vec4& c : faded)
                c.a *= fade;
            colours->getBuffer().memcopyFrom(faded);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            if ((f - begin) % step == 0) {
                finishFrame(f, readImage(vc, frame->getImage(0), splatting->getOutputLayout(), extent), 1.0f);
                std::cout << "frame " << f << ": morph loss " << morph.getTraining().loss() << std::endl;
            }
        }
        pumpkinMesh = morph.getTraining().mesh();
        pumpkinTexture = morph.getTraining().texture();
        raster::writeObj(pumpkinMesh, outDir / "pumpkin.obj", "pumpkin.ppm");
        raster::writeTexturePpm(pumpkinTexture, morphOptions.training.textureWidth, morphOptions.training.textureHeight,
                                outDir / "pumpkin.ppm");
    } else {
        // Later chapters only: the pumpkin of an earlier run.
        std::filesystem::path texturePath;
        pumpkinMesh = raster::readObj(outDir / "pumpkin.obj", &texturePath);
        uint32_t w = 0, h = 0;
        pumpkinTexture = raster::readTexturePpm(texturePath, w, h);
    }

    // Where the pumpkin comes to rest: on the soil, at its trained size.
    BonsaiObjectPlacement placement;
    placement.object = std::make_shared<BonsaiObject>(bonsaiObjectFromMesh(
        pumpkinMesh, pumpkinTexture, morphOptions.training.textureWidth, morphOptions.training.textureHeight));
    placement.position = landing;
    placement.yaw = facing(glm::vec3(landing.x, 0.0f, landing.y), orbitPose(orbitStart).eye);
    placement.height = morphScale * (placement.object->high.y - placement.object->low.y);
    const glm::mat4 rest = bonsaiObjectTransform(sceneWithoutPumpkin, placement);
    const glm::vec3 restOrigin(rest[3]);

    BonsaiRenderConfig renderConfig;
    renderConfig.extent = extent;
    renderConfig.mode = BonsaiRenderMode::Classical;
    renderConfig.background = true;
    renderConfig.numberPaths = 1;

    // Chapter 1: the drop, the pumpkin still drawn by the morph (which keeps refining it), lit by the scene and
    // casting its shadow as it falls.
    if (fromChapter <= 1 && toChapter >= 1) {
        auto frame = std::make_shared<OffscreenTarget>(vc, extent, 1);
        BonsaiRenderConfig dropConfig = renderConfig;
        dropConfig.lightCeiling = lightCeiling;
        dropConfig.movingObjects.push_back(placement.object);
        BonsaiRenderer bonsai(vc, sceneWithoutPumpkin, frame, dropConfig);
        const VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        raster::MeshMorphOptions litMorph = morphOptions;
        litMorph.lightSettings = bonsai.getSettingsRef();
        litMorph.light = bonsai.getLightVolume().getLight();
        raster::MeshMorph morph(vc, pumpkinMesh, pumpkin->buffers(), frame, layout, bonsai.getOutput(), litMorph);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(morph.getOutput());
        bonsai.initialize(1);
        morph.initialize();
        morph.getTraining().setTexture(pumpkinTexture);
        bonsai.setBackground(glm::vec3(1.0f));
        const auto [begin, end] = chapterRange(1);
        const float duration = float(end - begin) / fps;
        for (uint32_t f = begin; f < end; ++f) {
            const float t = timeline.timeInChapter(f);
            // The camera comes down from the title to the pot; the pumpkin falls with it, lands and bounces once.
            const Pose pose = mix(titlePose, orbitPose(orbitStart), smooth(t / duration));
            const float fall = std::min(t / (0.75f * duration), 1.0f);
            glm::vec3 origin = glm::mix(sphereCentre, restOrigin, fall * fall);
            if (fall >= 1.0f) {
                const float after = (t - 0.75f * duration) / (0.25f * duration);
                origin.y += 0.12f * placement.height * std::sin(3.14159265f * std::clamp(after, 0.0f, 1.0f));
            }
            bonsai.getCamera()->ubo = cameraOf(pose, aspect);
            bonsai.getCamera()->update(0);
            bonsai.setView(pose.eye);
            bonsai.setTime(t, 0.5f, 2.0f, glm::vec4(0.0f));
            setTrees(bonsai, sceneWithoutPumpkin, 0.0f);
            morph.setCamera(bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view);
            const float settle = smooth(fall);
            // The short way round from the title's yaw to the resting one.
            const float titleYaw = facing(sphereCentre, titlePose.eye);
            const float turn = std::remainder(placement.yaw - titleYaw, 2.0f * 3.14159265f);
            const glm::mat4 model = pumpkinModel(origin, titleYaw + settle * turn,
                                                 glm::mix(titleScale, morphScale, settle), (1.0f - settle) * titleTilt);
            morph.setModel(model);
            bonsai.setMovingObject(0, model);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            if ((f - begin) % step == 0) {
                finishFrame(f, readImage(vc, frame->getImage(0), layout, extent), 1.0f);
                std::cout << "frame " << f << std::endl;
            }
        }
    }

    // Chapters 2 and 3: the pumpkin rests in the scene; the trees grow, then the days pass.
    if (toChapter >= 2) {
        BonsaiSceneConfig withPumpkin = sceneConfig;
        withPumpkin.objects.push_back(placement);
        const BonsaiScene scene = makeBonsaiScene(withPumpkin);
        auto frame = std::make_shared<OffscreenTarget>(vc, extent, 1);
        BonsaiRenderer bonsai(vc, scene, frame, renderConfig);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        BonsaiDays days;
        // The growth's summer noon turns into the night over the first seconds; then four days from midnight.
        const float afternoon = 4.0f;
        days.secondsPerDay = (kChapters[3].seconds - afternoon) / 4.0f;
        days.dawn = 0.0f;
        days.seasons[0] = 2.0f, days.seasons[1] = 3.25f, days.seasons[2] = 4.0f, days.seasons[3] = 5.25f;
        const uint32_t begin = timeline.chapterStart(std::max(fromChapter, 2)), end = timeline.frameCount();
        const uint32_t growthStart = timeline.chapterStart(2), daysStart = timeline.chapterStart(3);
        for (uint32_t f = begin; f < end; ++f) {
            const float sinceGrowth = float(f - growthStart) / fps;
            const Pose pose = orbitPose(growthOrbit(sinceGrowth));
            bonsai.getCamera()->ubo = cameraOf(pose, aspect);
            bonsai.getCamera()->update(0);
            bonsai.setView(pose.eye);
            float background = 1.0f, timeOfDay = 0.5f, season = 2.0f;
            if (f < daysStart) {
                // Quick at first, slowing down as the tree matures.
                const float t = std::clamp(sinceGrowth / kChapters[2].seconds, 0.0f, 1.0f);
                const float growth = 1.0f - (1.0f - t) * (1.0f - t);
                setTrees(bonsai, scene, 0.02f + 0.98f * growth);
                bonsai.setTime(sinceGrowth, timeOfDay, season, glm::vec4(1.0f, 0.0f, 0.0f, 0.3f * growth));
            } else {
                // The camera keeps orbiting.
                const float seconds = float(f - daysStart) / fps;
                setTrees(bonsai, scene, 1.0f);
                timeOfDay =
                    seconds < afternoon ? 0.5f + 0.5f * seconds / afternoon : days.timeOfDayAt(seconds - afternoon);
                season = seconds < afternoon ? days.seasons[0] : days.seasonAt(seconds - afternoon);
                bonsai.setTime(sinceGrowth, timeOfDay, season, glm::vec4(1.0f, 0.0f, 0.0f, 0.4f));
                background = backgroundAt(timeOfDay, season);
            }
            bonsai.setBackground(glm::vec3(background));
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            if ((f - begin) % step == 0) {
                std::vector<uint8_t> image =
                    readImage(vc, frame->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
                drawSky(image, bonsai.getCamera()->ubo.view, treeSkyAt(timeOfDay, season), background);
                finishFrame(f, std::move(image), background);
                std::cout << "frame " << f << std::endl;
            }
        }
    }
    std::cout << "frames in " << (outDir / "frames").string() << std::endl;
    return 0;
}
