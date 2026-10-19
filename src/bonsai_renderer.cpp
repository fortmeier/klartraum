// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/bonsai_renderer.hpp"

#include <algorithm>
#include <cmath>

#include "klartraum/gaussian_transform.hpp"

namespace klartraum {

std::vector<Gaussian3D> BonsaiRenderer::loadLantern(const BonsaiScene& scene, const std::string& path) {
    // The capture is stored with y pointing down; its ground lies at y = 0 and
    // the lantern stands about 0.28 high on a moss patch about 0.22 around.
    std::vector<Gaussian3D> all = loadGaussiansSpz(path, true);
    std::vector<Gaussian3D> kept;
    for (const Gaussian3D& g : all) {
        const float r = std::sqrt(g.position[0] * g.position[0] + g.position[2] * g.position[2]);
        if (r < 0.18f && g.position[1] > -0.05f && g.position[1] < 0.32f) {
            kept.push_back(g);
        }
    }
    transformGaussians(kept, scene.lanternRotation, scene.config.lanternScale, scene.lanternTranslation);
    return kept;
}

BonsaiRenderer::BonsaiRenderer(VulkanContext& vulkanContext, const BonsaiScene& scene,
                               std::shared_ptr<ImageViewSrc> target, const BonsaiRenderConfig& config)
    : config(config),
      lanternFlame(scene.lanternFlame) {
    const TreeFrame frame = defaultTreeFrame();
    generator = std::make_shared<TreeGenerator>(vulkanContext, scene.types, scene.trees,
                                                TreeCapacity::fromReference(scene.types, scene.trees, frame), frame);
    camera = std::make_shared<CameraUboType>();
    // The classical renderer draws at `supersample` times the resolution and filters down. Its level of
    // detail is set in pixels, which are smaller then.
    const bool classical = config.mode == BonsaiRenderMode::Classical;
    const uint32_t supersample = classical ? std::max(config.supersample, 1u) : 1u;
    const VkExtent2D renderExtent{config.extent.width * supersample, config.extent.height * supersample};
    TreeRenderSettings values = defaultTreeRenderSettings(renderExtent);
    values.viewport.z *= float(supersample);
    values.viewport.w *= float(supersample);
    values.leafLod *= glm::vec4(float(supersample), float(supersample), float(supersample), 1.0f);
    values.occlusion.x = classical ? 1.0f : 0.0f;
    // The light volume spans the pot and the tallest trees.
    const BonsaiPot& pot = scene.config.pot;
    configureTreeLightGrid(values, glm::vec3(-pot.halfLength - 0.15f, -0.05f, -pot.halfWidth - 0.15f),
                           glm::vec3(pot.halfLength + 0.15f, 2.7f, pot.halfWidth + 0.15f), config.lightVoxel);
    settings = std::make_shared<HostValues<TreeRenderSettings>>(vulkanContext, std::vector<TreeRenderSettings>{values});
    std::vector<Gaussian3D> lanternGaussians = loadLantern(scene, config.lanternPath);
    // The pot and the lantern cast shadows; the soil's gentle mound is left out, which would shadow itself
    // in voxels this coarse.
    std::vector<TreeSceneVertex> potMesh;
    for (const TreeSceneVertex& v : scene.mesh) {
        if (v.color.w == TREE_MATERIAL_GLAZE) {
            potMesh.push_back(v);
        }
    }
    lightVolume = std::make_unique<TreeLightVolume>(
        vulkanContext, *generator, settings,
        treeStaticOcclusion(values, sampleMeshSurfels(potMesh, 0.5f * config.lightVoxel, 3), lanternGaussians));

    if (classical) {
        std::shared_ptr<ImageViewSrc> renderTarget = target;
        if (supersample > 1) {
            renderTarget = std::make_shared<OffscreenTarget>(vulkanContext, renderExtent, config.numberPaths);
        }
        lantern = std::make_shared<GaussianDataStandard>(vulkanContext, std::move(lanternGaussians));
        const GaussianSoABuffers tinted =
            tintCapturedGaussians(vulkanContext, lantern->buffers(), settings, lanternTint, lightVolume->getLight());
        GsplatConfig splatConfig;
        splatConfig.outputDepth = true;
        auto splatting =
            createGaussianSplatting(vulkanContext, config.backend, renderTarget, camera, tinted, splatConfig);
        const TreeRenderInputs inputs{BufferRef{getGaussianSplattingDepth(splatting), -1}, lightVolume->getLight()};

        auto toGeneral = std::make_shared<ImageToGeneral>();
        toGeneral->setInput(renderTarget, 0);
        toGeneral->addDependency(splatting);

        stems = std::make_unique<TreeStemRenderer>(vulkanContext, *generator, camera, settings, renderExtent, inputs);
        leaves = std::make_unique<TreeLeafRenderer>(vulkanContext, *generator, scene.types, camera, settings,
                                                    renderExtent, inputs);
        meshes = std::make_unique<SceneMeshRenderer>(vulkanContext, scene.mesh, settings, renderExtent, inputs);

        pass = std::make_shared<RenderPass>(vulkanContext.getSwapChainImageFormat(), renderExtent);
        pass->setInput(toGeneral, 0);
        pass->setInput(camera, 1);
        pass->setLoadExisting(true);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(stems->getClassify());
        pass->addComputeDependency(leaves->getClassify());
        pass->addComputeDependency(settings);
        pass->addComputeDependency(lightVolume->getElement());
        pass->addDrawComponent(meshes->getDrawComponent());
        pass->addDrawComponent(stems->getDrawComponent());
        pass->addDrawComponent(leaves->getDrawComponent());
        output = pass;
        if (supersample > 1) {
            // Bilinear filtering at half the resolution averages 2 x 2 pixels.
            auto resample =
                std::make_shared<ImageResample>(vulkanContext, renderExtent, config.extent, ResampleFilter::Bilinear,
                                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            resample->setInput(pass, 0);
            resample->setInput(target, 1);
            auto toFinal = std::make_shared<ImageToFinalLayout>();
            toFinal->setInput(resample, 0, 1);
            output = toFinal;
        }
    } else {
        gaussians = std::make_unique<TreeGaussians>(
            vulkanContext, *generator, camera, settings, std::move(lanternGaussians),
            sampleMeshSurfels(scene.mesh, config.meshGaussianSpacing, 1), lightVolume->getLight());
        output = createGaussianSplatting(vulkanContext, config.backend, target, camera, gaussians->getGaussians());
    }
}

void BonsaiRenderer::setSeason(float season) {
    TreeFrame frame = generator->getFrame().values().at(0);
    frame.timeSeason.y = season;
    generator->getFrame().set(0, frame);
    TreeRenderSettings values = settings->values().at(0);
    values.season = glm::vec4(season, tree::treeSnowCover(season), 0.0f, 0.0f);
    settings->set(0, values);
}

void BonsaiRenderer::setTime(float seconds, float timeOfDay, float season, glm::vec4 wind) {
    TreeFrame frame = generator->getFrame().values().at(0);
    frame.timeSeason = glm::vec4(seconds, season, 0.0f, 0.0f);
    frame.wind = wind;
    generator->getFrame().set(0, frame);
    TreeRenderSettings values = settings->values().at(0);
    applyTreeSky(values, treeSkyAt(timeOfDay, season), lanternFlame, seconds);
    values.season = glm::vec4(season, tree::treeSnowCover(season), 0.0f, 0.0f);
    settings->set(0, values);
}

void BonsaiRenderer::initialize(uint32_t numberPaths) {
    if (gaussians) {
        gaussians->initializeBuffers(numberPaths);
    }
}

} // namespace klartraum
