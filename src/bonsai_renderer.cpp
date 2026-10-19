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
    : config(config) {
    const TreeFrame frame = defaultTreeFrame();
    generator = std::make_shared<TreeGenerator>(vulkanContext, scene.types, scene.trees,
                                                TreeCapacity::fromReference(scene.types, scene.trees, frame), frame);
    camera = std::make_shared<CameraUboType>();
    TreeRenderSettings values = defaultTreeRenderSettings(config.extent);
    values.occlusion.x = config.mode == BonsaiRenderMode::Classical ? 1.0f : 0.0f;
    // The light volume spans the pot and the tallest trees.
    const BonsaiPot& pot = scene.config.pot;
    configureTreeLightGrid(values, glm::vec3(-pot.halfLength - 0.15f, -0.05f, -pot.halfWidth - 0.15f),
                           glm::vec3(pot.halfLength + 0.15f, 2.7f, pot.halfWidth + 0.15f), config.lightVoxel);
    settings = std::make_shared<HostValues<TreeRenderSettings>>(vulkanContext, std::vector<TreeRenderSettings>{values});
    lightVolume = std::make_unique<TreeLightVolume>(vulkanContext, *generator, settings);
    std::vector<Gaussian3D> lanternGaussians = loadLantern(scene, config.lanternPath);

    if (config.mode == BonsaiRenderMode::Classical) {
        lantern = std::make_shared<GaussianDataStandard>(vulkanContext, std::move(lanternGaussians));
        GsplatConfig splatConfig;
        splatConfig.outputDepth = true;
        auto splatting = createGaussianSplatting(vulkanContext, config.backend, target, camera, lantern, splatConfig);
        const TreeRenderInputs inputs{BufferRef{getGaussianSplattingDepth(splatting), -1}, lightVolume->getLight()};

        auto toGeneral = std::make_shared<ImageToGeneral>();
        toGeneral->setInput(target, 0);
        toGeneral->addDependency(splatting);

        stems = std::make_unique<TreeStemRenderer>(vulkanContext, *generator, camera, settings, config.extent, inputs);
        leaves = std::make_unique<TreeLeafRenderer>(vulkanContext, *generator, scene.types, camera, settings,
                                                    config.extent, inputs);
        meshes = std::make_unique<SceneMeshRenderer>(vulkanContext, scene.mesh, settings, config.extent, inputs);

        pass = std::make_shared<RenderPass>(vulkanContext.getSwapChainImageFormat(), config.extent);
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
    } else {
        gaussians = std::make_unique<TreeGaussians>(
            vulkanContext, *generator, camera, settings, std::move(lanternGaussians),
            sampleMeshSurfels(scene.mesh, config.meshGaussianSpacing, 1), lightVolume->getLight());
        output = createGaussianSplatting(vulkanContext, config.backend, target, camera, gaussians->getGaussians());
    }
}

void BonsaiRenderer::initialize(uint32_t numberPaths) {
    if (gaussians) {
        gaussians->initializeBuffers(numberPaths);
    }
}

} // namespace klartraum
