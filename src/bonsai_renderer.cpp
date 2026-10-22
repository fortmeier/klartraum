// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/bonsai_renderer.hpp"

#include <algorithm>
#include <cmath>

#include "klartraum/gaussian_transform.hpp"
#include "klartraum/tree/tree_placement.hpp"

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
    transformGaussians(kept, scene.lanternRotation, scene.lanternScale, scene.lanternTranslation);
    return kept;
}

BonsaiRenderer::BonsaiRenderer(VulkanContext& vulkanContext, const BonsaiScene& scene,
                               std::shared_ptr<ImageViewSrc> target, const BonsaiRenderConfig& config)
    : config(config),
      lanternFlame(scene.lanternFlame),
      terrainParams(scene.terrain),
      placement(scene.placement),
      types(scene.types),
      seed(scene.config.seed),
      nearTrees(scene.placed ? uint32_t(scene.trees.size()) : 0u) {
    const TreeFrame frame = defaultTreeFrame();
    TreeCapacity capacity = TreeCapacity::fromReference(scene.types, scene.trees, frame);
    if (scene.placed) {
        // Room for the near trees all of either type at the tallest height of the placement.
        for (uint32_t type = 0; type < scene.types.size(); ++type) {
            std::vector<TreeInstance> tallest = scene.trees;
            for (TreeInstance& inst : tallest) {
                inst.ids.x = type;
                inst.posScale.w =
                    placement.size.x * (1.0f + placement.size.y) / placement.grid.w / scene.types[type].shapeBase.z;
            }
            const TreeCapacity c = TreeCapacity::fromReference(scene.types, tallest, frame);
            for (size_t level = 0; level < capacity.stems.size(); ++level) {
                capacity.stems[level] = std::max(capacity.stems[level], c.stems[level]);
            }
            capacity.segments = std::max(capacity.segments, c.segments);
            capacity.leaves = std::max(capacity.leaves, c.leaves);
        }
    }
    generator = std::make_shared<TreeGenerator>(vulkanContext, scene.types, scene.trees, capacity, frame);
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
                           glm::vec3(pot.halfLength + 0.15f,
                                     std::max(1.0f, 1.6f * scene.config.scale.treeHeightUnits() + 0.4f),
                                     pot.halfWidth + 0.15f),
                           config.lightVoxel);
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
    // Patches of the captured moss, placed over the soil with a level of detail each; too flat to cast shadows.
    const std::vector<MossPatch> mossPatches = extractMossPatches(loadGaussiansSpz(config.lanternPath, true));
    const std::vector<TreeLodInstance> mossCopies =
        // As many copies as cover the soil at the lantern's scale, the moss scaled with the lantern.
        placeMossPatches(
            scene, uint32_t(mossPatches.size()),
            uint32_t(std::min(float(config.mossPatches) / (scene.lanternScale * scene.lanternScale), 20000.0f)),
            scene.config.seed, 1.3f * scene.lanternScale);
    TreeGaussianSources sources;
    sources.captured = std::move(lanternGaussians);
    sources.light = lightVolume->getLight();
    const uint32_t mossSlots = mossCopies.empty() ? 0u : config.mossPool;
    const bool terrainGaussianPool = !classical && config.terrainLod;
    const uint32_t proxySlots = scene.placed && !classical ? 4u * config.proxies.capacity : 0u;
    // The canopy and the proxies crossfade over the same band of tree sizes, in rendered pixels.
    TreeProxyConfig proxyConfig = config.proxies;
    TreeCanopy canopy;
    if (scene.placed) {
        canopy.enabled = true;
        canopy.placement = placement;
        canopy.fullBelow = config.canopyPixels.x * float(supersample);
        canopy.noneAbove = config.canopyPixels.y * float(supersample);
        proxyConfig.minPixels = canopy.fullBelow;
        proxyConfig.fadeAbove = canopy.noneAbove;
    }
    sources.extraSlots = mossSlots + (terrainGaussianPool ? config.terrainPool : 0u) + proxySlots;
    // The classical renderer draws only the captured lantern and moss as Gaussians.
    sources.trees = !classical;
    if (!classical) {
        sources.surfels = sampleMeshSurfels(config.terrainLod ? potMesh : scene.mesh, config.meshGaussianSpacing, 1);
    }
    gaussians = std::make_unique<TreeGaussians>(vulkanContext, *generator, camera, settings, std::move(sources));
    GaussianSoABuffers splats = gaussians->getGaussians();
    if (!mossCopies.empty()) {
        moss = std::make_unique<TreeMossLod>(vulkanContext, *gaussians, mossPatches, mossCopies, camera, settings,
                                             lightVolume->getLight(), config.mossLod, mossSlots);
        splats = moss->getGaussians();
    }
    if (config.terrainLod) {
        terrain = std::make_unique<TreeTerrain>(vulkanContext, scene.terrain, camera, config.terrain, canopy, settings);
    }
    if (terrainGaussianPool) {
        terrainGaussians = std::make_unique<TreeTerrainGaussians>(
            vulkanContext, *terrain, splats, gaussians->getExtraOffset() + mossSlots, config.terrainPool, settings,
            camera, lightVolume->getLight());
        splats = terrainGaussians->getGaussians();
    }
    if (scene.placed) {
        TreeTileLayout layout = config.tiles;
        if (layout.range <= 0.0f) {
            const float focal = 0.5f * float(renderExtent.height) / std::tan(glm::radians(20.0f));
            layout.range = scene.config.scale.treeHeightUnits() * focal / proxyConfig.minPixels;
        }
        tiles = std::make_unique<TreePlacementTiles>(vulkanContext, scene.terrain, placement, layout);
    }
    if (tiles && !classical) {
        proxies = std::make_unique<TreeProxies>(vulkanContext, *tiles, camera, settings, renderExtent,
                                                TreeRenderInputs{BufferRef{}, lightVolume->getLight()}, proxyConfig);
        proxyGaussians = std::make_unique<TreeProxyGaussians>(
            vulkanContext, *proxies, splats, gaussians->getExtraOffset() + sources.extraSlots - proxySlots,
            config.proxies.capacity, settings, camera, lightVolume->getLight());
        splats = proxyGaussians->getGaussians();
    }

    if (classical) {
        std::shared_ptr<ImageViewSrc> renderTarget = target;
        if (supersample > 1) {
            renderTarget = std::make_shared<OffscreenTarget>(vulkanContext, renderExtent, config.numberPaths);
        }
        GsplatConfig splatConfig;
        splatConfig.outputDepth = true;
        auto splatting =
            createGaussianSplatting(vulkanContext, config.backend, renderTarget, camera, splats, splatConfig);
        const TreeRenderInputs inputs{BufferRef{getGaussianSplattingDepth(splatting), -1}, lightVolume->getLight()};

        auto toGeneral = std::make_shared<ImageToGeneral>();
        toGeneral->setInput(renderTarget, 0);
        toGeneral->addDependency(splatting);

        stems = std::make_unique<TreeStemRenderer>(vulkanContext, *generator, camera, settings, renderExtent, inputs);
        if (scene.placed && config.reducedTrees > 0) {
            // The reduced trees, all hidden until setView() places them.
            for (const TreeTypeParams& type : scene.types) {
                reducedTypes.push_back(treeReducedType(type));
            }
            reducedCount = config.reducedTrees;
            std::vector<TreeInstance> hidden(reducedCount, scene.trees.front());
            std::vector<TreeInstance> tallest = hidden;
            TreeCapacity reducedCapacity{};
            for (uint32_t type = 0; type < reducedTypes.size(); ++type) {
                for (TreeInstance& inst : tallest) {
                    inst.ids.x = type;
                    inst.posScale.w = placement.size.x * (1.0f + placement.size.y) / placement.grid.w /
                                      reducedTypes[type].shapeBase.z;
                }
                const TreeCapacity c = TreeCapacity::fromReference(reducedTypes, tallest, frame);
                for (size_t level = 0; level < reducedCapacity.stems.size(); ++level) {
                    reducedCapacity.stems[level] = std::max(reducedCapacity.stems[level], c.stems[level]);
                }
                reducedCapacity.segments = std::max(reducedCapacity.segments, c.segments);
                reducedCapacity.leaves = std::max(reducedCapacity.leaves, c.leaves);
            }
            reducedGenerator =
                std::make_shared<TreeGenerator>(vulkanContext, reducedTypes, hidden, reducedCapacity, frame);
            reducedStems = std::make_unique<TreeStemRenderer>(vulkanContext, *reducedGenerator, camera, settings,
                                                              renderExtent, inputs);
            reducedLeaves = std::make_unique<TreeLeafRenderer>(vulkanContext, *reducedGenerator, reducedTypes, camera,
                                                               settings, renderExtent, inputs);
        }
        leaves = std::make_unique<TreeLeafRenderer>(vulkanContext, *generator, scene.types, camera, settings,
                                                    renderExtent, inputs);
        meshes = std::make_unique<SceneMeshRenderer>(vulkanContext, config.terrainLod ? potMesh : scene.mesh, settings,
                                                     renderExtent, inputs);
        if (tiles) {
            proxies = std::make_unique<TreeProxies>(vulkanContext, *tiles, camera, settings, renderExtent, inputs,
                                                    proxyConfig);
        }
        if (terrain) {
            terrainRenderer =
                std::make_unique<TreeTerrainRenderer>(vulkanContext, *terrain, settings, renderExtent, inputs);
        }

        pass = std::make_shared<RenderPass>(vulkanContext.getSwapChainImageFormat(), renderExtent);
        pass->setInput(toGeneral, 0);
        pass->setInput(camera, 1);
        pass->setLoadExisting(true);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(stems->getClassify());
        pass->addComputeDependency(leaves->getClassify());
        pass->addComputeDependency(settings);
        if (reducedGenerator) {
            pass->addComputeDependency(reducedStems->getClassify());
            pass->addComputeDependency(reducedLeaves->getClassify());
            pass->addDrawComponent(reducedStems->getDrawComponent());
            pass->addDrawComponent(reducedLeaves->getDrawComponent());
        }
        pass->addComputeDependency(lightVolume->getElement());
        pass->addDrawComponent(meshes->getDrawComponent());
        if (terrain) {
            pass->addComputeDependency(terrain->getElement());
            pass->addDrawComponent(terrainRenderer->getDrawComponent());
        }
        if (proxies) {
            pass->addComputeDependency(proxies->getElement());
            if (config.impostors) {
                atlas = std::make_unique<TreeImpostorAtlas>(vulkanContext, scene.types, config.impostorAtlas);
                impostors = std::make_unique<TreeImpostorRenderer>(
                    vulkanContext, proxies->getVisible(), proxies->getDrawArgs(), sizeof(VkDrawIndirectCommand), *atlas,
                    settings, renderExtent, inputs);
                pass->addDrawComponent(impostors->getDrawComponent());
            } else {
                pass->addDrawComponent(proxies->getDrawComponent());
            }
        }
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
        output = createGaussianSplatting(vulkanContext, config.backend, target, camera, splats);
    }
}

void BonsaiRenderer::setSeason(float season) {
    TreeFrame frame = generator->getFrame().values().at(0);
    frame.timeSeason.y = season;
    generator->getFrame().set(0, frame);
    if (reducedGenerator) {
        reducedGenerator->getFrame().set(0, frame);
    }
    TreeRenderSettings values = settings->values().at(0);
    values.season = glm::vec4(season, tree::treeSnowCover(season), 0.0f, 0.0f);
    settings->set(0, values);
}

void BonsaiRenderer::setTime(float seconds, float timeOfDay, float season, glm::vec4 wind) {
    TreeFrame frame = generator->getFrame().values().at(0);
    frame.timeSeason = glm::vec4(seconds, season, 0.0f, 0.0f);
    frame.wind = wind;
    generator->getFrame().set(0, frame);
    if (reducedGenerator) {
        reducedGenerator->getFrame().set(0, frame);
    }
    TreeRenderSettings values = settings->values().at(0);
    applyTreeSky(values, treeSkyAt(timeOfDay, season), lanternFlame, seconds);
    values.season = glm::vec4(season, tree::treeSnowCover(season), 0.0f, 0.0f);
    settings->set(0, values);
}

void BonsaiRenderer::setView(glm::vec3 eye) {
    if (!tiles) {
        return;
    }
    tiles->update(eye);
    // The nearest trees in full, the next ones reduced (L1).
    const std::vector<TreeSite> sites = nearestTreeSites(terrainParams, placement, eye, nearTrees + reducedCount);
    auto instancesOf = [&](size_t from, size_t count, const std::vector<TreeTypeParams>& of) {
        std::vector<TreeInstance> instances;
        for (size_t i = from; i < std::min(sites.size(), from + count); ++i) {
            instances.push_back(treeInstanceFor(sites[i], of, seed));
        }
        // Over open land fewer trees are near; the rest are tiny and far below the pot.
        while (instances.size() < count) {
            TreeInstance hidden{};
            hidden.posScale = glm::vec4(0.0f, -100.0f, 0.0f, 1e-4f);
            hidden.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
            instances.push_back(hidden);
        }
        return instances;
    };
    generator->getInstances().set(instancesOf(0, nearTrees, types));
    if (reducedGenerator) {
        reducedGenerator->getInstances().set(instancesOf(nearTrees, reducedCount, reducedTypes));
    }
    proxies->setNearRadius(sites.empty() ? -1.0f : glm::length(glm::vec3(sites.back().posHeight) - eye));
}

void BonsaiRenderer::initialize(uint32_t numberPaths) { gaussians->initializeBuffers(numberPaths); }

} // namespace klartraum
