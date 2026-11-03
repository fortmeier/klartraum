// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/bonsai_renderer.hpp"

#include "klartraum/tree/tree_gaussian_features.hpp"

#include <algorithm>
#include <cmath>

#include "klartraum/rendering/gaussian_transform.hpp"
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

std::shared_ptr<ImageViewSrc> BonsaiRenderer::withBackground(const std::shared_ptr<ImageViewSrc>& frame) {
    if (!config.background)
        return frame;
    backgroundFill = std::make_shared<ImageFill>(glm::vec3(1.0f));
    backgroundFill->setInput(frame, 0);
    return backgroundFill;
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
    if (scene.placed) {
        // A grid of fixed size that follows the near trees (setView()), with voxels a fraction of a tree.
        values.lightGridSize = glm::uvec4(kFollowGrid, 1u);
    }
    if (scene.terrain.relief.x > 0.0f) {
        // Haze in the world above the soil, up to its peaks and the trees on them.
        values.atmosphere =
            glm::vec4(scene.config.scale.metersPerUnit(), config.visibilityMeters, config.valleyFog, 0.0f);
        values.atmosphereBox = glm::vec4(scene.terrain.outline.x, scene.terrain.outline.y, scene.terrain.soil.x,
                                         scene.terrain.soil.x + scene.terrain.soil.y + scene.terrain.relief.x +
                                             scene.config.scale.treeHeightUnits());
    }
    settings = std::make_shared<HostValues<TreeRenderSettings>>(vulkanContext, std::vector<TreeRenderSettings>{values});
    std::vector<Gaussian3D> lanternGaussians = loadLantern(scene, config.lanternPath);
    // The captured pot and objects join the lantern.
    lanternGaussians.insert(lanternGaussians.end(), scene.objectGaussians.begin(), scene.objectGaussians.end());
    // The pot, the objects and the lantern cast shadows; the soil's gentle mound is left out, which would shadow
    // itself in voxels this coarse.
    std::vector<TreeSceneVertex> potMesh;
    for (const TreeSceneVertex& v : scene.mesh) {
        if (v.color.w == TREE_MATERIAL_GLAZE || v.color.w == TREE_MATERIAL_CAPTURED) {
            potMesh.push_back(v);
        }
    }
    // The moving objects' surfaces in their own frames, sampled finely for their size.
    TreeMovingOccluders moving;
    moving.objectCount = uint32_t(config.movingObjects.size());
    for (uint32_t i = 0; i < moving.objectCount; ++i) {
        const BonsaiObject& object = *config.movingObjects[i];
        const glm::vec3 extent = object.high - object.low;
        const float spacing = 0.02f * std::max({extent.x, extent.y, extent.z, 1e-3f});
        const std::vector<glm::vec4> samples =
            treeOccluderSamples(sampleMeshSurfels(object.mesh, spacing, 3), object.gaussians);
        moving.surfels.insert(moving.surfels.end(), samples.begin(), samples.end());
        moving.objects.insert(moving.objects.end(), samples.size(), i);
    }
    // The pot and the lantern cast shadows in the grid over the pot; a grid following the trees has none.
    lightVolume = std::make_unique<TreeLightVolume>(
        vulkanContext, *generator, settings,
        scene.placed
            ? std::vector<float>{}
            : treeStaticOcclusion(values, sampleMeshSurfels(potMesh, 0.5f * config.lightVoxel, 3), lanternGaussians),
        std::move(moving));
    for (uint32_t i = 0; i < config.movingObjects.size(); ++i) {
        lightVolume->setMovingTransform(i, glm::mat4(0.0f));
    }
    // Patches of the captured moss, placed over the soil with a level of detail each; too flat to cast shadows.
    const std::vector<MossPatch> mossPatches = extractMossPatches(loadGaussiansSpz(config.lanternPath, true));
    // As many copies as cover the soil at the lantern's scale, the moss scaled with the lantern; none where the
    // trees follow the placement, where a patch of moss is far smaller than a pixel.
    const std::vector<TreeLodInstance> mossCopies =
        scene.placed
            ? std::vector<TreeLodInstance>{}
            : placeMossPatches(
                  scene, uint32_t(mossPatches.size()),
                  uint32_t(std::min(float(config.mossPatches) / (scene.lanternScale * scene.lanternScale), 20000.0f)),
                  scene.config.seed, 1.3f * scene.lanternScale);
    TreeGaussianSources sources;
    sources.captured = std::move(lanternGaussians);
    sources.light = lightVolume->getLight();
    const uint32_t mossSlots = mossCopies.empty() ? 0u : config.mossPool;
    const bool terrainGaussianPool = !classical && config.terrainLod;
    const uint32_t proxySlots = scene.placed && !classical ? config.forestPool : 0u;
    const uint32_t grassSlots = scene.placed && !classical && config.grass && config.terrainLod ? config.grassPool : 0u;
    const uint32_t floorSlots = scene.placed && config.forestFloor && !mossPatches.empty() ? config.floorPool : 0u;
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
        proxyConfig.bySize = !classical;
        canopyBand = glm::vec2(canopy.fullBelow, canopy.noneAbove);
        TreeRenderSettings band = settings->values().at(0);
        band.canopyBand = glm::vec4(canopyBand, 0.0f, 0.0f);
        settings->set(0, band);
    }
    // The pools in the extra slots, one after the other.
    const uint32_t terrainSlots = terrainGaussianPool ? config.terrainPool : 0u;
    sources.extraSlots = mossSlots + terrainSlots + grassSlots + floorSlots + proxySlots;
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
    if (grassSlots > 0) {
        grass = std::make_unique<TreeGrass>(vulkanContext, scene.terrain, makeTreeGrass(scene, config.grassBlades),
                                            camera, settings, renderExtent);
        grassGaussians = std::make_unique<TreeGrassGaussians>(vulkanContext, *grass, splats,
                                                              gaussians->getExtraOffset() + mossSlots + terrainSlots,
                                                              grassSlots, settings, camera, lightVolume->getLight());
        splats = grassGaussians->getGaussians();
    }
    if (floorSlots > 0) {
        const float scale = 1.3f * scene.lanternScale;
        forestFloor = std::make_unique<TreeForestFloor>(
            vulkanContext,
            makeTreeFloor(scene, uint32_t(mossPatches.size()), mossPatchRadius(mossPatches), scale, config.floor),
            splats, mossPatches, camera, settings, lightVolume->getLight(),
            glm::vec4(glm::vec3(config.floorLod) * float(supersample), config.floorLod.w), floorSlots,
            gaussians->getExtraOffset() + mossSlots + terrainSlots + grassSlots);
        splats = forestFloor->getGaussians();
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
        TreeGaussianForestConfig forestConfig = config.forest;
        forestConfig.trees = std::min(forestConfig.trees, proxyConfig.capacity);
        forestConfig.seed = seed;
        forest = std::make_unique<TreeGaussianForest>(
            vulkanContext, *proxies, splats, gaussians->getExtraOffset() + sources.extraSlots - proxySlots, proxySlots,
            scene.types, camera, settings, lightVolume->getLight(), forestConfig);
        splats = forest->getGaussians();
    }

    if (classical) {
        std::shared_ptr<ImageViewSrc> renderTarget = target;
        if (supersample > 1) {
            renderTarget = std::make_shared<OffscreenTarget>(vulkanContext, renderExtent, config.numberPaths);
        }
        GsplatConfig splatConfig;
        splatConfig.outputDepth = true;
        auto splatting = createGaussianSplatting(vulkanContext, config.backend, withBackground(renderTarget), camera,
                                                 splats, splatConfig);
        const TreeRenderInputs inputs{BufferRef{splatting->getDepth(), -1}, lightVolume->getLight()};

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
        // A captured pot of Gaussians leaves no mesh where the terrain draws the soil.
        const std::vector<TreeSceneVertex>& sceneMesh = config.terrainLod ? potMesh : scene.mesh;
        if (!sceneMesh.empty()) {
            meshes = std::make_unique<SceneMeshRenderer>(vulkanContext, sceneMesh, settings, renderExtent, inputs);
        }
        if (tiles && config.farTrees) {
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
        pass->setStoreDepth(config.storeDepth);
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
        if (meshes) {
            pass->addDrawComponent(meshes->getDrawComponent());
        }
        if (terrain) {
            pass->addComputeDependency(terrain->getElement());
            pass->addDrawComponent(terrainRenderer->getDrawComponent());
        }
        if (scene.placed && config.grass && config.terrainLod) {
            grass = std::make_unique<TreeGrass>(vulkanContext, scene.terrain, makeTreeGrass(scene, config.grassBlades),
                                                camera, settings, renderExtent, inputs);
            pass->addComputeDependency(grass->getElement());
            pass->addDrawComponent(grass->getDrawComponent());
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
        auto splatting = createGaussianSplatting(vulkanContext, config.backend, withBackground(target), camera, splats);
        output = splatting;
        if (config.features) {
            // The same Gaussians with feature colours (tree_gaussian_features.comp), splatted into targets of their
            // own; the output joins all three splattings.
            GaussianFeaturePush ranges{};
            ranges.treeBegin = gaussians->getTreeOffset();
            ranges.treeEnd = gaussians->getExtraOffset();
            if (forest) {
                ranges.copyBegin = gaussians->getExtraOffset() + gaussians->getExtraSlots() - config.forestPool;
                ranges.copyEnd = gaussians->getExtraOffset() + gaussians->getExtraSlots();
            }
            ranges.depthUnit = 1.0f / scene.config.scale.metersPerUnit();
            const GaussianFeatures features = splatGaussianFeatures(
                vulkanContext, splats, camera, config.extent, config.numberPaths, ranges, splatting, config.backend);
            featureTargets = features.targets;
            output = features.done;
        }
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
    if (grass) {
        grass->setWind(seconds, wind.w, glm::vec2(wind.x, wind.z));
    }
}

float TreeDetailBudget::update(float frameMs) {
    if (frameMs > 0.0f) {
        detail *= std::clamp(std::sqrt(budget / frameMs), 0.8f, 1.25f);
        detail = std::clamp(detail, minimum, 1.0f);
    }
    return detail;
}

void BonsaiRenderer::setView(glm::vec3 eye) {
    if (!tiles) {
        return;
    }
    if (grass) {
        grass->setView(eye);
    }
    if (forestFloor) {
        forestFloor->setView(eye);
    }
    const std::vector<glm::ivec4> slotsBefore = tiles->getCache().getSlots();
    tiles->update(eye);
    const bool tilesChanged = tiles->getCache().getSlots() != slotsBefore;
    // The canopy takes over more trees when the last frame's visible trees came near the proxies' capacity, and
    // hands them back down to its configured band when they are few: the band grows by up to 10 % or shrinks by up
    // to 8 % per frame (where many trees have about the same size on screen, their number changes steeply with
    // it), left alone between 25 and 100 % of the target.
    const float visible = proxies ? float(proxies->readCounter()[0]) : 0.0f;
    const float target = proxies ? 0.8f * float(proxies->getConfig().capacity) : 0.0f;
    TreeRenderSettings values = settings->values().at(0);
    glm::vec2 band(values.canopyBand);
    const glm::vec2 before = band;
    // A list by size keeps itself within its capacity (TreeProxyConfig::bySize).
    if (proxies && !proxies->getConfig().bySize && (visible > target || visible < 0.25f * target)) {
        band = glm::max(band * std::clamp(std::sqrt(std::max(visible, 1.0f) / target), 0.92f, 1.1f), canopyBand);
    }
    // Before a frame showed the trees of every tile, the band cannot be judged yet.
    bandChanged = band != before || !viewedOnce || tilesChanged;
    viewedOnce = true;
    // The canopy also covers the trees below the library copies' cut-off of the last frame, as they fade out.
    glm::vec2 copyBand(0.0f);
    if (forest) {
        const float cut = forest->readCut().cut;
        if (cut > TREE_SIZE_LOG_MIN) {
            copyBand = glm::vec2(std::exp2(cut), std::exp2(cut + TREE_COPY_FADE_WIDTH));
        }
    }
    values.canopyBand = glm::vec4(band, copyBand);
    settings->set(0, values);
    // The nearest trees in full, the next ones reduced (L1).
    // With less detail fewer of them, the farther ones left to the impostors or library copies.
    const size_t activeNear = std::max<size_t>(1, size_t(std::lround(detail * nearTrees)));
    const size_t activeReduced = size_t(std::lround(detail * reducedCount));
    const std::vector<TreeSite> sites =
        nearestTreeSites(terrainParams, placement, eye, uint32_t(activeNear + activeReduced));
    size_t generatedCount = std::min(sites.size(), activeNear);
    if (forest && !sites.empty()) {
        // Within the near radius, sorted by distance as the sites are.
        auto distanceOf = [&](size_t i) { return glm::length(glm::vec3(sites[i].posHeight) - eye); };
        const float reach = distanceOf(generatedCount - 1);
        const float target = sites.size() >= activeNear ? 0.8f * reach : reach;
        nearRadius = nearRadius < 0.0f ? target : std::clamp(target, 0.99f * nearRadius, 1.01f * nearRadius);
        nearRadius = std::min(nearRadius, reach);
        while (generatedCount > 0 && distanceOf(generatedCount - 1) > nearRadius) {
            --generatedCount;
        }
    }
    auto instancesOf = [&](size_t from, size_t active, size_t count, const std::vector<TreeTypeParams>& of) {
        std::vector<TreeInstance> instances;
        for (size_t i = from; i < std::min(sites.size(), from + active); ++i) {
            // The Gaussian renderer's near trees are the library's (TreeGaussianForestConfig::variants).
            instances.push_back(treeInstanceFor(sites[i], of, seed, forest ? config.forest.variants : 0u));
        }
        // Over open land fewer trees are near, and with less detail fewer generated: the rest have scale 0, no tree.
        while (instances.size() < count) {
            TreeInstance hidden{};
            hidden.posScale = glm::vec4(0.0f);
            hidden.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
            instances.push_back(hidden);
        }
        return instances;
    };
    generator->getInstances().set(instancesOf(0, generatedCount, nearTrees, types));
    // The light grid around the near trees: their crowns and a margin of empty voxels, so that what lies outside
    // reads the light of an empty edge.
    if (!sites.empty()) {
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (size_t i = 0; i < generatedCount; ++i) {
            const glm::vec3 foot(sites[i].posHeight);
            const float h = sites[i].posHeight.w;
            lo = glm::min(lo, foot - glm::vec3(0.4f * h, 0.1f * h, 0.4f * h));
            hi = glm::max(hi, foot + glm::vec3(0.4f * h, 1.1f * h, 0.4f * h));
        }
        const glm::vec3 extent = hi - lo;
        const glm::vec3 counts(kFollowGrid);
        // Two voxels of margin on every side.
        const float voxel = std::max(extent.x / (counts.x - 4.0f),
                                     std::max(extent.y / (counts.y - 4.0f), extent.z / (counts.z - 4.0f)));
        TreeRenderSettings grid = settings->values().at(0);
        grid.lightGridMin = glm::vec4(0.5f * (lo + hi) - 0.5f * voxel * counts, voxel);
        settings->set(0, grid);
    }
    if (reducedGenerator) {
        reducedGenerator->getInstances().set(instancesOf(activeNear, activeReduced, reducedCount, reducedTypes));
    }
    if (!proxies) {
        return;
    }
    proxies->setNearRadius(sites.empty() ? -1.0f
                           : forest      ? nearRadius
                                         : glm::length(glm::vec3(sites.back().posHeight) - eye));
}

void BonsaiRenderer::initialize(uint32_t numberPaths) {
    gaussians->initializeBuffers(numberPaths);
    if (proxies) {
        proxies->initialize(numberPaths);
    }
    if (forest) {
        forest->initialize(numberPaths);
    }
}

} // namespace klartraum
