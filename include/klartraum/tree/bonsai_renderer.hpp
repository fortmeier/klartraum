// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_BONSAI_RENDERER_HPP
#define KLARTRAUM_TREE_BONSAI_RENDERER_HPP

#include <algorithm>
#include <memory>
#include <string>

#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageresample.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/scene_mesh_renderer.hpp"
#include "klartraum/tree/tree_gaussians.hpp"
#include "klartraum/tree/tree_generator.hpp"
#include "klartraum/tree/tree_leaf_renderer.hpp"
#include "klartraum/tree/tree_light_volume.hpp"
#include "klartraum/tree/tree_moss_lod.hpp"
#include "klartraum/tree/tree_impostors.hpp"
#include "klartraum/tree/tree_library.hpp"
#include "klartraum/tree/tree_floor.hpp"
#include "klartraum/tree/tree_grass.hpp"
#include "klartraum/tree/tree_proxies.hpp"
#include "klartraum/tree/tree_sky.hpp"
#include "klartraum/tree/tree_stem_renderer.hpp"
#include "klartraum/tree/tree_terrain.hpp"
#include "klartraum/tree/tree_terrain_gaussians.hpp"
#include "klartraum/tree/tree_terrain_renderer.hpp"

namespace klartraum {

enum class BonsaiRenderMode {
    Classical, ///< rasterized trees and pot; the lantern splatted, the two composited by depth
    Gaussian,  ///< everything as Gaussians, sorted and splatted together
};

/**
 * @brief Keeps the frame time near a budget by the detail of the trees (BonsaiRenderer::setDetail(), after Kuth et
 *        al. Sec. 3.9): after each frame, update() scales the detail by the square root of budget / frame time,
 *        at most 25 % per frame, within [minimum, 1].
 */
class TreeDetailBudget {
public:
    explicit TreeDetailBudget(float budgetMs, float minimum = 0.25f)
        : budget(budgetMs),
          minimum(minimum) {}

    /// The detail for the next frame after one that took `frameMs`.
    float update(float frameMs);
    float getDetail() const { return detail; }

private:
    float budget;
    float minimum;
    float detail = 1.0f;
};

struct BonsaiRenderConfig {
    VkExtent2D extent{1280, 720};
    BonsaiRenderMode mode = BonsaiRenderMode::Classical;
    GsplatBackend backend = GsplatBackend::Raster;
    std::string lanternPath = "data/lantern.spz";
    float meshGaussianSpacing = 0.008f; ///< of the pot and soil Gaussians
    float lightVoxel = 0.035f;          ///< voxel size of the light volume (shadows and sky occlusion)
    uint32_t supersample = 2;           ///< the classical renderer draws at this times the resolution
    uint32_t numberPaths = 1;           ///< of the graph the renderer is compiled into
    uint32_t mossPatches = 1500;        ///< patches of the captured moss laid on the soil
    uint32_t mossPool = 1500000;        ///< Gaussians of the moss patches' levels of detail per frame, at most
    /// Pixel diameters below which a moss patch uses its levels 1, 2 and 3, and the crossfade band (TreeMossLod);
    /// twice those of klartraum-spacegame, as ground cover is seen close up and shows its texture.
    glm::vec4 mossLod{320.0f, 96.0f, 24.0f, 0.15f};
    /// Both renderers draw the soil as a TreeTerrain with its level of detail; false draws the soil mesh of the
    /// BonsaiScene instead (classically, or as surfels).
    bool terrainLod = true;
    uint32_t terrainPool = 65536; ///< Gaussians of the terrain per frame, at most (256 per selected node)
    /// When the trees follow the placement (BonsaiScene::placed): its tiles, and the proxies of the trees beyond
    /// the near ones. A range of 0 keeps the tiles in which trees still span the proxies' minPixels at a field of
    /// view of 40 degrees; the nearest tiles come first when the pool is too small.
    TreeTileLayout tiles{64, 512, 64, 0.0f};
    TreeProxyConfig proxies{131072, 3.0f};
    /// When the trees follow the placement: the terrain carries the forest canopy where trees span fewer than
    /// x output pixels, the proxies take over from y (crossfading between); proxies.minPixels and fadeAbove
    /// follow from these.
    glm::vec2 canopyPixels{4.0f, 8.0f};
    /// The classical renderer draws the trees beyond the near ones as impostors (baked from the scene's tree types
    /// when the renderer is made) instead of proxies.
    bool impostors = true;
    TreeImpostorConfig impostorAtlas;
    /// The classical renderer generates this many trees beyond the near ones with one stem level less
    /// (treeReducedType(), the level of detail L1); the impostors start beyond them.
    uint32_t reducedTrees = 96;
    /// The Gaussian renderer draws the trees beyond the near ones as copies of a Gaussian library
    /// (TreeGaussianForest) with this many Gaussians per frame at most.
    TreeGaussianForestConfig forest;
    uint32_t forestPool = 1500000;
    /// The miniature world's haze (treeAtmosphere()) where the soil has relief: how far one sees in its metres,
    /// and how much thicker the haze lies in the valleys (0: not at all).
    float visibilityMeters = 15000.0f;
    float valleyFog = 0.0f;
    /// When the trees follow the placement, both renderers draw blades of grass near the camera; the Gaussian
    /// renderer with this many Gaussians per frame at most.
    bool grass = true;
    TreeGrassConfig grassBlades;
    uint32_t grassPool = 131072;
    /// When the trees follow the placement, both renderers draw copies of the captured moss on the forest floor
    /// near the camera (TreeForestFloor) with this many Gaussians at most, their levels of detail chosen by
    /// floorLod (as mossLod) in output pixels: the copies are near and large, coarser levels keep them in the pool.
    bool forestFloor = true;
    TreeFloorConfig floor;
    uint32_t floorPool = 196608;
    glm::vec4 floorLod{640.0f, 192.0f, 48.0f, 0.15f};
    TreeTerrainLodConfig terrain;
};

/**
 * @brief The compute graph that draws a BonsaiScene into a target, either way.
 *
 * Per frame only the camera, the TreeRenderSettings and the TreeFrame come
 * from the CPU; the trees are generated, their shadows computed in a
 * TreeLightVolume, culled, tessellated or turned into Gaussians, and drawn on
 * the GPU. Compile a graph from getOutput(), then call
 * initialize() once before the first submission.
 */
class BonsaiRenderer {
public:
    /// The light grid's voxels per axis when it follows the near trees (scenes whose trees follow the placement).
    static constexpr glm::uvec3 kFollowGrid{48u, 32u, 48u};

    BonsaiRenderer(VulkanContext& vulkanContext, const BonsaiScene& scene, std::shared_ptr<ImageViewSrc> target,
                   const BonsaiRenderConfig& config);

    ComputeGraphElementPtr getOutput() const { return output; }
    void initialize(uint32_t numberPaths);

    /// Sets the season S in [0, 4) (0 winter … 3 fall) of the next frames: the leaves and the snow cover.
    void setSeason(float season);

    /**
     * @brief Sets everything that changes over time for the next frame.
     *
     * `seconds` drives the wind and the lantern's flicker, `timeOfDay` (0
     * midnight, 0.5 noon) and `season` the sky (treeSkyAt()) and the trees;
     * `wind` is the direction the wind blows to (xyz) and its strength (w).
     */
    void setTime(float seconds, float timeOfDay, float season, glm::vec4 wind);

    /**
     * @brief Tells the renderer where the camera is for the next frame, when the trees follow the placement:
     *        the tiles around it, and the scene's nearTrees nearest trees generated in full (the others as
     *        proxies). Does nothing for a planting.
     */
    void setView(glm::vec3 eye);
    /// The share in [0.25, 1] of the near and reduced trees generated (the others become impostors or library
    /// copies) from the next setView() on; see TreeDetailBudget.
    void setDetail(float detail) { this->detail = std::clamp(detail, 0.25f, 1.0f); }
    float getDetail() const { return detail; }
    /// The generator of the reduced trees (L1), or nullptr.
    TreeGenerator* getReducedGenerator() { return reducedGenerator.get(); }
    /// Tiles in range of the last setView() still to be generated (they come a few per frame); 0 for a planting.
    uint32_t getPendingTiles() const { return tiles ? tiles->getCache().getMissing() : 0u; }
    /// Whether the last setView() found nothing more to do: all tiles there and the canopy's band steady. A still
    /// submits and calls setView() again until it is.
    bool isSettled() const { return getPendingTiles() == 0 && !bandChanged; }

    std::shared_ptr<CameraUboType> getCamera() const { return camera; }
    HostValues<TreeRenderSettings>& getSettings() { return *settings; }
    TreeGenerator& getGenerator() { return *generator; }
    TreeStemRenderer* getStems() { return stems.get(); }
    TreeLeafRenderer* getLeaves() { return leaves.get(); }
    TreeLightVolume& getLightVolume() { return *lightVolume; }
    /// The terrain, or nullptr.
    TreeTerrain* getTerrain() { return terrain.get(); }
    /// The terrain's Gaussians of the Gaussian renderer, or nullptr.
    TreeTerrainGaussians* getTerrainGaussians() { return terrainGaussians.get(); }
    /// The placement's tiles and proxies, or nullptr for a planting.
    TreePlacementTiles* getTiles() { return tiles.get(); }
    TreeProxies* getProxies() { return proxies.get(); }
    /// The grass near the camera, or nullptr.
    TreeGrass* getGrass() { return grass.get(); }
    /// The grass's Gaussians in the Gaussian renderer, or nullptr.
    TreeGrassGaussians* getGrassGaussians() { return grassGaussians.get(); }
    /// The moss on the forest floor near the camera, or nullptr.
    TreeForestFloor* getForestFloor() { return forestFloor.get(); }
    /// The Gaussian renderer's library copies of the trees, or nullptr.
    TreeGaussianForest* getForest() { return forest.get(); }

    /// The lantern capture, cropped to the lantern and its moss and placed on the soil.
    static std::vector<Gaussian3D> loadLantern(const BonsaiScene& scene, const std::string& path);

    /// The level of detail of the moss, or nullptr without moss.
    TreeMossLod* getMoss() { return moss.get(); }

private:
    BonsaiRenderConfig config;
    glm::vec3 lanternFlame{0.0f};
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::shared_ptr<TreeGenerator> generator;
    std::unique_ptr<TreeLightVolume> lightVolume;

    // Captured lantern and moss (classical) or everything (Gaussian) as Gaussians
    std::unique_ptr<TreeGaussians> gaussians;
    std::unique_ptr<TreeMossLod> moss;
    std::unique_ptr<TreeTerrainGaussians> terrainGaussians;
    std::unique_ptr<TreePlacementTiles> tiles;
    std::unique_ptr<TreeProxies> proxies;
    std::unique_ptr<TreeGrass> grass;
    std::unique_ptr<TreeGrassGaussians> grassGaussians;
    std::unique_ptr<TreeForestFloor> forestFloor;
    std::unique_ptr<TreeGaussianForest> forest;
    std::unique_ptr<TreeImpostorAtlas> atlas;
    std::shared_ptr<TreeGenerator> reducedGenerator;
    std::unique_ptr<TreeStemRenderer> reducedStems;
    std::unique_ptr<TreeLeafRenderer> reducedLeaves;
    std::vector<TreeTypeParams> reducedTypes;
    uint32_t reducedCount = 0;
    float detail = 1.0f;
    glm::vec2 canopyBand{0.0f}; ///< the configured band in rendered pixels; the settings' follows the visible trees
    bool bandChanged = false;
    bool viewedOnce = false;
    std::unique_ptr<TreeImpostorRenderer> impostors;
    // The placement, its seed and the tree types, for setView().
    TreeTerrainParams terrainParams{};
    TreePlacementParams placement{};
    std::vector<TreeTypeParams> types;
    uint32_t seed = 0;
    uint32_t nearTrees = 0;

    // Classical
    std::unique_ptr<TreeStemRenderer> stems;
    std::unique_ptr<TreeLeafRenderer> leaves;
    std::unique_ptr<SceneMeshRenderer> meshes;
    std::unique_ptr<TreeTerrain> terrain;
    std::unique_ptr<TreeTerrainRenderer> terrainRenderer;
    std::shared_ptr<RenderPass> pass;

    ComputeGraphElementPtr output;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_BONSAI_RENDERER_HPP
