// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - scaleAnchors: the pot holds 3.6 m at scale 0 and 8 km at 1, growing steadily; the lantern keeps its size in
 *   the pot up to kForest and shrinks beyond; one tree at 0, a planting at kForest; tree size and style continuous
 * - smallestScaleIsOneBonsai: at scale 0 the scene has one bonsai-styled tree filling the pot and the lantern as
 *   captured
 * - scaleKeepsTheForestDensity: at larger scales the tree count follows 650 trees per hectare of soil
 * - treesStandInThePot: 8, 13 and 20 trees stand on the soil inside the pot, apart from each other and from the
 *   lantern
 * - speciesAndStyleChooseTheTrees: a planting of pines has only pines; style 1 at the forest scale gives the
 *   bonsai-styled types (the pine with its needle pads), style 0 the natural ones, whatever the scale says
 * - treesAreFourToNineLanternsTall: the generated trees are 4 to 9 times as tall as the lantern
 * - largeScalesFollowThePlacement: beyond a planting of 60 trees the scene's trees are the near trees of the
 *   placement; a planting of more than 60 trees asked for explicitly is refused
 * - treesStandInTheForest: in a landscape with open land and rock, every tree stands where the land cover is mostly
 *   forest
 * - mossCoversPartOfTheSoil: moss grows on 30 to 80 % of the soil in patches a few centimetres wide, greener
 *   than the bare soil, and fading the fine detail keeps the mean colour but smooths it
 * - soilLiesInThePot: the soil rises from below the rim at the pot's edge towards the middle
 * - potWallsFaceOutwards: the mesh is a triangle list; outer wall normals point away from the middle, the
 *   bottom's down, the soil's up
 * - meshSurfelsCoverTheMesh: sampleMeshSurfels places about area / spacing^2 samples on the triangles, with
 *   their normal and spacing
 * - mossPatchesAreGreenSectors: a synthetic ring of green and brown ground gives patches only of its green
 *   sectors, centred at the origin with their ground at 0 and faded towards the cuts
 * - mossPatchesFromTheCapture: the lantern capture gives several mostly green patches
 * - scatteredMossLiesOnTheSoil: placed copies of patches lie inside the pot, away from the lantern, on the plane
 *   touching the soil, at their height above it
 * - mossGrowsOnTheForestFloor: in a landscape with open land and rock, every copy lies where the land cover is
 *   mostly forest
 * - capturedGaussiansTakeSnow: treeGaussianNormal is the thinnest axis turned up; treeCapturedColor covers what
 *   faces up with snow in winter only and scales by tint and shade
 * - lanternStandsOnTheSoil: the placed lantern capture's ground lies on the soil, its moss inside the pot
 * - capturedPotIsMeasured: a fitted pot mesh (tapered body on a saucer, a lid over its opening, a guessed bulge
 *   underneath, turned about +y) loads with its rim height, foot, rim rectangle and turn found
 * - capturedPotReplacesTheProceduralOne: with that pot the scene has no procedural pot; the captured one stands
 *   on y = 0 with its rim's long axis along x and the procedural length, nothing of it above its rim or below its
 *   foot, and the soil inside it; an object placed on the soil stands on it at its height
 * - classicalGeometryIsHiddenBehindSplats: a quad behind an opaque block of splats is hidden by it, a quad in
 *   front covers it, with the splats' depth output
 * - supersamplingSmoothsEdges: the classical render at 2x supersampling differs less between neighbouring pixels
 *   than without, at the same mean brightness
 * - nightIsDarkerAndTheLanternGlows: at midnight the scene is darker than at noon, and the moss next to the
 *   lantern is lit warmer than at noon, in both renderers
 * - bonsaiRendersBothWays: the scene renders classically and as Gaussians, written to build/TestingOutput/tree/
 * - backgroundFollowsTheFrame: with a background, both renderers draw over the colour set for the frame: the
 *   frame's corners are white, then black after setBackground() between submissions, and the scene itself (where
 *   the black frame is lit) is the same on both
 * - potOfGaussiansRendersBothWays: with a captured pot of Gaussians and no mesh object, which leaves no scene
 *   mesh besides the soil, the scene renders both ways and the pot covers the frame's lower part
 * - mossReceivesObjectShadows: in the morning sun, the soil and the moss where a captured object's shadow falls
 *   (clear of the trees and the lantern) are darker than without the object, the brightest pixel there too (no
 *   unshadowed moss)
 * - capturedObjectsShadowThemselves: a grey ball, as a captured mesh and as captured Gaussians, seen across the
 *   morning sun: the side towards the sun keeps its captured colour under the light's tint, evenly (no speckles),
 *   the side away from it is clearly darker
 * - capturedObjectsShadowEachOther: a ball between the sun and another ball darkens the other's sunny side
 * - movingObjectsCastShadowsWhereTheyAre: a moving object's shadow on the soil appears where its transform puts it,
 *   follows it when it moves, and is gone when its transform has scale 0
 * - litMorphShadesLikeTheScene: a MeshMorph drawn into the bonsai frame with the scene's light, its sphere a moving
 *   occluder at the same place: the side towards the morning sun keeps the mesh's colour under the light's tint,
 *   the side away from it is clearly darker
 * - potShadowsItsOwnFarSide: the light a captured pot's walls read (treeCapturedLightAt()): the wall facing away from
 *   the morning sun lies in the pot's own shadow, through its soil, whatever the trees do; the sunny wall is lit
 * - treeShadowsStayOffAPotsShadowSide: the wall of a captured pot facing away from the morning sun looks the same with
 *   the tree and without it, although the tree's shadow reaches it: in the pot's own shadow there is no sun left for
 *   the tree to block
 * - shadowsMoveSmoothlyWithTheSun: as the morning sun moves on in small steps, no pixel of a captured ball and the
 *   soil around it changes by much from one step to the next (no shadow edges hopping between voxels)
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/raster/mesh.hpp"
#include "klartraum/raster/mesh_morph.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/tree_placement.hpp"
#include "klartraum/tree/tree_sky.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

constexpr float kLanternHeight = 0.28f;

TreeSceneVertex vertex(glm::vec3 p, glm::vec3 n, glm::vec3 color) {
    TreeSceneVertex v{};
    v.pos = glm::vec4(p, 1.0f);
    v.normal = glm::vec4(n, 0.0f);
    v.color = glm::vec4(color, TREE_MATERIAL_PLAIN);
    return v;
}

// An axis-aligned quad in the plane z, facing +z.
void addQuad(std::vector<TreeSceneVertex>& mesh, float x0, float x1, float y0, float y1, float z, glm::vec3 color) {
    const glm::vec3 n(0.0f, 0.0f, 1.0f);
    const TreeSceneVertex a = vertex({x0, y0, z}, n, color), b = vertex({x1, y0, z}, n, color),
                          c = vertex({x1, y1, z}, n, color), d = vertex({x0, y1, z}, n, color);
    mesh.insert(mesh.end(), {a, b, c, a, c, d});
}

// A fitted pot mesh as mesh_texture_fitting_example would write it, turned by `yaw` about +y: rings of 128 points
// at heights y with rectangular outlines of half sizes (x, z), from a guessed bulge underneath over a saucer and a
// body widening to the rim at y = 0.5 to a lid over the opening; the rim's half sizes are 1.0 x 0.6.
std::filesystem::path writeFittedPot(float yaw) {
    std::vector<glm::vec3> levels; // y, half length, half width
    levels.push_back({-0.2f, 0.2f, 0.15f});
    levels.push_back({-0.1f, 0.4f, 0.3f});
    for (int k = 0; k <= 4; ++k)
        levels.push_back({0.0125f * float(k), 0.95f, 0.65f});
    for (int k = 0; k <= 36; ++k) {
        const float t = float(k) / 36.0f;
        levels.push_back({0.05f + 0.45f * t, 0.8f + 0.2f * t, 0.52f + 0.08f * t});
    }
    for (int k = 1; k <= 12; ++k) {
        const float t = float(k) / 12.0f;
        levels.push_back({0.5f + 0.3f * t, 0.9f * (1.0f - t) + 0.05f * t, 0.55f * (1.0f - t) + 0.03f * t});
    }
    raster::TriangleMesh mesh;
    const int around = 128;
    const glm::mat4 turn = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    for (const glm::vec3& level : levels)
        for (int i = 0; i < around; ++i) {
            const float angle = 2.0f * 3.14159265f * float(i) / float(around);
            const glm::vec2 d(std::cos(angle), std::sin(angle));
            const float r = 1.0f / std::max(std::abs(d.x) / level.y, std::abs(d.y) / level.z);
            mesh.positions.push_back(glm::vec3(turn * glm::vec4(r * d.x, level.x, r * d.y, 1.0f)));
        }
    for (uint32_t l = 0; l + 1 < levels.size(); ++l)
        for (uint32_t i = 0; i < uint32_t(around); ++i) {
            const uint32_t a = l * around + i, b = l * around + (i + 1) % around;
            for (const glm::uvec3 t : {glm::uvec3(a, a + around, b), glm::uvec3(b, a + around, b + around)}) {
                mesh.triangles.push_back(t);
                mesh.cornerUvs.insert(mesh.cornerUvs.end(), 3, glm::vec2(0.0f));
            }
        }
    const std::filesystem::path path = "build/TestingOutput/bonsai_scene/fitted_pot.obj";
    std::filesystem::create_directories(path.parent_path());
    raster::writeObj(mesh, path);
    return path;
}

} // namespace

TEST(BonsaiScene, scaleAnchors) {
    EXPECT_NEAR(BonsaiScale{0.0f}.regionMeters(), 3.6f, 1e-4f);
    EXPECT_NEAR(BonsaiScale{1.0f}.regionMeters(), 8000.0f, 1.0f);
    float region = 0.0f, previousHeight = BonsaiScale{0.0f}.treeHeightUnits();
    uint32_t trees = 0;
    const float lanternAtForest = BonsaiScale{BonsaiScale::kForest}.lanternUnits();
    for (int i = 0; i <= 100; ++i) {
        const float s = float(i) / 100.0f;
        const BonsaiScale scale{s};
        EXPECT_GT(scale.regionMeters(), region);
        region = scale.regionMeters();
        if (s <= BonsaiScale::kForest) {
            EXPECT_NEAR(scale.lanternUnits(), 0.28f, 0.01f) << s;
        } else {
            EXPECT_LT(scale.lanternUnits(), lanternAtForest) << s;
        }
        const uint32_t count = scale.treeCount(4.47f);
        EXPECT_GE(count, trees) << s;
        trees = count;
        // Steady in steps of 0.01 (faster just after kForest, where the trees grow to their full height).
        EXPECT_LT(std::abs(scale.treeHeightUnits() - previousHeight), 0.15f) << s;
        previousHeight = scale.treeHeightUnits();
    }
    // Continuous at kForest.
    const float atForest = BonsaiScale{BonsaiScale::kForest}.treeHeightUnits();
    EXPECT_NEAR(BonsaiScale{BonsaiScale::kForest + 1e-4f}.treeHeightUnits(), atForest, 2e-3f);
    EXPECT_NEAR(BonsaiScale{BonsaiScale::kForest - 1e-4f}.treeHeightUnits(), atForest, 2e-3f);
    EXPECT_LT(BonsaiScale{1.0f}.lanternUnits(), 0.001f);
    EXPECT_EQ(BonsaiScale{0.0f}.treeCount(4.47f), 1u);
    EXPECT_NEAR(float(BonsaiScale{BonsaiScale::kForest}.treeCount(4.47f)), 12.0f, 1.0f);
    EXPECT_EQ(BonsaiScale{0.0f}.bonsaiStyle(), 1.0f);
    EXPECT_EQ(BonsaiScale{BonsaiScale::kForest}.bonsaiStyle(), 0.0f);
    EXPECT_NEAR(BonsaiScale{0.0f}.treeHeightUnits(), 2.0f, 1e-5f);
    EXPECT_NEAR(BonsaiScale{1.0f}.treeHeightMeters(), 25.0f, 1e-3f);
}

TEST(BonsaiScene, smallestScaleIsOneBonsai) {
    BonsaiSceneConfig config;
    config.scale.s = 0.0f;
    const BonsaiScene scene = makeBonsaiScene(config);
    ASSERT_EQ(scene.trees.size(), 1u);
    EXPECT_NEAR(scene.lanternScale, 1.0f, 0.01f);
    const TreeTypeParams natural = tree_presets::bonsaiBroadleaf();
    const TreeTypeParams& bonsai = scene.types[0];
    EXPECT_GT(bonsai.ratios.y, 2.0f * natural.ratios.y);  // a thick trunk
    EXPECT_GT(bonsai.trunk.y, natural.trunk.y);           // spreading roots
    EXPECT_GT(bonsai.nDownAngle.y, natural.nDownAngle.y); // flatter branches
    // It fills the pot: about 2 units tall.
    const TreeRecords records = generateTreesReference(scene.types, scene.trees, defaultTreeFrame());
    float top = 0.0f;
    for (const SegmentRecord& d : records.segments) {
        top = std::max(top, d.toPosRadius.y);
    }
    EXPECT_GT(top - scene.trees[0].posScale.y, 1.4f);
    EXPECT_LT(top - scene.trees[0].posScale.y, 2.6f);
}

TEST(BonsaiScene, scaleKeepsTheForestDensity) {
    BonsaiSceneConfig config;
    config.scale.s = 0.3f;
    const BonsaiScene scene = makeBonsaiScene(config);
    const float squareMeters = scene.soilArea * std::pow(config.scale.metersPerUnit(), 2.0f);
    const float expected = BonsaiScale::kTreesPerHectare * squareMeters / 1e4f;
    // The placement keeps trees apart and away from the lantern; nearly all fit.
    EXPECT_GT(float(scene.trees.size()), 0.9f * expected);
    EXPECT_LE(float(scene.trees.size()), expected + 1.0f);
}

TEST(BonsaiScene, treesStandInThePot) {
    for (uint32_t count : {8u, 13u, 20u}) {
        BonsaiSceneConfig config;
        config.treeCount = count;
        const BonsaiScene scene = makeBonsaiScene(config);
        ASSERT_EQ(scene.trees.size(), count);
        const glm::vec2 lantern(config.lanternPosition.x, config.lanternPosition.z);
        for (size_t i = 0; i < scene.trees.size(); ++i) {
            const glm::vec4 p = scene.trees[i].posScale;
            EXPECT_LT(scene.potMeasure(p.x, p.z, 0.9f), 1.0f) << i;
            EXPECT_NEAR(p.y, scene.soilHeight(p.x, p.z) - 0.01f, 1e-5f) << i;
            EXPECT_GE(glm::length(glm::vec2(p.x, p.z) - lantern), 0.38f) << i;
            for (size_t j = 0; j < i; ++j) {
                const glm::vec4 q = scene.trees[j].posScale;
                EXPECT_GE(glm::length(glm::vec2(p.x - q.x, p.z - q.z)), 0.3f - 1e-5f) << i << " " << j;
            }
        }
    }
}

TEST(BonsaiScene, speciesAndStyleChooseTheTrees) {
    BonsaiSceneConfig config;
    config.treeCount = 8;
    config.species = BonsaiSpecies::Pine;
    config.style = 1.0f;
    const BonsaiScene pines = makeBonsaiScene(config);
    for (const TreeInstance& tree : pines.trees)
        EXPECT_EQ(tree.ids.x, 1u);
    const TreeTypeParams padded = tree_presets::bonsaiPinePads(tree_presets::bonsaiStyled(tree_presets::bonsaiPine()));
    EXPECT_NEAR(pines.types[1].ratios.y, padded.ratios.y, 1e-6f); // the thick trunk
    EXPECT_NEAR(pines.types[1].leafCount.x, padded.leafCount.x, 1e-3f);
    config.style = 0.0f;
    config.species = BonsaiSpecies::Broadleaf;
    const BonsaiScene natural = makeBonsaiScene(config);
    for (const TreeInstance& tree : natural.trees)
        EXPECT_EQ(tree.ids.x, 0u);
    EXPECT_NEAR(natural.types[1].ratios.y, tree_presets::bonsaiPine().ratios.y, 1e-6f);
}

TEST(BonsaiScene, treesAreFourToNineLanternsTall) {
    const BonsaiScene scene = makeBonsaiScene();
    const TreeRecords records = generateTreesReference(scene.types, scene.trees, defaultTreeFrame());
    std::vector<float> top(scene.trees.size(), 0.0f);
    for (const SegmentRecord& d : records.segments) {
        top[d.ids.z] = std::max(top[d.ids.z], d.toPosRadius.y);
    }
    for (size_t i = 0; i < top.size(); ++i) {
        const float height = top[i] - scene.trees[i].posScale.y;
        EXPECT_GE(height / kLanternHeight, 4.0f) << i;
        EXPECT_LE(height / kLanternHeight, 9.0f) << i;
    }
}

TEST(BonsaiScene, largeScalesFollowThePlacement) {
    BonsaiSceneConfig config;
    config.scale.s = 0.75f;
    const BonsaiScene scene = makeBonsaiScene(config);
    EXPECT_TRUE(scene.placed);
    ASSERT_EQ(scene.trees.size(), config.nearTrees);
    const glm::vec3 middle(0.0f, scene.soilHeight(0.0f, 0.0f), 0.0f);
    const std::vector<TreeSite> near = nearestTreeSites(scene.terrain, scene.placement, middle, config.nearTrees);
    for (size_t i = 0; i < near.size(); ++i) {
        EXPECT_EQ(glm::vec3(scene.trees[i].posScale), glm::vec3(near[i].posHeight));
    }
    EXPECT_FALSE(makeBonsaiScene(BonsaiSceneConfig{}).placed);
    BonsaiSceneConfig tooMany;
    tooMany.treeCount = 61;
    EXPECT_THROW(makeBonsaiScene(tooMany), std::invalid_argument);
}

TEST(BonsaiScene, treesStandInTheForest) {
    BonsaiSceneConfig config;
    config.scale.s = 1.0f;
    config.treeCount = 40;
    const BonsaiScene scene = makeBonsaiScene(config);
    EXPECT_EQ(scene.trees.size(), 40u);
    for (const TreeInstance& t : scene.trees) {
        EXPECT_GT(tree::treeLandCover(scene.terrain, t.posScale.x, t.posScale.z).vegetation.x, 0.5f);
    }
}

TEST(BonsaiScene, mossCoversPartOfTheSoil) {
    const glm::vec3 base(0.075f, 0.11f, 0.025f);
    int moss = 0, samples = 0;
    glm::vec3 meanFine(0.0f), meanSmooth(0.0f), mossColour(0.0f), soilColour(0.0f);
    for (float x = -1.0f; x < 1.0f; x += 0.01f) {
        for (float z = -0.6f; z < 0.6f; z += 0.01f) {
            const glm::vec3 p(x, 0.3f, z);
            const float coverage = tree::treeMossCoverage(p);
            moss += coverage > 0.5f ? 1 : 0;
            ++samples;
            const glm::vec3 a = tree::treeMossAlbedo(base, p, 1.0f);
            meanFine += a;
            meanSmooth += tree::treeMossAlbedo(base, p, 0.0f);
            mossColour += coverage > 0.95f ? a : glm::vec3(0.0f);
            soilColour += coverage < 0.05f ? a : glm::vec3(0.0f);
        }
    }
    const float fraction = float(moss) / float(samples);
    EXPECT_GT(fraction, 0.3f);
    EXPECT_LT(fraction, 0.8f);
    EXPECT_NEAR(meanFine.g / meanSmooth.g, 1.0f, 0.05f);
    // Moss is greener than bare soil.
    EXPECT_GT(mossColour.g / mossColour.r, 1.2f * soilColour.g / soilColour.r);

    // Along a densely sampled line the fine structure (cushions, grit) changes the colour between neighbouring
    // samples; without detail it is smooth.
    double fine = 0.0, smooth = 0.0;
    glm::vec3 previousFine(0.0f), previousSmooth(0.0f);
    for (int i = 0; i <= 4000; ++i) {
        const glm::vec3 p(-1.0f + 0.0005f * float(i), 0.3f, 0.1f);
        const glm::vec3 a = tree::treeMossAlbedo(base, p, 1.0f);
        const glm::vec3 b = tree::treeMossAlbedo(base, p, 0.0f);
        if (i > 0) {
            fine += glm::length(a - previousFine);
            smooth += glm::length(b - previousSmooth);
        }
        previousFine = a;
        previousSmooth = b;
    }
    EXPECT_LT(smooth, 0.7 * fine);
}

TEST(BonsaiScene, soilLiesInThePot) {
    const BonsaiScene scene = makeBonsaiScene();
    const BonsaiPot& pot = scene.config.pot;
    // At the inner rim the soil is below the rim; in the middle it is mounded.
    EXPECT_LT(scene.soilHeight(pot.halfLength - pot.wall, 0.0f), pot.height);
    EXPECT_NEAR(scene.soilHeight(0.0f, 0.0f), pot.soilHeight + pot.soilMound, 0.01f);
    EXPECT_GT(scene.soilHeight(0.0f, 0.0f), scene.soilHeight(1.0f, 0.4f));
}

TEST(BonsaiScene, potWallsFaceOutwards) {
    const BonsaiScene scene = makeBonsaiScene();
    ASSERT_EQ(scene.mesh.size() % 3, 0u);
    size_t outerWall = 0, soil = 0;
    for (const TreeSceneVertex& v : scene.mesh) {
        const glm::vec3 p(v.pos), n(v.normal);
        EXPECT_NEAR(glm::length(n), 1.0f, 1e-4f);
        if (v.color.w == TREE_MATERIAL_GLAZE && p.y == 0.0f && std::abs(n.y) > 0.5f) {
            EXPECT_LT(n.y, -0.99f); // the bottom
        } else if (v.color.w == TREE_MATERIAL_GLAZE && p.y < 0.5f * scene.config.pot.height) {
            EXPECT_GT(glm::dot(n, glm::vec3(p.x, 0.0f, p.z)), 0.0f);
            ++outerWall;
        }
        if (v.color.w == TREE_MATERIAL_MOSS) {
            EXPECT_GT(n.y, 0.8f);
            ++soil;
        }
    }
    EXPECT_GT(outerWall, 100u);
    EXPECT_GT(soil, 100u);
}

TEST(BonsaiScene, meshSurfelsCoverTheMesh) {
    std::vector<TreeSceneVertex> mesh;
    addQuad(mesh, 0.0f, 2.0f, 0.0f, 1.0f, 0.5f, glm::vec3(0.5f));
    const float spacing = 0.02f;
    const auto surfels = sampleMeshSurfels(mesh, spacing, 3);
    EXPECT_NEAR(double(surfels.size()), 2.0 / (spacing * spacing), 0.02 * 2.0 / (spacing * spacing));
    for (const TreeSceneVertex& v : surfels) {
        ASSERT_NEAR(v.pos.z, 0.5f, 1e-5f);
        ASSERT_GE(v.pos.x, 0.0f);
        ASSERT_LE(v.pos.x, 2.0f);
        ASSERT_GE(v.pos.y, 0.0f);
        ASSERT_LE(v.pos.y, 1.0f);
        ASSERT_EQ(v.pos.w, spacing);
        ASSERT_NEAR(v.normal.z, 1.0f, 1e-5f);
    }
}

namespace {

// A flat Gaussian of the given colour (degree 0 coefficients) at (x, y, z).
Gaussian3D groundGaussian(float x, float y, float z, glm::vec3 color) {
    Gaussian3D g{};
    g.position = {x, y, z};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.scale = {0.004f, 0.0005f, 0.004f};
    g.color = {color.x, color.y, color.z};
    g.alpha = 1.0f;
    return g;
}

} // namespace

TEST(BonsaiScene, mossPatchesAreGreenSectors) {
    // Ground at y = 0.01 around the origin: green for x < 0, brown for x > 0.
    std::vector<Gaussian3D> capture;
    for (float x = -0.25f; x <= 0.25f; x += 0.004f) {
        for (float z = -0.25f; z <= 0.25f; z += 0.004f) {
            const glm::vec3 color = x < 0.0f ? glm::vec3(-0.5f, 0.5f, -0.8f) : glm::vec3(0.4f, 0.0f, -0.6f);
            capture.push_back(groundGaussian(x, 0.01f, z, color));
        }
    }
    capture.push_back(groundGaussian(0.18f, 0.2f, 0.0f, glm::vec3(1.0f))); // too high: the lantern
    const std::vector<MossPatch> patches = extractMossPatches(capture, 12);
    // Of the 12 sectors of 30 degrees, those within 90 to 270 degrees are green.
    ASSERT_GE(patches.size(), 4u);
    ASSERT_LE(patches.size(), 6u);
    for (const MossPatch& patch : patches) {
        glm::vec3 mean(0.0f);
        float minAlpha = 1.0f, maxAlpha = 0.0f;
        for (const Gaussian3D& g : patch) {
            mean += glm::vec3(g.position[0], g.position[1], g.position[2]);
            EXPECT_GT(g.color[1], g.color[0]);
            minAlpha = std::min(minAlpha, g.alpha);
            maxAlpha = std::max(maxAlpha, g.alpha);
        }
        mean /= float(patch.size());
        EXPECT_NEAR(mean.x, 0.0f, 1e-3f);
        EXPECT_NEAR(mean.z, 0.0f, 1e-3f);
        EXPECT_NEAR(mean.y, 0.0f, 1e-5f);
        EXPECT_EQ(maxAlpha, 1.0f);
        EXPECT_LT(minAlpha, 0.3f); // faded at the cuts
    }
}

TEST(BonsaiScene, mossPatchesFromTheCapture) {
    const std::vector<MossPatch> patches = extractMossPatches(loadGaussiansSpz("data/lantern.spz", true));
    EXPECT_GE(patches.size(), 3u);
    for (const MossPatch& patch : patches) {
        EXPECT_GT(patch.size(), 500u);
        size_t green = 0;
        for (const Gaussian3D& g : patch) {
            green += g.color[1] > g.color[0] && g.color[1] > g.color[2] ? 1 : 0;
        }
        EXPECT_GE(float(green), 0.6f * float(patch.size()));
    }
}

TEST(BonsaiScene, scatteredMossLiesOnTheSoil) {
    const BonsaiScene scene = makeBonsaiScene();
    MossPatch patch;
    for (float x = -0.04f; x <= 0.04f; x += 0.01f) {
        patch.push_back(groundGaussian(x, 0.003f, 0.0f, glm::vec3(0.0f, 0.5f, 0.0f)));
    }
    const std::vector<TreeLodInstance> copies = placeMossPatches(scene, 1, 50, 3, 1.0f);
    ASSERT_EQ(copies.size(), 50u);
    const std::vector<Gaussian3D> moss = scatterMossPatches({patch}, copies);
    EXPECT_EQ(moss.size(), 50u * patch.size());
    const glm::vec2 lantern(scene.lanternTranslation.x, scene.lanternTranslation.z);
    for (const Gaussian3D& g : moss) {
        const glm::vec3 p(g.position[0], g.position[1], g.position[2]);
        EXPECT_LT(scene.potMeasure(p.x, p.z, 0.95f), 1.0f);
        EXPECT_GT(glm::length(glm::vec2(p.x, p.z) - lantern), 0.25f);
        // On the plane that touches the soil at the copy's centre: as high above the soil as above the patch's
        // ground, within the soil's curvature over the patch.
        EXPECT_NEAR(p.y - scene.soilHeight(p.x, p.z), 0.003f + 0.002f, 1e-3f);
    }
}

TEST(BonsaiScene, mossGrowsOnTheForestFloor) {
    BonsaiSceneConfig config;
    config.scale.s = 1.0f;
    config.treeCount = 12;
    const BonsaiScene scene = makeBonsaiScene(config);
    const std::vector<TreeLodInstance> copies = placeMossPatches(scene, 4, 2000, 5, 0.01f);
    EXPECT_EQ(copies.size(), 2000u);
    for (const TreeLodInstance& c : copies) {
        EXPECT_GE(tree::treeLandCover(scene.terrain, c.posYaw.x, c.posYaw.z).vegetation.x, 0.5f);
    }
}

TEST(BonsaiScene, capturedGaussiansTakeSnow) {
    // Thinnest axis y, turned upside down: the normal points up again.
    const glm::vec3 n = tree::treeGaussianNormal(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(0.1f, 0.01f, 0.1f));
    EXPECT_NEAR(n.y, 1.0f, 1e-6f);
    EXPECT_NEAR(tree::treeGaussianNormal(glm::vec4(0, 0, 0, 1), glm::vec3(0.01f, 0.1f, 0.1f)).x, 1.0f, 1e-6f);

    TreeRenderSettings s = defaultTreeRenderSettings({64, 64});
    const glm::vec3 green(0.2f, 0.6f, 0.2f);
    const glm::vec3 up(0.0f, 1.0f, 0.0f), p(0.1f, 0.3f, 0.2f);
    s.season.y = 0.0f;
    EXPECT_NEAR(glm::length(tree::treeCapturedColor(green, up, p, 0.5f, s) - 0.5f * green), 0.0f, 1e-6f);
    s.season.y = 1.0f;
    const glm::vec3 snowy = tree::treeCapturedColor(green, up, p, 1.0f, s);
    EXPECT_GT(snowy.r, 0.8f);
    const glm::vec3 side = tree::treeCapturedColor(green, glm::vec3(1.0f, 0.0f, 0.0f), p, 1.0f, s);
    EXPECT_NEAR(glm::length(side - green), 0.0f, 1e-6f);
}

TEST(BonsaiScene, lanternStandsOnTheSoil) {
    const BonsaiScene scene = makeBonsaiScene();
    const std::vector<Gaussian3D> lantern = BonsaiRenderer::loadLantern(scene, "data/lantern.spz");
    ASSERT_GT(lantern.size(), 10000u);
    std::vector<float> heights;
    for (const Gaussian3D& g : lantern) {
        const glm::vec3 p(g.position[0], g.position[1], g.position[2]);
        EXPECT_LT(scene.potMeasure(p.x, p.z, 0.95f), 1.0f);
        const float r =
            glm::length(glm::vec2(p.x, p.z) - glm::vec2(scene.lanternTranslation.x, scene.lanternTranslation.z));
        if (r > 0.12f) {
            heights.push_back(p.y - scene.soilHeight(p.x, p.z));
        }
    }
    // The moss around the lantern lies on the soil, within the capture's own unevenness.
    std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
    EXPECT_NEAR(heights[heights.size() / 2], 0.0f, 0.02f);
}

TEST(BonsaiScene, capturedPotIsMeasured) {
    const BonsaiObject pot = loadBonsaiObject(writeFittedPot(0.4f).string());
    // Slices are a fortieth of the height (-0.2 to 0.8) apart.
    EXPECT_NEAR(pot.rimHeight, 0.5f, 0.03f);
    EXPECT_NEAR(pot.footHeight, 0.0f, 0.03f);
    EXPECT_NEAR(pot.rim.x, 1.0f, 0.03f);
    EXPECT_NEAR(pot.rim.y, 0.6f, 0.03f);
    EXPECT_NEAR(std::sin(pot.rim.z - 0.4f), 0.0f, 0.03f); // the long axis, either way along it
    EXPECT_NEAR(glm::length(pot.rimCenter), 0.0f, 0.02f);
    EXPECT_FALSE(pot.mesh.empty());
    EXPECT_TRUE(pot.gaussians.empty());
}

TEST(BonsaiScene, capturedPotReplacesTheProceduralOne) {
    auto pot = std::make_shared<BonsaiObject>(loadBonsaiObject(writeFittedPot(0.4f).string()));
    BonsaiSceneConfig config;
    config.potObject = pot;
    config.objects.push_back({pot, glm::vec2(-0.5f, 0.2f), 0.3f, 0.0f});
    const BonsaiScene scene = makeBonsaiScene(config);
    const float potScale = config.pot.halfLength / pot->rim.x;
    EXPECT_NEAR(scene.config.pot.halfLength, config.pot.halfLength, 1e-6f);
    EXPECT_NEAR(scene.config.pot.halfWidth, pot->rim.y * potScale, 1e-4f);
    EXPECT_NEAR(scene.config.pot.height, (pot->rimHeight - pot->footHeight) * potScale, 1e-4f);
    // The captured triangles: first the pot's, then the object's (it has more than the pot keeps).
    std::vector<glm::vec3> captured;
    int glaze = 0, soil = 0;
    for (const TreeSceneVertex& v : scene.mesh) {
        glaze += v.color.w == TREE_MATERIAL_GLAZE;
        if (v.color.w == TREE_MATERIAL_MOSS) {
            ++soil;
            EXPECT_LT(scene.potMeasure(v.pos.x, v.pos.z), 1.0f);
            EXPECT_LT(v.pos.y, scene.config.pot.height + scene.config.pot.soilMound); // mounded towards the middle
        }
        if (v.color.w == TREE_MATERIAL_CAPTURED)
            captured.emplace_back(v.pos);
    }
    EXPECT_EQ(glaze, 0);
    EXPECT_GT(soil, 0);
    const size_t objectCorners = pot->mesh.size();
    ASSERT_GT(captured.size(), objectCorners);
    const std::vector<glm::vec3> potCorners(captured.begin(), captured.end() - objectCorners);
    glm::vec3 low(1e9f), high(-1e9f);
    for (const glm::vec3& p : potCorners) {
        low = glm::min(low, p);
        high = glm::max(high, p);
    }
    const float margin = 0.03f * scene.config.pot.height;
    EXPECT_GT(low.y, -margin);                           // nothing below the foot
    EXPECT_LT(high.y, scene.config.pot.height + margin); // nor above the rim: the lid is gone
    // The rim's long axis along x, as long as the procedural pot; across, the saucer (0.65) is wider than the rim.
    EXPECT_NEAR(std::max(high.x, -low.x), config.pot.halfLength, 0.03f * config.pot.halfLength);
    EXPECT_NEAR(std::max(high.z, -low.z), 0.65f * potScale, 0.03f * config.pot.halfLength);
    // The object stands on the soil, 0.3 tall, sunk by 5 % of its height.
    glm::vec3 objectLow(1e9f), objectHigh(-1e9f);
    for (auto p = captured.end() - objectCorners; p != captured.end(); ++p) {
        objectLow = glm::min(objectLow, *p);
        objectHigh = glm::max(objectHigh, *p);
    }
    EXPECT_NEAR(objectHigh.y - objectLow.y, 0.3f, 0.01f);
    EXPECT_NEAR(objectLow.y, scene.soilHeight(-0.5f, 0.2f) - 0.015f, 0.01f);
    EXPECT_NEAR(0.5f * (objectLow.x + objectHigh.x), -0.5f, 0.02f);
    EXPECT_NEAR(0.5f * (objectLow.z + objectHigh.z), 0.2f, 0.02f);
}

class BonsaiRenderingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(BonsaiRenderingTest, classicalGeometryIsHiddenBehindSplats) {
    const VkExtent2D extent{128, 128};
    // An opaque red block of Gaussians around the origin, 0.4 wide.
    std::vector<Gaussian3D> block;
    const float c0 = 0.28209479f;
    for (int i = 0; i < 8000; ++i) {
        Gaussian3D g{};
        g.position = {0.02f * float(i % 20) - 0.19f, 0.02f * float(i / 20 % 20) - 0.19f,
                      0.02f * float(i / 400) - 0.19f};
        g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        g.scale = {0.015f, 0.015f, 0.015f};
        g.color = {0.5f / c0, -0.5f / c0, -0.5f / c0};
        g.alpha = 0.99f;
        block.push_back(g);
    }
    auto model = std::make_shared<GaussianDataStandard>(*vc, std::move(block));
    // A blue quad behind the block on the left, a green quad in front on the right.
    std::vector<TreeSceneVertex> mesh;
    addQuad(mesh, -0.6f, 0.0f, -0.6f, 0.6f, -0.5f, glm::vec3(0.0f, 0.0f, 1.0f));
    addQuad(mesh, 0.1f, 0.6f, -0.6f, 0.6f, 0.5f, glm::vec3(0.0f, 1.0f, 0.0f));

    for (GsplatBackend backend : {GsplatBackend::Raster, GsplatBackend::Compute}) {
        SCOPED_TRACE(backend == GsplatBackend::Raster ? "raster" : "compute");
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        auto camera = std::make_shared<CameraUboType>();
        GsplatConfig config;
        config.outputDepth = true;
        auto splatting = createGaussianSplatting(*vc, backend, target, camera, model, config);
        auto toGeneral = std::make_shared<ImageToGeneral>();
        toGeneral->setInput(target, 0);
        toGeneral->addDependency(splatting);
        TreeRenderSettings values = defaultTreeRenderSettings(extent);
        values.occlusion.x = 1.0f;
        // Unlit: full ambient from every side, no sun.
        values.sun.w = 0.0f;
        values.sky = values.ground = glm::vec4(1.0f);
        auto settings = std::make_shared<HostValues<TreeRenderSettings>>(*vc, std::vector<TreeRenderSettings>{values});
        SceneMeshRenderer meshes(*vc, mesh, settings, extent,
                                 TreeRenderInputs{BufferRef{splatting->getDepth(), -1}, {}});
        auto pass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), extent);
        pass->setInput(toGeneral, 0);
        pass->setInput(camera, 1);
        pass->setLoadExisting(true);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(settings);
        pass->addDrawComponent(meshes.getDrawComponent());
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(pass);
        setCamera(camera->ubo, glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f), 40.0f, 1.0f);
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        writePpm(std::string("composite_") + (backend == GsplatBackend::Raster ? "raster" : "compute"), image, extent);

        auto pixel = [&](float x, float y) {
            const glm::vec4 clip = camera->ubo.proj * camera->ubo.view * glm::vec4(x, y, 0.0f, 1.0f);
            const uint32_t px = uint32_t((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
            const uint32_t py = uint32_t((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
            const size_t i = (size_t(py) * extent.width + px) * 4;
            return glm::ivec3(image[i + 2], image[i + 1], image[i]); // rgb
        };
        // Left: the block in front of the blue quad.
        const glm::ivec3 left = pixel(-0.1f, 0.0f);
        EXPECT_GT(left.r, 200);
        EXPECT_LT(left.b, 60);
        // Right: the green quad in front of the block.
        const glm::ivec3 right = pixel(0.15f, 0.0f);
        EXPECT_GT(right.g, 200);
        EXPECT_LT(right.r, 60);
        // Outside the block: the blue quad.
        const glm::ivec3 behind = pixel(-0.4f, 0.0f);
        EXPECT_GT(behind.b, 200);
    }
}

TEST_F(BonsaiRenderingTest, supersamplingSmoothsEdges) {
    const BonsaiScene scene = makeBonsaiScene();
    const VkExtent2D extent{480, 270};
    std::vector<double> variation, brightness;
    for (uint32_t supersample : {1u, 2u}) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.supersample = supersample;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(1.4f, 1.3f, 3.8f), scene.focus, 40.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        writePpm("bonsai_supersample_" + std::to_string(supersample), image, extent);
        double v = 0.0, b = 0.0;
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 1; x < extent.width; ++x) {
                const size_t i = (size_t(y) * extent.width + x) * 4;
                for (int c = 0; c < 3; ++c) {
                    v += std::abs(int(image[i + c]) - int(image[i - 4 + c]));
                    b += image[i + c];
                }
            }
        }
        variation.push_back(v);
        brightness.push_back(b);
    }
    std::cout << "neighbour variation " << variation[0] << " -> " << variation[1] << std::endl;
    EXPECT_LT(variation[1], 0.85 * variation[0]);
    EXPECT_NEAR(brightness[1] / brightness[0], 1.0, 0.1);
}

TEST_F(BonsaiRenderingTest, nightIsDarkerAndTheLanternGlows) {
    const BonsaiScene scene = makeBonsaiScene();
    const VkExtent2D extent{480, 270};
    for (BonsaiRenderMode mode : {BonsaiRenderMode::Classical, BonsaiRenderMode::Gaussian}) {
        const char* name = mode == BonsaiRenderMode::Classical ? "classical" : "gaussian";
        SCOPED_TRACE(name);
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = mode;
        config.meshGaussianSpacing = 0.02f;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(1.4f, 1.3f, 3.8f), scene.focus, 40.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        // The moss just in front of the lantern, in pixels.
        const glm::vec3 moss = scene.lanternTranslation + glm::vec3(0.0f, 0.0f, 0.2f);
        const glm::vec4 clip = bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view * glm::vec4(moss, 1.0f);
        const int px = int((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
        const int py = int((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
        auto render = [&](float timeOfDay) {
            bonsai.setTime(1.0f, timeOfDay, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
            writePpm(std::string("bonsai_") + name + (timeOfDay < 0.25f ? "_night" : "_noon"), image, extent);
            glm::dvec3 total(0.0), near(0.0);
            for (uint32_t y = 0; y < extent.height; ++y) {
                for (uint32_t x = 0; x < extent.width; ++x) {
                    const size_t i = (size_t(y) * extent.width + x) * 4;
                    const glm::dvec3 rgb(image[i + 2], image[i + 1], image[i]);
                    total += rgb;
                    if (std::abs(int(x) - px) <= 3 && std::abs(int(y) - py) <= 3) {
                        near += rgb;
                    }
                }
            }
            return std::make_pair(total, near);
        };
        const auto noon = render(0.5f);
        const auto night = render(0.0f);
        const double noonBrightness = noon.first.r + noon.first.g + noon.first.b;
        const double nightBrightness = night.first.r + night.first.g + night.first.b;
        EXPECT_LT(nightBrightness, 0.6 * noonBrightness);
        // Warmer: more red than blue next to the lantern at night, compared with noon.
        EXPECT_GT(night.second.r / std::max(night.second.b, 1.0), noon.second.r / std::max(noon.second.b, 1.0));
    }
}

TEST_F(BonsaiRenderingTest, bonsaiRendersBothWays) {
    const BonsaiScene scene = makeBonsaiScene();
    const VkExtent2D extent{480, 270};
    for (BonsaiRenderMode mode : {BonsaiRenderMode::Classical, BonsaiRenderMode::Gaussian}) {
        const char* name = mode == BonsaiRenderMode::Classical ? "classical" : "gaussian";
        SCOPED_TRACE(name);
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = mode;
        config.meshGaussianSpacing = 0.03f;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(1.4f, 1.3f, 3.8f), scene.focus, 40.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        writePpm(std::string("bonsai_") + name, image, extent);
        // Trees and pot fill a good part of the frame; the background stays black.
        const size_t coverage = coveredCount(image, extent);
        EXPECT_GT(coverage, extent.width * extent.height / 5);
        EXPECT_LT(coverage, extent.width * extent.height * 4 / 5);
        EXPECT_EQ(bonsai.getGenerator().readCounters()[kTreeCounterOverflow], 0u);
    }
}

TEST_F(BonsaiRenderingTest, potOfGaussiansRendersBothWays) {
    // A pot of Gaussians: a box shell 2 x 0.5 x 1.2 of opaque grey splats, open at the top.
    auto pot = std::make_shared<BonsaiObject>();
    for (int i = 0; i <= 40; ++i)
        for (int j = 0; j <= 10; ++j)
            for (const glm::vec3 p :
                 {glm::vec3(-1.0f + 0.05f * i, 0.05f * j, -0.6f), glm::vec3(-1.0f + 0.05f * i, 0.05f * j, 0.6f),
                  glm::vec3(-1.0f, 0.05f * j, -0.6f + 0.03f * i), glm::vec3(1.0f, 0.05f * j, -0.6f + 0.03f * i)}) {
                Gaussian3D g{};
                g.position = {p.x, p.y, p.z};
                g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
                g.scale = {0.04f, 0.04f, 0.04f};
                g.color = {0.0f, 0.0f, 0.0f};
                g.alpha = 0.95f;
                pot->gaussians.push_back(g);
            }
    pot->low = glm::vec3(-1.0f, 0.0f, -0.6f);
    pot->high = glm::vec3(1.0f, 0.5f, 0.6f);
    pot->rimHeight = 0.5f;
    pot->rim = glm::vec3(1.0f, 0.6f, 0.0f);
    BonsaiSceneConfig sceneConfig;
    sceneConfig.potObject = pot;
    sceneConfig.treeCount = 2;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const VkExtent2D extent{320, 180};
    for (BonsaiRenderMode mode : {BonsaiRenderMode::Classical, BonsaiRenderMode::Gaussian}) {
        SCOPED_TRACE(mode == BonsaiRenderMode::Classical ? "classical" : "gaussian");
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = mode;
        config.meshGaussianSpacing = 0.03f;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(0.0f, 1.0f, 5.0f), scene.focus, 40.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        EXPECT_GT(coveredCount(image, extent), extent.width * extent.height / 8);
    }
}

TEST_F(BonsaiRenderingTest, backgroundFollowsTheFrame) {
    BonsaiSceneConfig sceneConfig;
    sceneConfig.treeCount = 2;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const VkExtent2D extent{320, 180};
    for (BonsaiRenderMode mode : {BonsaiRenderMode::Classical, BonsaiRenderMode::Gaussian}) {
        SCOPED_TRACE(mode == BonsaiRenderMode::Classical ? "classical" : "gaussian");
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = mode;
        config.meshGaussianSpacing = 0.03f;
        config.background = true;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(1.4f, 1.3f, 3.8f), scene.focus, 40.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        std::vector<std::vector<uint8_t>> images;
        for (const float grey : {1.0f, 0.0f}) {
            bonsai.setBackground(glm::vec3(grey));
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
            images.push_back(image);
            for (const size_t pixel : {size_t(0), size_t(extent.width - 1), size_t(extent.width) * (extent.height - 1)})
                for (int c = 0; c < 3; ++c)
                    EXPECT_NEAR(float(image[4 * pixel + c]), 255.0f * grey, 2.0f)
                        << "pixel " << pixel << " grey " << grey;
        }
    }
}

TEST_F(BonsaiRenderingTest, mossReceivesObjectShadows) {
    // An opaque grey ball, 0.5 tall, on the soil in front of the trees.
    const raster::TriangleMesh ball = raster::icosphere(3);
    const std::vector<float> grey(4 * 8 * 4, 0.5f);
    BonsaiObjectPlacement placement;
    placement.object = std::make_shared<BonsaiObject>(bonsaiObjectFromMesh(ball, grey, 8, 4));
    placement.position = glm::vec2(0.75f, -0.3f);
    placement.height = 0.5f;
    const float timeOfDay = 0.38f, season = 2.0f;
    const TreeSky sky = treeSkyAt(timeOfDay, season);
    ASSERT_GT(sky.sunElevation, 0.3f);
    ASSERT_LT(sky.sunElevation, 0.9f);

    // Where the ball's shadow falls clear of the ball: from high on it away from the sun down to the soil.
    BonsaiSceneConfig withBall;
    withBall.treeCount = 1;
    withBall.objects.push_back(placement);
    const BonsaiScene sceneWith = makeBonsaiScene(withBall);
    BonsaiSceneConfig withoutBall = withBall;
    withoutBall.objects.clear();
    const BonsaiScene sceneWithout = makeBonsaiScene(withoutBall);
    const float soil = sceneWith.soilHeight(placement.position.x, placement.position.y);
    const glm::vec3 high(placement.position.x, soil + 0.75f * placement.height, placement.position.y);
    glm::vec3 shadow = high - sky.sunDirection * (0.75f * placement.height / sky.sunDirection.y);
    shadow.y = sceneWith.soilHeight(shadow.x, shadow.z);
    ASSERT_LT(std::abs(shadow.x), 0.8f * sceneWith.config.pot.halfLength);
    ASSERT_LT(std::abs(shadow.z), 0.8f * sceneWith.config.pot.halfWidth);
    // Clear of the trees' trunks and the lantern, whose own shadows would hide the ball's.
    for (const TreeInstance& tree : sceneWith.trees)
        ASSERT_GT(glm::length(glm::vec2(tree.posScale.x - shadow.x, tree.posScale.z - shadow.z)), 0.4f);
    ASSERT_GT(
        glm::length(glm::vec2(sceneWith.lanternTranslation.x - shadow.x, sceneWith.lanternTranslation.z - shadow.z)),
        0.4f);

    const VkExtent2D extent{480, 270};
    // Seen from above, across the sun's direction, so that the ball hides none of its shadow.
    const glm::vec3 across = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), sky.sunDirection));
    const glm::vec3 eye = shadow + 2.5f * across + glm::vec3(0.0f, 2.5f, 0.0f);
    auto render = [&](const BonsaiScene& scene, int& px, int& py) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = BonsaiRenderMode::Classical;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, eye, shadow, 40.0f, float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        bonsai.setTime(1.0f, timeOfDay, season, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const glm::vec4 clip = bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view * glm::vec4(shadow, 1.0f);
        px = int((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
        py = int((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
        return readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    };
    int px = 0, py = 0;
    const auto lit = render(sceneWithout, px, py);
    const auto shaded = render(sceneWith, px, py);
    writePpm("bonsai_ball_shadow", shaded, extent);
    // Mean and brightest brightness in a window around the middle of the shadow.
    auto measure = [&](const std::vector<uint8_t>& image) {
        double sum = 0.0, brightest = 0.0;
        int count = 0;
        for (int y = py - 4; y <= py + 4; ++y)
            for (int x = px - 4; x <= px + 4; ++x) {
                const size_t i = (size_t(y) * extent.width + size_t(x)) * 4;
                const double value = double(image[i]) + image[i + 1] + image[i + 2];
                sum += value;
                brightest = std::max(brightest, value);
                ++count;
            }
        return std::make_pair(sum / count, brightest);
    };
    const auto [litMean, litBrightest] = measure(lit);
    const auto [shadedMean, shadedBrightest] = measure(shaded);
    std::cout << "shadow window mean " << litMean << " -> " << shadedMean << ", brightest " << litBrightest << " -> "
              << shadedBrightest << std::endl;
    EXPECT_LT(shadedMean, 0.65 * litMean);
    EXPECT_LT(shadedBrightest, 0.75 * litBrightest);
}

namespace {

// A captured grey ball of unit radius around (0, 0, 0): a mesh, or opaque Gaussians on its surface.
std::shared_ptr<BonsaiObject> greyBall(bool gaussians) {
    if (!gaussians) {
        const std::vector<float> grey(4 * 8 * 4, 0.5f);
        return std::make_shared<BonsaiObject>(bonsaiObjectFromMesh(raster::icosphere(3), grey, 8, 4));
    }
    auto ball = std::make_shared<BonsaiObject>();
    const int count = 4000;
    for (int i = 0; i < count; ++i) {
        // A Fibonacci sphere.
        const float y = 1.0f - 2.0f * (float(i) + 0.5f) / float(count);
        const float r = std::sqrt(1.0f - y * y);
        const float phi = 2.39996323f * float(i);
        Gaussian3D g{};
        g.position = {r * std::cos(phi), y, r * std::sin(phi)};
        g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        g.scale = {0.04f, 0.04f, 0.04f};
        g.color = {0.0f, 0.0f, 0.0f}; // 0.5 grey
        g.alpha = 0.98f;
        ball->gaussians.push_back(g);
    }
    ball->low = glm::vec3(-1.0f);
    ball->high = glm::vec3(1.0f);
    return ball;
}

// The morning sun of the shadow tests, with an elevation that casts long shadows.
constexpr float kMorning = 0.36f;

} // namespace

TEST_F(BonsaiRenderingTest, capturedObjectsShadowThemselves) {
    const TreeSky sky = treeSkyAt(kMorning, 2.0f);
    ASSERT_GT(sky.sunElevation, 0.25f);
    const glm::vec3 sunAcross = glm::normalize(glm::vec3(sky.sunDirection.x, 0.0f, sky.sunDirection.z));
    const glm::vec3 across = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), sunAcross));
    const VkExtent2D extent{320, 240};
    for (const bool gaussians : {false, true}) {
        SCOPED_TRACE(gaussians ? "Gaussians" : "mesh");
        BonsaiObjectPlacement placement;
        placement.object = greyBall(gaussians);
        placement.position = glm::vec2(0.6f, -0.2f);
        placement.height = 0.6f;
        BonsaiSceneConfig config;
        config.treeCount = 1;
        config.objects.push_back(placement);
        const BonsaiScene scene = makeBonsaiScene(config);
        const float radius = 0.5f * placement.height;
        const glm::vec3 centre(placement.position.x,
                               scene.soilHeight(placement.position.x, placement.position.y) + 0.45f * placement.height,
                               placement.position.y);

        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig renderConfig;
        renderConfig.extent = extent;
        renderConfig.mode = BonsaiRenderMode::Classical;
        BonsaiRenderer bonsai(*vc, scene, target, renderConfig);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        // Level with the ball, across the sun: its sunny half on one side, its dark half on the other.
        setCamera(bonsai.getCamera()->ubo, centre + 2.0f * across, centre, 30.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        bonsai.setTime(1.0f, kMorning, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        writePpm(gaussians ? "bonsai_ball_gaussians_relit" : "bonsai_ball_mesh_relit", image, extent);
        const glm::vec3 tint = glm::vec3(bonsai.getSettings().values().at(0).capturedTint);
        const glm::mat4 clipFromWorld = bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view;

        // Mean and smallest brightness (summed over rgb) in a window around the surface point in direction n.
        auto measure = [&](glm::vec3 n) {
            const glm::vec4 clip = clipFromWorld * glm::vec4(centre + 0.85f * radius * glm::normalize(n), 1.0f);
            const int px = int((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
            const int py = int((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
            double sum = 0.0, darkest = 1e9;
            int count = 0;
            for (int y = py - 3; y <= py + 3; ++y)
                for (int x = px - 3; x <= px + 3; ++x) {
                    const size_t i = (size_t(y) * extent.width + size_t(x)) * 4;
                    const double value = double(image[i]) + image[i + 1] + image[i + 2];
                    sum += value;
                    darkest = std::min(darkest, value);
                    ++count;
                }
            return std::make_pair(sum / count, darkest);
        };
        const auto [sunny, sunnyDarkest] = measure(across + sunAcross + glm::vec3(0.0f, 0.3f, 0.0f));
        const auto [dark, darkDarkest] = measure(across - sunAcross);
        const double fullLight = 127.5 * double(tint.x + tint.y + tint.z);
        std::cout << "sunny " << sunny << " (darkest " << sunnyDarkest << "), away " << dark << ", full light "
                  << fullLight << std::endl;
        EXPECT_NEAR(sunny, fullLight, 0.12 * fullLight);
        EXPECT_GT(sunnyDarkest, 0.8 * sunny);
        EXPECT_LT(dark, 0.7 * sunny);
    }
}

TEST_F(BonsaiRenderingTest, capturedObjectsShadowEachOther) {
    // Earlier, a lower sun: the occluder's shadow reaches the lit ball from beside it.
    const float early = 0.31f;
    const TreeSky sky = treeSkyAt(early, 2.0f);
    ASSERT_GT(sky.sunElevation, 0.15f);
    const glm::vec3 sunAcross = glm::normalize(glm::vec3(sky.sunDirection.x, 0.0f, sky.sunDirection.z));
    const glm::vec3 across = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), sunAcross));
    const VkExtent2D extent{320, 240};
    // The lit ball, and a larger one towards the sun from it, on the line from the lit ball's sunny side to the sun.
    BonsaiObjectPlacement lit;
    lit.object = greyBall(false);
    lit.position = glm::vec2(0.2f, -0.35f);
    lit.height = 0.3f;
    BonsaiSceneConfig config;
    config.treeCount = 1;
    config.objects.push_back(lit);
    const BonsaiScene withoutOccluder = makeBonsaiScene(config);
    const glm::vec3 centre(lit.position.x,
                           withoutOccluder.soilHeight(lit.position.x, lit.position.y) + 0.45f * lit.height,
                           lit.position.y);
    const glm::vec3 sunny = centre + 0.85f * 0.5f * lit.height * glm::normalize(across + sunAcross);
    const float distance = 0.7f;
    const glm::vec3 occluderCentre =
        sunny + sky.sunDirection * (distance / glm::length(glm::vec3(sky.sunDirection.x, 0.0f, sky.sunDirection.z)));
    BonsaiObjectPlacement occluder;
    occluder.object = greyBall(true);
    occluder.position = glm::vec2(occluderCentre.x, occluderCentre.z);
    occluder.height = 2.0f * (occluderCentre.y - withoutOccluder.soilHeight(occluderCentre.x, occluderCentre.z)) / 0.9f;
    ASSERT_LT(occluder.height, 1.6f);
    ASSERT_LT(std::abs(occluderCentre.x), withoutOccluder.config.pot.halfLength);
    ASSERT_LT(std::abs(occluderCentre.z), withoutOccluder.config.pot.halfWidth);
    config.objects.push_back(occluder);
    const BonsaiScene withOccluder = makeBonsaiScene(config);

    auto render = [&](const BonsaiScene& scene) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig renderConfig;
        renderConfig.extent = extent;
        renderConfig.mode = BonsaiRenderMode::Classical;
        BonsaiRenderer bonsai(*vc, scene, target, renderConfig);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, centre + 2.0f * across, centre, 30.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        bonsai.setTime(1.0f, early, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const glm::mat4 clipFromWorld = bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view;
        const glm::vec4 clip = clipFromWorld * glm::vec4(sunny, 1.0f);
        const int px = int((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
        const int py = int((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
        // The occluder, seen beside the lit ball, stays clear of the measured pixels.
        const glm::vec4 occluderClip = clipFromWorld * glm::vec4(occluderCentre, 1.0f);
        const glm::vec4 topClip =
            clipFromWorld * glm::vec4(occluderCentre + glm::vec3(0.0f, 0.5f * occluder.height, 0.0f), 1.0f);
        const float occluderPixels =
            0.5f * float(extent.height) * std::abs(topClip.y / topClip.w - occluderClip.y / occluderClip.w);
        const glm::vec2 occluderAt((occluderClip.x / occluderClip.w * 0.5f + 0.5f) * float(extent.width),
                                   (occluderClip.y / occluderClip.w * 0.5f + 0.5f) * float(extent.height));
        EXPECT_GT(glm::length(occluderAt - glm::vec2(px, py)), occluderPixels + 6.0f);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        double sum = 0.0;
        for (int y = py - 2; y <= py + 2; ++y)
            for (int x = px - 2; x <= px + 2; ++x) {
                const size_t i = (size_t(y) * extent.width + size_t(x)) * 4;
                sum += double(image[i]) + image[i + 1] + image[i + 2];
            }
        return sum / 25.0;
    };
    const double inSun = render(withoutOccluder);
    const double inShadow = render(withOccluder);
    std::cout << "sunny side " << inSun << " -> " << inShadow << " behind the other ball" << std::endl;
    EXPECT_LT(inShadow, 0.7 * inSun);
}

TEST_F(BonsaiRenderingTest, movingObjectsCastShadowsWhereTheyAre) {
    BonsaiSceneConfig sceneConfig;
    sceneConfig.treeCount = 1;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const VkExtent2D extent{64, 64};
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig config;
    config.extent = extent;
    config.movingObjects.push_back(greyBall(false));
    BonsaiRenderer bonsai(*vc, scene, target, config);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(bonsai.getOutput());
    bonsai.initialize(1);
    setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(0.0f, 1.0f, 4.0f), scene.focus, 40.0f, 1.0f);
    bonsai.getCamera()->update(0);
    // Noon: the shadow lies about under the ball.
    bonsai.setTime(1.0f, 0.5f, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));

    // Per soil column just above the soil, the sun through all occluders relative to that through the trees.
    auto soilLight = [&](const glm::mat4& worldFromBall) {
        bonsai.setMovingObject(0, worldFromBall);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto light = bonsai.getLightVolume().readLight();
        const TreeRenderSettings v = bonsai.getSettings().values().at(0);
        std::vector<std::pair<glm::vec2, float>> columns;
        for (uint32_t z = 0; z < v.lightGridSize.z; ++z)
            for (uint32_t x = 0; x < v.lightGridSize.x; ++x) {
                const glm::vec2 at(v.lightGridMin.x + (float(x) + 0.5f) * v.lightGridMin.w,
                                   v.lightGridMin.z + (float(z) + 0.5f) * v.lightGridMin.w);
                const float y = scene.soilHeight(at.x, at.y) + 0.03f;
                const auto cell = uint32_t((y - v.lightGridMin.y) / v.lightGridMin.w);
                const glm::vec4 l = light[(size_t(z) * v.lightGridSize.y + cell) * v.lightGridSize.x + x];
                columns.emplace_back(at, l.x / std::max(l.z, 1e-6f));
            }
        return columns;
    };
    // The middle of the columns that are clearly darker than without the ball, and their number.
    const auto without = soilLight(glm::scale(glm::mat4(1.0f), glm::vec3(0.0f)));
    auto shadowOf = [&](const glm::mat4& worldFromBall) {
        const auto with = soilLight(worldFromBall);
        glm::vec2 sum(0.0f);
        int count = 0;
        for (size_t i = 0; i < with.size(); ++i)
            if (with[i].second < without[i].second - 0.3f) {
                sum += with[i].first;
                ++count;
            }
        return std::make_pair(count > 0 ? sum / float(count) : glm::vec2(1e9f), count);
    };
    auto ballAt = [&](glm::vec2 at) {
        const float radius = 0.2f;
        return glm::translate(glm::mat4(1.0f), glm::vec3(at.x, scene.soilHeight(at.x, at.y) + 0.5f, at.y)) *
               glm::scale(glm::mat4(1.0f), glm::vec3(radius));
    };
    const auto [first, firstCount] = shadowOf(ballAt(glm::vec2(-0.4f, 0.1f)));
    const auto [second, secondCount] = shadowOf(ballAt(glm::vec2(0.2f, 0.1f)));
    const auto [gone, goneCount] = shadowOf(glm::scale(glm::mat4(1.0f), glm::vec3(0.0f)));
    std::cout << "shadow " << firstCount << " columns at " << first.x << " " << first.y << ", then " << secondCount
              << " at " << second.x << " " << second.y << ", removed " << goneCount << std::endl;
    // About the area of the ball's cross-section, 0.13 units^2, in voxel columns.
    const float voxel = bonsai.getSettings().values().at(0).lightGridMin.w;
    EXPECT_GT(float(firstCount), 0.5f * 0.126f / (voxel * voxel));
    EXPECT_NEAR(float(secondCount), float(firstCount), 0.25f * float(firstCount));
    EXPECT_NEAR(second.x - first.x, 0.6f, 0.05f);
    EXPECT_NEAR(second.y - first.y, 0.0f, 0.05f);
    EXPECT_EQ(goneCount, 0);
}

TEST_F(BonsaiRenderingTest, litMorphShadesLikeTheScene) {
    const TreeSky sky = treeSkyAt(kMorning, 2.0f);
    const glm::vec3 sunAcross = glm::normalize(glm::vec3(sky.sunDirection.x, 0.0f, sky.sunDirection.z));
    const glm::vec3 across = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), sunAcross));
    BonsaiSceneConfig sceneConfig;
    sceneConfig.treeCount = 1;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const glm::vec2 at(0.6f, -0.2f);
    const float radius = 0.3f;
    const glm::vec3 centre(at.x, scene.soilHeight(at.x, at.y) + radius + 0.05f, at.y);
    const glm::mat4 model = glm::translate(glm::mat4(1.0f), centre) * glm::scale(glm::mat4(1.0f), glm::vec3(radius));

    const VkExtent2D extent{320, 240};
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig config;
    config.extent = extent;
    const std::vector<float> grey(4 * 64 * 32, 0.5f);
    config.movingObjects.push_back(
        std::make_shared<BonsaiObject>(bonsaiObjectFromMesh(raster::icosphere(4), grey, 64, 32)));
    BonsaiRenderer bonsai(*vc, scene, target, config);

    // A grey sphere morphing towards grey Gaussians on the unit sphere: it stays a grey sphere.
    auto targets = std::make_shared<GaussianDataStandard>(*vc, greyBall(true)->gaussians);
    raster::MeshMorphOptions options;
    for (int k = 0; k < 4; ++k) {
        const float angle = float(k) * 1.5708f;
        const glm::vec3 eye = 3.5f * glm::vec3(std::cos(angle), 0.3f, std::sin(angle));
        CameraMVP view{};
        view.model = glm::mat4(1.0f);
        view.view = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        view.proj = raster::perspective(0.8f, 1.0f, 0.1f, 20.0f);
        view.cameraWorldPos = glm::vec4(eye, 1.0f);
        options.views.push_back(view);
    }
    options.training.width = options.training.height = 64;
    options.frame = extent;
    options.lightSettings = bonsai.getSettingsRef();
    options.light = bonsai.getLightVolume().getLight();
    const VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    raster::MeshMorph morph(*vc, raster::icosphere(4), targets->buffers(), target, layout, bonsai.getOutput(), options);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(morph.getOutput());
    bonsai.initialize(1);
    morph.initialize();
    morph.getTraining().setTexture(grey);

    setCamera(bonsai.getCamera()->ubo, centre + 2.0f * across, centre, 30.0f, float(extent.width) / extent.height);
    bonsai.getCamera()->update(0);
    bonsai.setTime(1.0f, kMorning, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    bonsai.setMovingObject(0, model);
    const glm::mat4 clipFromWorld = bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view;
    morph.setCamera(clipFromWorld);
    morph.setModel(model);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    const auto image = readImage(*vc, target->getImage(0), layout, extent);
    writePpm("bonsai_lit_morph", image, extent);
    const glm::vec3 tint = glm::vec3(bonsai.getSettings().values().at(0).capturedTint);

    auto measure = [&](glm::vec3 n) {
        const glm::vec4 clip = clipFromWorld * glm::vec4(centre + 0.8f * radius * glm::normalize(n), 1.0f);
        const int px = int((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
        const int py = int((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
        double sum = 0.0;
        for (int y = py - 2; y <= py + 2; ++y)
            for (int x = px - 2; x <= px + 2; ++x) {
                const size_t i = (size_t(y) * extent.width + size_t(x)) * 4;
                sum += double(image[i]) + image[i + 1] + image[i + 2];
            }
        return sum / 25.0;
    };
    const double sunny = measure(across + sunAcross + glm::vec3(0.0f, 0.3f, 0.0f));
    const double away = measure(across - sunAcross);
    const double fullLight = 127.5 * double(tint.x + tint.y + tint.z);
    std::cout << "lit morph: sunny " << sunny << ", away " << away << ", full light " << fullLight << std::endl;
    EXPECT_NEAR(sunny, fullLight, 0.12 * fullLight);
    EXPECT_LT(away, 0.7 * sunny);
}

TEST_F(BonsaiRenderingTest, shadowsMoveSmoothlyWithTheSun) {
    BonsaiObjectPlacement placement;
    placement.object = greyBall(false);
    placement.position = glm::vec2(0.6f, -0.2f);
    placement.height = 0.6f;
    BonsaiSceneConfig config;
    config.treeCount = 1;
    config.objects.push_back(placement);
    const BonsaiScene scene = makeBonsaiScene(config);
    const glm::vec3 centre(0.6f, scene.soilHeight(0.6f, -0.2f) + 0.27f, -0.2f);
    const VkExtent2D extent{320, 240};
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig rc;
    rc.extent = extent;
    BonsaiRenderer bonsai(*vc, scene, target, rc);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(bonsai.getOutput());
    bonsai.initialize(1);
    setCamera(bonsai.getCamera()->ubo, centre + glm::vec3(0.0f, 0.3f, 1.6f), centre, 30.0f, 4.0f / 3.0f);
    bonsai.getCamera()->update(0);
    std::vector<uint8_t> previous;
    double worst = 0, sum = 0;
    for (int i = 0; i <= 60; ++i) {
        bonsai.setTime(1.0f, 0.36f + 0.002f * i, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        if (!previous.empty()) {
            // The ball and the soil around it: the middle of the frame.
            double maxDiff = 0, meanDiff = 0;
            int n = 0;
            for (uint32_t y = 70; y < 170; ++y)
                for (uint32_t x = 110; x < 210; ++x) {
                    const size_t k = (y * extent.width + x) * 4;
                    double d = 0;
                    for (int c = 0; c < 3; ++c)
                        d += std::abs(int(image[k + c]) - int(previous[k + c]));
                    maxDiff = std::max(maxDiff, d);
                    meanDiff += d;
                    ++n;
                }
            meanDiff /= n;
            worst = std::max(worst, maxDiff);
            sum += meanDiff;
        }
        previous = image;
    }
    std::cout << "largest change of a pixel between steps " << worst << ", mean " << sum / 60 << std::endl;
    // A shadow edge hopping from voxel to voxel changes pixels by more than 100 at once.
    EXPECT_LT(worst, 30.0);
}

namespace {

// A captured pot of Gaussians: a box shell 2 x 0.5 x 1.2 of opaque grey splats, open at the top, flat along their
// walls as a trained capture's are.
std::shared_ptr<BonsaiObject> boxPot() {
    auto pot = std::make_shared<BonsaiObject>();
    for (int i = 0; i <= 40; ++i)
        for (int j = 0; j <= 10; ++j)
            for (int wall = 0; wall < 4; ++wall) {
                const glm::vec3 p = wall == 0   ? glm::vec3(-1.0f + 0.05f * i, 0.05f * j, -0.6f)
                                    : wall == 1 ? glm::vec3(-1.0f + 0.05f * i, 0.05f * j, 0.6f)
                                    : wall == 2 ? glm::vec3(-1.0f, 0.05f * j, -0.6f + 0.03f * i)
                                                : glm::vec3(1.0f, 0.05f * j, -0.6f + 0.03f * i);
                Gaussian3D g{};
                g.position = {p.x, p.y, p.z};
                g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
                g.scale =
                    wall < 2 ? std::array<float, 3>{0.04f, 0.04f, 0.008f} : std::array<float, 3>{0.008f, 0.04f, 0.04f};
                g.color = {0.0f, 0.0f, 0.0f};
                g.alpha = 0.95f;
                pot->gaussians.push_back(g);
            }
    pot->low = glm::vec3(-1.0f, 0.0f, -0.6f);
    pot->high = glm::vec3(1.0f, 0.5f, 0.6f);
    pot->rimHeight = 0.5f;
    pot->rim = glm::vec3(1.0f, 0.6f, 0.0f);
    return pot;
}

// The light volume at p as tree_light.slang reads it: trilinear between voxel centres.
glm::vec4 lightAt(const std::vector<glm::vec4>& light, const TreeRenderSettings& s, glm::vec3 p) {
    const glm::ivec3 size(s.lightGridSize);
    const glm::vec3 g =
        glm::clamp((p - glm::vec3(s.lightGridMin)) / s.lightGridMin.w - 0.5f, glm::vec3(0.0f), glm::vec3(size - 1));
    const glm::ivec3 i0 = glm::ivec3(glm::floor(g));
    const glm::vec3 f = g - glm::vec3(i0);
    glm::vec4 sum(0.0f);
    for (int corner = 0; corner < 8; ++corner) {
        const glm::ivec3 o(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        const glm::ivec3 q = glm::min(i0 + o, size - 1);
        const float w = (o.x ? f.x : 1.0f - f.x) * (o.y ? f.y : 1.0f - f.y) * (o.z ? f.z : 1.0f - f.z);
        sum += w * light[(size_t(q.z) * size.y + q.y) * size.x + q.x];
    }
    return sum;
}

} // namespace

TEST_F(BonsaiRenderingTest, potShadowsItsOwnFarSide) {
    BonsaiSceneConfig sceneConfig;
    sceneConfig.potObject = boxPot();
    sceneConfig.treeCount = 1;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const VkExtent2D extent{64, 64};
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig config;
    config.extent = extent;
    BonsaiRenderer bonsai(*vc, scene, target, config);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(bonsai.getOutput());
    bonsai.initialize(1);
    setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(0.0f, 1.0f, 4.0f), scene.focus, 40.0f, 1.0f);
    bonsai.getCamera()->update(0);
    bonsai.setTime(1.0f, kMorning, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    const auto light = bonsai.getLightVolume().readLight();
    const TreeRenderSettings s = bonsai.getSettings().values().at(0);
    const glm::vec3 sun = glm::normalize(glm::vec3(s.sun));
    const glm::vec2 sunAcross = glm::normalize(glm::vec2(sun.x, sun.z));

    // The walls' Gaussians at mid-height, on the side towards the sun and on the side away from it.
    float reach = 0.0f, top = 0.0f;
    for (const Gaussian3D& g : scene.objectGaussians) {
        reach = std::max(reach, std::abs(glm::dot(glm::vec2(g.position[0], g.position[2]), sunAcross)));
        top = std::max(top, g.position[1]);
    }
    double sunnySum = 0.0, farSum = 0.0, sunnyLeast = 1.0;
    int sunnyCount = 0, farCount = 0;
    for (const Gaussian3D& g : scene.objectGaussians) {
        const glm::vec3 p(g.position[0], g.position[1], g.position[2]);
        if (p.y < 0.25f * top || p.y > 0.75f * top)
            continue;
        const float side = glm::dot(glm::vec2(p.x, p.z), sunAcross) / reach;
        // The sun through the static occluders only (all over the trees), as treeCapturedLightAt() reads it.
        const glm::vec4 l = lightAt(light, s, p + sun * (1.5f * s.lightGridMin.w));
        const double statics = l.x / std::max(l.z, 1e-4f);
        if (side > 0.95f) {
            sunnySum += statics;
            sunnyLeast = std::min(sunnyLeast, statics);
            ++sunnyCount;
        } else if (side < -0.95f) {
            farSum += statics;
            ++farCount;
        }
    }
    ASSERT_GT(sunnyCount, 20);
    ASSERT_GT(farCount, 20);
    std::cout << "sun through the pot: sunny wall " << sunnySum / sunnyCount << " (least " << sunnyLeast
              << "), far wall " << farSum / farCount << std::endl;
    EXPECT_GT(sunnySum / sunnyCount, 0.9);
    EXPECT_LT(farSum / farCount, 0.15);
}

TEST_F(BonsaiRenderingTest, treeShadowsStayOffAPotsShadowSide) {
    BonsaiSceneConfig sceneConfig;
    sceneConfig.potObject = boxPot();
    sceneConfig.treeCount = 1;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const TreeSky sky = treeSkyAt(kMorning, 2.0f);
    const glm::vec2 sunAcross = glm::normalize(glm::vec2(sky.sunDirection.x, sky.sunDirection.z));
    // The far wall's Gaussians at mid-height: of the box's four walls, the one facing most directly away from the sun.
    const bool alongX = std::abs(sunAcross.x) > std::abs(sunAcross.y);
    const float away = alongX ? -sunAcross.x : -sunAcross.y;
    float reach = 0.0f, top = 0.0f;
    for (const Gaussian3D& g : scene.objectGaussians) {
        reach = std::max(reach, (away > 0.0f ? 1.0f : -1.0f) * (alongX ? g.position[0] : g.position[2]));
        top = std::max(top, g.position[1]);
    }
    std::vector<glm::vec3> farWall;
    for (const Gaussian3D& g : scene.objectGaussians) {
        const float out = (away > 0.0f ? 1.0f : -1.0f) * (alongX ? g.position[0] : g.position[2]);
        if (out > reach - 0.01f && g.position[1] > 0.3f * top && g.position[1] < 0.7f * top)
            farWall.emplace_back(g.position[0], g.position[1], g.position[2]);
    }
    ASSERT_GT(farWall.size(), 20u);
    glm::vec3 middle(0.0f);
    for (const glm::vec3& p : farWall)
        middle += p / float(farWall.size());

    const VkExtent2D extent{320, 240};
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig config;
    config.extent = extent;
    BonsaiRenderer bonsai(*vc, scene, target, config);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(bonsai.getOutput());
    bonsai.initialize(1);
    // Level with the far wall, facing it from outside the pot.
    const glm::vec3 outwards =
        alongX ? glm::vec3(away > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 0.0f, away > 0.0f ? 1.0f : -1.0f);
    setCamera(bonsai.getCamera()->ubo, middle + 2.5f * outwards, middle, 40.0f, float(extent.width) / extent.height);
    bonsai.getCamera()->update(0);
    bonsai.setTime(1.0f, kMorning, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    const glm::mat4 clipFromWorld = bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view;

    auto render = [&](float age) {
        std::vector<TreeInstance> trees = scene.trees;
        for (TreeInstance& tree : trees)
            setTreeAge(tree, age);
        bonsai.getGenerator().getInstances().set(trees);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        return readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    };
    const auto withTree = render(1.0f);
    // The tree's shadow reaches the far wall: the sun through the trees there is clearly less than full.
    const auto light = bonsai.getLightVolume().readLight();
    const TreeRenderSettings s = bonsai.getSettings().values().at(0);
    int shadedByTree = 0;
    for (const glm::vec3& p : farWall)
        if (lightAt(light, s, p + glm::normalize(glm::vec3(s.sun)) * (1.5f * s.lightGridMin.w)).z < 0.7f)
            ++shadedByTree;
    ASSERT_GT(shadedByTree, int(farWall.size() / 10)) << "the tree's shadow misses the far wall";
    const auto withoutTree = render(0.0f);
    writePpm("bonsai_far_wall_tree", withTree, extent);
    writePpm("bonsai_far_wall_no_tree", withoutTree, extent);

    // The far wall's pixels, the same with and without the tree.
    double largest = 0.0, total = 0.0;
    int count = 0;
    for (const glm::vec3& p : farWall) {
        const glm::vec4 clip = clipFromWorld * glm::vec4(p, 1.0f);
        const int x = int((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
        const int y = int((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
        if (x < 0 || y < 0 || x >= int(extent.width) || y >= int(extent.height))
            continue;
        const size_t i = (size_t(y) * extent.width + size_t(x)) * 4;
        double difference = 0.0;
        for (int c = 0; c < 3; ++c)
            difference += std::abs(int(withTree[i + c]) - int(withoutTree[i + c]));
        largest = std::max(largest, difference);
        total += difference;
        ++count;
    }
    ASSERT_GT(count, 20);
    std::cout << "far wall with and without the tree: mean difference " << total / count << ", largest " << largest
              << " (" << shadedByTree << " of " << farWall.size() << " points in the tree's shadow)" << std::endl;
    EXPECT_LT(total / count, 2.0);
    EXPECT_LT(largest, 12.0);
}
