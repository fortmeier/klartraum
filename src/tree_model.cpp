// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_model.hpp"

namespace klartraum {
namespace tree {

namespace {

// Inputs and outputs of the CPU reference generation, passed through the
// TREE_CTX_* hooks of tree_generate.hpp.
struct ReferenceContext {
    const std::vector<TreeTypeParams>* types;
    const std::vector<TreeInstance>* instances;
    TreeFrame frame;
    TreeRecords* out;
};

void treeEmitStem(const StemRecord& s, ReferenceContext& ctx) { ctx.out->stems[treeLevel(s.ids)].push_back(s); }

void treeEmitSegment(const SegmentRecord& d, ReferenceContext& ctx) { ctx.out->segments.push_back(d); }

void treeEmitLeaf(const LeafRecord& l, ReferenceContext& ctx) { ctx.out->leaves.push_back(l); }

#define TREE_FN inline
#define TREE_CTX_PARAM , ReferenceContext& ctx
#define TREE_CTX_ARG , ctx
#define TREE_PARAMS(type) (ctx.types->at(type))
#define TREE_INSTANCE(index) (ctx.instances->at(index))
#define TREE_FRAME (ctx.frame)
#include "klartraum/tree/tree_generate.hpp"
#undef TREE_FN
#undef TREE_CTX_PARAM
#undef TREE_CTX_ARG
#undef TREE_PARAMS
#undef TREE_INSTANCE
#undef TREE_FRAME

} // namespace

} // namespace tree

TreeRecords generateTreesReference(const std::vector<TreeTypeParams>& types, const std::vector<TreeInstance>& instances,
                                   const TreeFrame& frame) {
    TreeRecords records;
    tree::ReferenceContext ctx{&types, &instances, frame, &records};
    for (uint32_t i = 0; i < instances.size(); ++i) {
        records.stems[0].push_back(tree::treeRootStem(instances[i], i, ctx));
    }
    // Breadth first, level by level, as the GPU passes run.
    for (int level = 0; level < TREE_MAX_LEVELS; ++level) {
        const std::vector<StemRecord> stems = records.stems[level];
        for (const StemRecord& stem : stems) {
            for (uint32_t t = 0; t < TREE_WAVE_SIZE; ++t) {
                tree::treeGenerateStem(stem, t, ctx);
            }
        }
    }
    return records;
}

TreeFrame defaultTreeFrame() {
    TreeFrame frame{};
    frame.worldUp = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
    frame.wind = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    frame.timeSeason = glm::vec4(0.0f, 2.0f, 0.0f, 0.0f);
    return frame;
}

TreeRenderSettings defaultTreeRenderSettings(VkExtent2D extent) {
    TreeRenderSettings settings{};
    settings.viewport = glm::vec4(float(extent.width), float(extent.height), 4.0f, 0.5f);
    settings.sun = glm::vec4(glm::normalize(glm::vec3(-0.4f, 0.8f, 0.45f)), 1.6f);
    settings.sky = glm::vec4(0.32f, 0.36f, 0.42f, 0.0f);
    settings.ground = glm::vec4(0.16f, 0.13f, 0.1f, 0.0f);
    settings.leafLod = glm::vec4(24.0f, 8.0f, 4.0f, 0.25f);
    return settings;
}

glm::vec4 treeUpRotation(const glm::vec3& up) {
    const glm::vec3 from(0.0f, 0.0f, 1.0f);
    const glm::vec3 to = glm::normalize(up);
    const float d = glm::dot(from, to);
    if (d < -0.999999f) {
        return glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const glm::vec3 axis = glm::cross(from, to);
    return glm::normalize(glm::vec4(axis.x, axis.y, axis.z, 1.0f + d));
}

namespace tree_presets {

namespace {

// Per-level values for the stem levels 0 to 3.
glm::vec4 levels(float l0, float l1, float l2, float l3) { return glm::vec4(l0, l1, l2, l3); }

} // namespace

TreeTypeParams quakingAspen() {
    TreeTypeParams p{};
    p.shapeBase = glm::vec4(7.0f, 0.4f, 13.0f, 3.0f);
    p.ratios = glm::vec4(3.0f, 0.015f, 1.2f, 0.6f);
    p.leafCount = glm::vec4(25.0f, 0.17f, 1.0f, 0.5f);
    p.leafAngles = glm::vec4(45.0f, 10.0f, 77.0f, 0.0f);
    p.leafShape = glm::vec4(70.0f, 0.0f, 50.0f, 0.4f);
    p.leafLobes = glm::vec4(1.0f, 0.0f, 1.0f, 0.02f);
    p.barkColor = glm::vec4(0.55f, 0.53f, 0.48f, 0.0f);
    p.leafColor = glm::vec4(0.28f, 0.45f, 0.12f, 0.0f);
    p.leafColorFall = glm::vec4(0.85f, 0.65f, 0.1f, 0.0f);
    p.nLength = levels(1.0f, 0.3f, 0.6f, 0.0f);
    p.nLengthV = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nTaper = levels(1.0f, 1.0f, 1.0f, 1.0f);
    p.nSegSplits = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nSplitAngle = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nSplitAngleV = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nCurveRes = levels(3.0f, 5.0f, 3.0f, 1.0f);
    p.nCurve = levels(0.0f, -40.0f, -40.0f, 0.0f);
    p.nCurveBack = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nCurveV = levels(20.0f, 50.0f, 75.0f, 0.0f);
    p.nDownAngle = levels(0.0f, 60.0f, 45.0f, 45.0f);
    p.nDownAngleV = levels(0.0f, -50.0f, 10.0f, 10.0f);
    p.nRotate = levels(0.0f, 140.0f, 140.0f, 77.0f);
    p.nRotateV = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nBranches = levels(1.0f, 50.0f, 30.0f, 10.0f);
    return p;
}

TreeTypeParams blackTupelo() {
    TreeTypeParams p{};
    p.shapeBase = glm::vec4(4.0f, 0.2f, 23.0f, 5.0f);
    p.ratios = glm::vec4(4.0f, 0.015f, 1.3f, 1.0f);
    p.leafCount = glm::vec4(6.0f, 0.3f, 0.5f, 0.5f);
    p.leafAngles = glm::vec4(45.0f, 10.0f, 140.0f, 0.0f);
    p.leafShape = glm::vec4(40.0f, 0.0f, 30.0f, 0.5f);
    p.leafLobes = glm::vec4(1.0f, 0.0f, 1.0f, 0.03f);
    p.barkColor = glm::vec4(0.35f, 0.3f, 0.27f, 0.0f);
    p.leafColor = glm::vec4(0.2f, 0.38f, 0.1f, 0.0f);
    p.leafColorFall = glm::vec4(0.75f, 0.15f, 0.08f, 0.1f);
    p.nLength = levels(1.0f, 0.3f, 0.6f, 0.4f);
    p.nLengthV = levels(0.0f, 0.05f, 0.1f, 0.0f);
    p.nTaper = levels(1.0f, 1.0f, 1.0f, 1.0f);
    p.nSegSplits = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nSplitAngle = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nSplitAngleV = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nCurveRes = levels(10.0f, 10.0f, 10.0f, 1.0f);
    p.nCurve = levels(0.0f, 0.0f, -10.0f, 0.0f);
    p.nCurveBack = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nCurveV = levels(40.0f, 90.0f, 150.0f, 0.0f);
    p.nDownAngle = levels(0.0f, 60.0f, 30.0f, 45.0f);
    p.nDownAngleV = levels(0.0f, -40.0f, 10.0f, 10.0f);
    p.nRotate = levels(0.0f, 140.0f, 140.0f, 140.0f);
    p.nRotateV = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nBranches = levels(1.0f, 50.0f, 25.0f, 12.0f);
    return p;
}

TreeTypeParams sassafras() {
    TreeTypeParams p{};
    p.shapeBase = glm::vec4(2.0f, 0.3f, 15.0f, 3.0f);
    p.ratios = glm::vec4(3.0f, 0.02f, 1.3f, 0.5f);
    p.leafCount = glm::vec4(12.0f, 0.25f, 0.8f, 0.3f);
    p.leafAngles = glm::vec4(40.0f, 15.0f, 137.0f, 10.0f);
    p.leafShape = glm::vec4(45.0f, 0.0f, 35.0f, 0.5f);
    p.leafLobes = glm::vec4(3.0f, 45.0f, 0.75f, 0.03f);
    p.barkColor = glm::vec4(0.42f, 0.33f, 0.26f, 0.0f);
    p.leafColor = glm::vec4(0.3f, 0.5f, 0.15f, 0.0f);
    p.leafColorFall = glm::vec4(0.9f, 0.4f, 0.1f, 0.05f);
    p.nLength = levels(1.0f, 0.35f, 0.5f, 0.0f);
    p.nLengthV = levels(0.0f, 0.05f, 0.1f, 0.0f);
    p.nTaper = levels(1.0f, 1.0f, 1.0f, 1.0f);
    p.nSegSplits = levels(0.15f, 0.1f, 0.0f, 0.0f);
    p.nSplitAngle = levels(25.0f, 20.0f, 0.0f, 0.0f);
    p.nSplitAngleV = levels(5.0f, 5.0f, 0.0f, 0.0f);
    p.nCurveRes = levels(6.0f, 6.0f, 4.0f, 1.0f);
    p.nCurve = levels(0.0f, -20.0f, -20.0f, 0.0f);
    p.nCurveBack = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nCurveV = levels(20.0f, 40.0f, 60.0f, 0.0f);
    p.nDownAngle = levels(0.0f, 55.0f, 40.0f, 40.0f);
    p.nDownAngleV = levels(0.0f, -30.0f, 10.0f, 10.0f);
    p.nRotate = levels(0.0f, 140.0f, 140.0f, 137.0f);
    p.nRotateV = levels(0.0f, 10.0f, 10.0f, 0.0f);
    p.nBranches = levels(1.0f, 35.0f, 15.0f, 0.0f);
    return p;
}

TreeTypeParams pine() {
    TreeTypeParams p{};
    p.shapeBase = glm::vec4(0.0f, 0.2f, 15.0f, 3.0f);
    p.ratios = glm::vec4(3.0f, 0.015f, 1.5f, 0.3f);
    p.leafCount = glm::vec4(100.0f, 0.12f, 0.06f, 0.2f);
    p.leafAngles = glm::vec4(40.0f, 10.0f, 137.0f, 0.0f);
    p.leafShape = glm::vec4(5.0f, 0.0f, 5.0f, 0.5f);
    p.leafLobes = glm::vec4(1.0f, 0.0f, 1.0f, 0.0f);
    p.barkColor = glm::vec4(0.4f, 0.28f, 0.2f, 0.0f);
    p.leafColor = glm::vec4(0.12f, 0.28f, 0.12f, 0.0f);
    p.leafColorFall = glm::vec4(0.12f, 0.28f, 0.12f, 0.0f);
    p.nLength = levels(1.0f, 0.25f, 0.3f, 0.0f);
    p.nLengthV = levels(0.0f, 0.05f, 0.05f, 0.0f);
    p.nTaper = levels(1.0f, 1.0f, 1.0f, 1.0f);
    p.nSegSplits = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nSplitAngle = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nSplitAngleV = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nCurveRes = levels(8.0f, 4.0f, 3.0f, 1.0f);
    p.nCurve = levels(0.0f, 10.0f, 0.0f, 0.0f);
    p.nCurveBack = levels(0.0f, 0.0f, 0.0f, 0.0f);
    p.nCurveV = levels(10.0f, 30.0f, 40.0f, 0.0f);
    p.nDownAngle = levels(0.0f, 80.0f, 40.0f, 40.0f);
    p.nDownAngleV = levels(0.0f, -10.0f, 10.0f, 10.0f);
    p.nRotate = levels(0.0f, 140.0f, 140.0f, 137.0f);
    p.nRotateV = levels(0.0f, 10.0f, 10.0f, 0.0f);
    p.nBranches = levels(1.0f, 60.0f, 20.0f, 0.0f);
    return p;
}

} // namespace tree_presets

} // namespace klartraum
