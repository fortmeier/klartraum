// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gguf/gguf_layers.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/gguf/gguf_quants.hpp"

namespace klartraum::gguf_layers {

namespace {

struct MatVecPushConstants {
    uint32_t rows;
    uint32_t columns;
    uint32_t rowBytes;
    uint32_t accumulate;
    uint32_t lastTokenOnly;
    uint32_t inputStride;
    uint32_t outputStride;
};

struct RmsNormPushConstants {
    uint32_t width;
    uint32_t rowsPerToken;
    float epsilon;
};

struct AttentionPrepPushConstants {
    uint32_t headDim;
    uint32_t queryHeads;
    uint32_t kvHeads;
    uint32_t ropeDims;
    uint32_t contextLength;
    float ropeBase;
    float epsilon;
};

struct AttentionPushConstants {
    uint32_t headDim;
    uint32_t queryHeads;
    uint32_t kvHeads;
    uint32_t contextLength;
    float scale;
};

struct WidthPushConstants {
    uint32_t width;
};

struct DeltaNetPushConstants {
    uint32_t keyHeads;
    uint32_t valueHeads;
    uint32_t keyDim;
    uint32_t valueDim;
    uint32_t channels;
    float epsilon;
};

template <typename PushConstants>
std::shared_ptr<GeneralComputation<PushConstants>> computation(VulkanContext& vulkanContext, const std::string& shader,
                                                               const PushConstants& constants) {
    auto operation = vulkanContext.create<GeneralComputation<PushConstants>>(shader);
    operation->setPushConstants({constants});
    return operation;
}

// Spreads `groups` workgroups over X and Y within the 65535 per-axis limit.
template <typename Operation>
void setGroups(Operation& operation, uint32_t groups) {
    const uint32_t x = groups < 65535u ? groups : 65535u;
    operation.setGroupCount(x, (groups + x - 1) / x, 1);
}

const char* matVecShader(GgmlType type) {
    switch (type) {
    case GgmlType::F32:
        return "shaders/gguf/matvec_f32.comp.spv";
    case GgmlType::F16:
        return "shaders/gguf/matvec_f16.comp.spv";
    case GgmlType::BF16:
        return "shaders/gguf/matvec_bf16.comp.spv";
    case GgmlType::Q8_0:
        return "shaders/gguf/matvec_q8_0.comp.spv";
    case GgmlType::Q3_K:
        return "shaders/gguf/matvec_q3_k.comp.spv";
    case GgmlType::Q4_K:
        return "shaders/gguf/matvec_q4_k.comp.spv";
    case GgmlType::Q5_K:
        return "shaders/gguf/matvec_q5_k.comp.spv";
    case GgmlType::Q6_K:
        return "shaders/gguf/matvec_q6_k.comp.spv";
    default:
        throw std::runtime_error(std::string("No GGUF matVec kernel for ") + ggmlTypeName(type));
    }
}

// Bytes of one block in the GPU layout.
uint32_t gpuBlockBytes(GgmlType type) {
    switch (type) {
    case GgmlType::Q3_K:
        return 112;
    case GgmlType::Q6_K:
        return 224;
    case GgmlType::Q8_0:
        return 36;
    default:
        return ggmlBlockBytes(type);
    }
}

} // namespace

uint64_t gpuRowBytes(GgmlType type, uint32_t columns) {
    ggmlRowBytes(type, columns); // checks for whole blocks
    return uint64_t(columns / ggmlBlockSize(type)) * gpuBlockBytes(type);
}

void packWeights(GgmlType type, const uint8_t* source, uint32_t rows, uint32_t columns, uint8_t* destination) {
    const uint64_t sourceRow = ggmlRowBytes(type, columns), destinationRow = gpuRowBytes(type, columns);
    if (sourceRow == destinationRow) {
        std::memcpy(destination, source, size_t(sourceRow * rows));
        return;
    }
    const uint32_t from = ggmlBlockBytes(type), to = gpuBlockBytes(type);
    const uint64_t blocks = uint64_t(rows) * (columns / ggmlBlockSize(type));
    for (uint64_t block = 0; block < blocks; ++block) {
        const uint8_t* in = source + block * from;
        uint8_t* out = destination + block * to;
        std::memset(out, 0, to);
        if (type == GgmlType::Q8_0) {
            std::memcpy(out, in, 2);          // half scale
            std::memcpy(out + 4, in + 2, 32); // quants, word aligned
        } else {
            std::memcpy(out, in, from); // trailing padding
        }
    }
}

ComputeGraphElementPtr matVec(VulkanContext& vulkanContext, GgmlType type, uint32_t rows, uint32_t columns,
                              uint32_t inputStride, uint32_t outputStride, bool accumulate, bool lastTokenOnly) {
    if (columns % 16 != 0 || columns % ggmlBlockSize(type) != 0) {
        throw std::runtime_error("GGUF matVec columns must be a multiple of 16 and of the block size");
    }
    if (inputStride % 4 != 0)
        throw std::runtime_error("GGUF matVec input stride must be a multiple of 4");
    MatVecPushConstants constants{
        rows,        columns,     uint32_t(gpuRowBytes(type, columns)), accumulate ? 1u : 0u, lastTokenOnly ? 1u : 0u,
        inputStride, outputStride};
    auto operation = computation(vulkanContext, matVecShader(type), constants);
    constexpr uint32_t rowsPerGroup = 8; // SUBGROUPS * ROWS_PER_SUBGROUP in matvec.glsl
    setGroups(*operation, (rows + rowsPerGroup - 1) / rowsPerGroup);
    return operation;
}

ComputeGraphElementPtr rmsNorm(VulkanContext& vulkanContext, uint32_t width, uint32_t rowsPerToken, float epsilon,
                               uint32_t maxTokens) {
    auto operation = computation(vulkanContext, "shaders/gguf/rms_norm.comp.spv",
                                 RmsNormPushConstants{width, rowsPerToken, epsilon});
    operation->setGroupCount(maxTokens * rowsPerToken, 1, 1);
    return operation;
}

ComputeGraphElementPtr attentionPrep(VulkanContext& vulkanContext, const AttentionShape& shape, uint32_t maxTokens) {
    if (shape.headDim > 256 || shape.ropeDims > shape.headDim || shape.ropeDims % 2 != 0) {
        throw std::runtime_error("Unsupported attention head shape");
    }
    auto operation =
        computation(vulkanContext, "shaders/gguf/attention_prep.comp.spv",
                    AttentionPrepPushConstants{shape.headDim, shape.queryHeads, shape.kvHeads, shape.ropeDims,
                                               shape.contextLength, shape.ropeBase, shape.epsilon});
    operation->setGroupCount(maxTokens, shape.queryHeads + 2 * shape.kvHeads, 1);
    return operation;
}

ComputeGraphElementPtr attention(VulkanContext& vulkanContext, const AttentionShape& shape, uint32_t maxTokens) {
    if (shape.headDim % 32 != 0 || shape.headDim > 256 || shape.queryHeads % shape.kvHeads != 0) {
        throw std::runtime_error("Unsupported attention head shape");
    }
    auto operation = computation(vulkanContext, "shaders/gguf/attention.comp.spv",
                                 AttentionPushConstants{shape.headDim, shape.queryHeads, shape.kvHeads,
                                                        shape.contextLength, 1.0f / std::sqrt(float(shape.headDim))});
    operation->setGroupCount(maxTokens, shape.queryHeads, 1);
    return operation;
}

ComputeGraphElementPtr linearConv(VulkanContext& vulkanContext, uint32_t channels) {
    auto operation = computation(vulkanContext, "shaders/gguf/linear_conv.comp.spv", WidthPushConstants{channels});
    setGroups(*operation, (channels + 63) / 64);
    return operation;
}

ComputeGraphElementPtr gatedDeltaNet(VulkanContext& vulkanContext, const DeltaNetShape& shape) {
    if (shape.keyDim > 128 || shape.valueDim > 128 || shape.valueHeads % shape.keyHeads != 0) {
        throw std::runtime_error("Unsupported delta-net head shape");
    }
    auto operation = computation(vulkanContext, "shaders/gguf/gated_delta.comp.spv",
                                 DeltaNetPushConstants{shape.keyHeads, shape.valueHeads, shape.keyDim, shape.valueDim,
                                                       shape.channels(), shape.epsilon});
    operation->setGroupCount(shape.valueHeads, 1, 1);
    return operation;
}

ComputeGraphElementPtr swiGlu(VulkanContext& vulkanContext, uint32_t width, uint32_t maxTokens) {
    auto operation = computation(vulkanContext, "shaders/gguf/swiglu.comp.spv", WidthPushConstants{width});
    setGroups(*operation, (maxTokens * width + 255) / 256);
    return operation;
}

} // namespace klartraum::gguf_layers
