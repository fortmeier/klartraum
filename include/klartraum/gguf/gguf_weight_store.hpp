// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GGUF_GGUF_WEIGHT_STORE_HPP
#define KLARTRAUM_GGUF_GGUF_WEIGHT_STORE_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/gguf/gguf_file.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/**
 * @brief GPU tensors of a GGUF file's weights, created on request while a network is built and uploaded in one batch
 * by upload().
 *
 * Matrices for gguf_layers::matMul() / matVec() stay in their GGUF encoding (encoded()); vectors such as norm scales
 * and biases are dequantized to float (floats()). Both can reorder rows on upload, e.g. to make the rotate-half pairs
 * of a NEOX rotary embedding adjacent.
 */
class GgufWeightStore {
public:
    using FloatTensor = TensorElementSinglePath<float>;
    using WordTensor = TensorElementSinglePath<uint32_t>;
    /** Maps a destination row to the source row it takes. */
    using RowOrder = std::function<uint64_t(uint64_t)>;

    /** @p file must outlive the store. */
    GgufWeightStore(VulkanContext& vulkanContext, const GgufFile& file);

    /** @brief The encoded matrix @p name, rows reordered by @p order. @throws For unsupported types. */
    std::shared_ptr<WordTensor> encoded(const std::string& name, RowOrder order = {});

    /** @brief The tensor @p name dequantized to float; @p order reorders its elements. */
    std::shared_ptr<FloatTensor> floats(const std::string& name, RowOrder order = {});

    /** @brief Uploads all tensors requested so far that are not uploaded yet. */
    void upload(VkQueue queue = VK_NULL_HANDLE);

    /** @brief GPU bytes of all requested tensors. */
    uint64_t getBytes() const { return bytes; }

private:
    template <typename Tensor>
    struct Entry {
        std::shared_ptr<Tensor> tensor;
        RowOrder order;
        bool uploaded = false;
    };

    VulkanContext& vulkanContext;
    const GgufFile& file;
    uint64_t bytes = 0;
    std::map<std::string, Entry<WordTensor>> encodedTensors;
    std::map<std::string, Entry<FloatTensor>> floatTensors;
};

} // namespace klartraum

#endif // KLARTRAUM_GGUF_GGUF_WEIGHT_STORE_HPP
