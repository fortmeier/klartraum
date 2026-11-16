// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gguf/gguf_weight_store.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "klartraum/batched_upload.hpp"
#include "klartraum/gguf/gguf_layers.hpp"
#include "klartraum/gguf/gguf_quants.hpp"

namespace klartraum {

namespace gl = gguf_layers;

GgufWeightStore::GgufWeightStore(VulkanContext& vulkanContext, const GgufFile& file)
    : vulkanContext(vulkanContext),
      file(file) {}

std::shared_ptr<GgufWeightStore::WordTensor> GgufWeightStore::encoded(const std::string& name, RowOrder order) {
    auto& entry = encodedTensors[name];
    if (!entry.tensor) {
        const auto& info = file.getTensor(name);
        if (!isSupportedWeightType(info.type))
            throw std::runtime_error("GgufWeightStore: tensor " + name + " has unsupported type " +
                                     ggmlTypeName(info.type));
        const uint64_t size = gl::gpuRowBytes(info.type, uint32_t(info.rowLength())) * info.rowCount();
        const uint64_t words = (size + gl::kWeightPaddingBytes + 3) / 4;
        if (words > UINT32_MAX)
            throw std::runtime_error("GgufWeightStore: tensor " + name + " is too large");
        entry.tensor = vulkanContext.create<WordTensor>(
            std::vector<uint32_t>{uint32_t(words)},
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        entry.tensor->setName(name.c_str());
        entry.order = std::move(order);
        bytes += words * 4;
    }
    return entry.tensor;
}

std::shared_ptr<GgufWeightStore::FloatTensor> GgufWeightStore::floats(const std::string& name, RowOrder order) {
    auto& entry = floatTensors[name];
    if (!entry.tensor) {
        const auto elements = uint32_t(file.getTensor(name).elementCount());
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        entry.tensor = vulkanContext.create<FloatTensor>(std::vector<uint32_t>{elements}, usage, usage,
                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        entry.tensor->setName(name.c_str());
        entry.order = std::move(order);
        bytes += entry.tensor->getBufferMemSize();
    }
    return entry.tensor;
}

void GgufWeightStore::upload(VkQueue queue) {
    BatchedUpload batch(vulkanContext, queue);
    std::vector<uint8_t> packed;
    for (auto& [name, entry] : encodedTensors) {
        if (entry.uploaded)
            continue;
        // Repacked into the GPU block layout in slices of rows, so no copy of a whole large tensor is held.
        const auto& info = file.getTensor(name);
        const uint32_t columns = uint32_t(info.rowLength()), rows = uint32_t(info.rowCount());
        const uint64_t sourceRow = ggmlRowBytes(info.type, columns), gpuRow = gl::gpuRowBytes(info.type, columns);
        const uint32_t sliceRows = uint32_t(std::max<uint64_t>(1, (uint64_t(64) << 20) / gpuRow));
        for (uint32_t first = 0; first < rows; first += sliceRows) {
            const uint32_t count = std::min(sliceRows, rows - first);
            packed.resize(size_t(gpuRow * count));
            if (entry.order) {
                for (uint32_t row = 0; row < count; ++row)
                    gl::packWeights(info.type, file.data(info) + entry.order(first + row) * sourceRow, 1, columns,
                                    packed.data() + row * gpuRow);
            } else {
                gl::packWeights(info.type, file.data(info) + first * sourceRow, count, columns, packed.data());
            }
            batch.add(entry.tensor->getDataBuffer().getBuffer(), packed.data(), packed.size(), first * gpuRow);
        }
        entry.uploaded = true;
    }
    std::vector<float> values, ordered;
    for (auto& [name, entry] : floatTensors) {
        if (entry.uploaded)
            continue;
        const auto& info = file.getTensor(name);
        values.resize(size_t(info.elementCount()));
        const uint64_t rowBytes = ggmlRowBytes(info.type, info.rowLength());
        for (uint64_t row = 0; row < info.rowCount(); ++row)
            dequantizeRow(info.type, file.data(info) + row * rowBytes, values.data() + row * info.rowLength(),
                          size_t(info.rowLength()));
        if (entry.order) {
            ordered.resize(values.size());
            for (size_t i = 0; i < values.size(); ++i)
                ordered[i] = values[entry.order(i)];
            values.swap(ordered);
        }
        entry.tensor->getDataBuffer().memcopyFrom(batch, values.data(), values.size());
        entry.uploaded = true;
    }
    batch.submit();
}

} // namespace klartraum
