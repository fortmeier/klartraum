// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - TypeGeometry: Block geometry and row sizes of the supported GGML types
 * - RejectsNonGgufFile: A file without the GGUF magic is rejected
 * - ScalarMetadata: Scalar metadata of every value type is read back with its value
 * - ArrayMetadata: String and numeric arrays are read back element by element
 * - TensorDirectoryWithCustomAlignment: Tensor directory: names, dimensions, types, and
 *   offsets after the aligned header, with a custom alignment
 * - TensorDataAtDirectoryOffset: Tensor data is read from the mapped file at the
 *   directory offset
 * - RejectsTensorOutsideFile: A tensor extending past the end of the file is rejected
 **/

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/gguf/gguf_file.hpp"

using namespace klartraum;

namespace {

// Minimal GGUF v3 writer for test files.
class GgufBuilder {
public:
    void addUint32(const std::string& key, uint32_t value) { entry(key, GgufValueType::Uint32, &value, 4); }
    void addInt8(const std::string& key, int8_t value) { entry(key, GgufValueType::Int8, &value, 1); }
    void addUint8(const std::string& key, uint8_t value) { entry(key, GgufValueType::Uint8, &value, 1); }
    void addInt16(const std::string& key, int16_t value) { entry(key, GgufValueType::Int16, &value, 2); }
    void addUint16(const std::string& key, uint16_t value) { entry(key, GgufValueType::Uint16, &value, 2); }
    void addInt32(const std::string& key, int32_t value) { entry(key, GgufValueType::Int32, &value, 4); }
    void addUint64(const std::string& key, uint64_t value) { entry(key, GgufValueType::Uint64, &value, 8); }
    void addInt64(const std::string& key, int64_t value) { entry(key, GgufValueType::Int64, &value, 8); }
    void addFloat32(const std::string& key, float value) { entry(key, GgufValueType::Float32, &value, 4); }
    void addFloat64(const std::string& key, double value) { entry(key, GgufValueType::Float64, &value, 8); }
    void addBool(const std::string& key, bool value) {
        const uint8_t byte = value;
        entry(key, GgufValueType::Bool, &byte, 1);
    }
    void addString(const std::string& key, const std::string& value) {
        string(metadata, key);
        pod(metadata, uint32_t(GgufValueType::String));
        string(metadata, value);
        ++metadataCount;
    }
    void addStringArray(const std::string& key, const std::vector<std::string>& values) {
        string(metadata, key);
        pod(metadata, uint32_t(GgufValueType::Array));
        pod(metadata, uint32_t(GgufValueType::String));
        pod(metadata, uint64_t(values.size()));
        for (const auto& value : values)
            string(metadata, value);
        ++metadataCount;
    }
    void addInt32Array(const std::string& key, const std::vector<int32_t>& values) {
        string(metadata, key);
        pod(metadata, uint32_t(GgufValueType::Array));
        pod(metadata, uint32_t(GgufValueType::Int32));
        pod(metadata, uint64_t(values.size()));
        for (auto value : values)
            pod(metadata, value);
        ++metadataCount;
    }
    void addFloat32Array(const std::string& key, const std::vector<float>& values) {
        string(metadata, key);
        pod(metadata, uint32_t(GgufValueType::Array));
        pod(metadata, uint32_t(GgufValueType::Float32));
        pod(metadata, uint64_t(values.size()));
        for (auto value : values)
            pod(metadata, value);
        ++metadataCount;
    }
    // Tensor data is placed at the next multiple of `alignment` in the data section.
    void addTensor(const std::string& name, const std::vector<uint64_t>& dimensions, GgmlType type,
                   const std::vector<uint8_t>& bytes) {
        while (data.size() % alignment != 0)
            data.push_back(0);
        string(directory, name);
        pod(directory, uint32_t(dimensions.size()));
        for (auto dimension : dimensions)
            pod(directory, dimension);
        pod(directory, uint32_t(type));
        pod(directory, uint64_t(data.size()));
        data.insert(data.end(), bytes.begin(), bytes.end());
        ++tensorCount;
    }
    void setAlignment(uint32_t value) {
        alignment = value;
        addUint32("general.alignment", value);
    }
    void write(const std::filesystem::path& path) const {
        std::vector<uint8_t> file;
        pod(file, uint32_t(0x46554747u));
        pod(file, uint32_t(3));
        pod(file, tensorCount);
        pod(file, metadataCount);
        file.insert(file.end(), metadata.begin(), metadata.end());
        file.insert(file.end(), directory.begin(), directory.end());
        while (file.size() % alignment != 0)
            file.push_back(0);
        file.insert(file.end(), data.begin(), data.end());
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(file.data()), std::streamsize(file.size()));
    }

private:
    std::vector<uint8_t> metadata, directory, data;
    uint64_t metadataCount = 0, tensorCount = 0;
    uint32_t alignment = 32;

    template <typename T>
    static void pod(std::vector<uint8_t>& out, T value) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
        out.insert(out.end(), bytes, bytes + sizeof(T));
    }
    static void string(std::vector<uint8_t>& out, const std::string& value) {
        pod(out, uint64_t(value.size()));
        out.insert(out.end(), value.begin(), value.end());
    }
    void entry(const std::string& key, GgufValueType type, const void* value, size_t bytes) {
        string(metadata, key);
        pod(metadata, uint32_t(type));
        const auto* raw = static_cast<const uint8_t*>(value);
        metadata.insert(metadata.end(), raw, raw + bytes);
        ++metadataCount;
    }
};

std::filesystem::path outputPath(const std::string& name) {
    std::filesystem::create_directories("build/TestingOutput");
    return std::filesystem::path("build/TestingOutput") / name;
}

} // namespace

TEST(GgufFileTest, TypeGeometry) {
    EXPECT_EQ(ggmlBlockSize(GgmlType::F32), 1u);
    EXPECT_EQ(ggmlBlockBytes(GgmlType::F16), 2u);
    EXPECT_EQ(ggmlBlockSize(GgmlType::Q8_0), 32u);
    EXPECT_EQ(ggmlBlockBytes(GgmlType::Q8_0), 34u);
    EXPECT_EQ(ggmlBlockBytes(GgmlType::Q3_K), 110u);
    EXPECT_EQ(ggmlBlockBytes(GgmlType::Q4_K), 144u);
    EXPECT_EQ(ggmlBlockBytes(GgmlType::Q5_K), 176u);
    EXPECT_EQ(ggmlBlockBytes(GgmlType::Q6_K), 210u);
    EXPECT_EQ(ggmlRowBytes(GgmlType::Q6_K, 5120), 4200u);
    EXPECT_EQ(ggmlRowBytes(GgmlType::BF16, 7), 14u);
    EXPECT_THROW(ggmlRowBytes(GgmlType::Q4_K, 100), std::runtime_error);
    EXPECT_STREQ(ggmlTypeName(GgmlType::Q5_K), "Q5_K");
}

TEST(GgufFileTest, RejectsNonGgufFile) {
    const auto path = outputPath("not_a_model.gguf");
    std::ofstream(path, std::ios::binary) << "this is not a GGUF file at all";
    EXPECT_THROW(GgufFile file(path.string()), std::runtime_error);
}

TEST(GgufFileTest, ScalarMetadata) {
    GgufBuilder builder;
    builder.addString("general.architecture", "qwen35");
    builder.addUint8("u8", 200);
    builder.addInt8("i8", -100);
    builder.addUint16("u16", 60000);
    builder.addInt16("i16", -30000);
    builder.addUint32("u32", 4000000000u);
    builder.addInt32("i32", -2000000000);
    builder.addUint64("u64", 1ull << 40);
    builder.addInt64("i64", -(1ll << 40));
    builder.addFloat32("f32", 1e-6f);
    builder.addFloat64("f64", 1e7);
    builder.addBool("flag", true);
    const auto path = outputPath("gguf_scalars.gguf");
    builder.write(path);

    GgufFile file(path.string());
    EXPECT_EQ(file.getVersion(), 3u);
    EXPECT_EQ(file.getString("general.architecture"), "qwen35");
    EXPECT_EQ(file.getInteger("u8"), 200);
    EXPECT_EQ(file.getInteger("i8"), -100);
    EXPECT_EQ(file.getInteger("u16"), 60000);
    EXPECT_EQ(file.getInteger("i16"), -30000);
    EXPECT_EQ(file.getInteger("u32"), 4000000000ll);
    EXPECT_EQ(file.getInteger("i32"), -2000000000ll);
    EXPECT_EQ(file.getInteger("u64"), 1ll << 40);
    EXPECT_EQ(file.getInteger("i64"), -(1ll << 40));
    EXPECT_FLOAT_EQ(float(file.getNumber("f32")), 1e-6f);
    EXPECT_DOUBLE_EQ(file.getNumber("f64"), 1e7);
    EXPECT_DOUBLE_EQ(file.getNumber("u32"), 4e9);
    EXPECT_EQ(file.getInteger("flag"), 1);
    EXPECT_EQ(file.getInteger("missing", 7), 7);
    EXPECT_EQ(file.getString("missing", "fallback"), "fallback");
    EXPECT_THROW(file.getInteger("general.architecture"), std::runtime_error);
    EXPECT_THROW(file.get("missing"), std::runtime_error);
    EXPECT_TRUE(file.getTensors().empty());
}

TEST(GgufFileTest, ArrayMetadata) {
    GgufBuilder builder;
    builder.addStringArray("tokens", {"!", "Ġthe", "", "日本"});
    builder.addInt32Array("sections", {11, 11, 10, 0});
    builder.addFloat32Array("scores", {0.5f, -1.25f});
    const auto path = outputPath("gguf_arrays.gguf");
    builder.write(path);

    GgufFile file(path.string());
    EXPECT_EQ(file.getStringArray("tokens"), (std::vector<std::string>{"!", "Ġthe", "", "日本"}));
    EXPECT_EQ(file.getIntegerArray("sections"), (std::vector<int64_t>{11, 11, 10, 0}));
    EXPECT_EQ(file.getFloatArray("scores"), (std::vector<double>{0.5, -1.25}));
    EXPECT_THROW(file.getStringArray("sections"), std::runtime_error);
    EXPECT_THROW(file.getFloatArray("sections"), std::runtime_error);
}

TEST(GgufFileTest, TensorDirectoryWithCustomAlignment) {
    GgufBuilder builder;
    builder.setAlignment(64);
    builder.addString("general.name", "tiny");
    std::vector<uint8_t> weights(3 * 4 * sizeof(float));
    std::vector<uint8_t> quantized(2 * 34);
    builder.addTensor("blk.0.weight", {4, 3}, GgmlType::F32, weights);
    builder.addTensor("blk.0.q", {32, 2}, GgmlType::Q8_0, quantized);
    const auto path = outputPath("gguf_directory.gguf");
    builder.write(path);

    GgufFile file(path.string());
    EXPECT_EQ(file.getAlignment(), 64u);
    ASSERT_EQ(file.getTensors().size(), 2u);
    const auto& first = file.getTensor("blk.0.weight");
    EXPECT_EQ(first.dimensions, (std::vector<uint64_t>{4, 3}));
    EXPECT_EQ(first.type, GgmlType::F32);
    EXPECT_EQ(first.rowLength(), 4u);
    EXPECT_EQ(first.rowCount(), 3u);
    EXPECT_EQ(first.elementCount(), 12u);
    EXPECT_EQ(first.bytes, 48u);
    EXPECT_EQ(first.offset % 64, 0u);
    const auto& second = file.getTensor("blk.0.q");
    EXPECT_EQ(second.type, GgmlType::Q8_0);
    EXPECT_EQ(second.bytes, 68u);
    EXPECT_EQ(second.offset, first.offset + 64);
    EXPECT_TRUE(file.hasTensor("blk.0.q"));
    EXPECT_FALSE(file.hasTensor("blk.1.q"));
    EXPECT_THROW(file.getTensor("blk.1.q"), std::runtime_error);
}

TEST(GgufFileTest, TensorDataAtDirectoryOffset) {
    GgufBuilder builder;
    const std::vector<float> values{1.0f, -2.0f, 3.5f, 1e-3f, 7.0f, 8.0f};
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    builder.addTensor("padding", {3}, GgmlType::F16, std::vector<uint8_t>(6, 0xff));
    builder.addTensor("values", {3, 2}, GgmlType::F32, bytes);
    const auto path = outputPath("gguf_data.gguf");
    builder.write(path);

    GgufFile file(path.string());
    const auto& tensor = file.getTensor("values");
    std::vector<float> read(values.size());
    std::memcpy(read.data(), file.data(tensor), tensor.bytes);
    EXPECT_EQ(read, values);
}

TEST(GgufFileTest, RejectsTensorOutsideFile) {
    GgufBuilder builder;
    // The directory claims 16 x 16 floats, but only 16 bytes follow.
    builder.addTensor("truncated", {16, 16}, GgmlType::F32, std::vector<uint8_t>(16));
    const auto path = outputPath("gguf_truncated.gguf");
    builder.write(path);
    EXPECT_THROW(GgufFile file(path.string()), std::runtime_error);
}
