// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GGUF_GGUF_FILE_HPP
#define KLARTRAUM_GGUF_GGUF_FILE_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace klartraum {

/** @brief Tensor element types of the GGML/GGUF format (values as stored in the file). */
enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
};

/** @brief Name of @p type as llama.cpp prints it (e.g. "Q4_K"). */
const char* ggmlTypeName(GgmlType type);

/** @brief Elements per quantization block of @p type (1 for plain types). */
uint32_t ggmlBlockSize(GgmlType type);

/** @brief Bytes per quantization block of @p type; throws for unknown types. */
uint32_t ggmlBlockBytes(GgmlType type);

/** @brief Bytes of a row of @p elements values of @p type; throws if not a whole number of blocks. */
uint64_t ggmlRowBytes(GgmlType type, uint64_t elements);

/** @brief Value types of GGUF metadata entries. */
enum class GgufValueType : uint32_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
};

/**
 * @brief One GGUF metadata value.
 *
 * Integers, booleans and floats are held both as integer and as double;
 * arrays keep their elements in `strings` (string arrays) or `integers`
 * and `floats` (numeric arrays).
 */
struct GgufValue {
    GgufValueType type = GgufValueType::Uint8;
    GgufValueType arrayType = GgufValueType::Uint8; ///< Element type of an array.
    int64_t integer = 0;
    double number = 0.0;
    std::string string;
    std::vector<std::string> strings;
    std::vector<int64_t> integers;
    std::vector<double> floats;
};

/** @brief Directory entry of one tensor. */
struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions; ///< GGML order: the innermost (contiguous) dimension first.
    GgmlType type = GgmlType::F32;
    uint64_t offset = 0; ///< Byte offset from the start of the file.
    uint64_t bytes = 0;

    uint64_t elementCount() const;
    /** @brief Elements of the innermost dimension, i.e. per row. */
    uint64_t rowLength() const { return dimensions.empty() ? 1 : dimensions[0]; }
    /** @brief Number of rows: the product of all dimensions but the innermost. */
    uint64_t rowCount() const;
};

/**
 * @brief A memory-mapped GGUF (version 2 or 3) file: metadata, tensor
 * directory, and read-only access to the tensor data.
 *
 * Tensor data stays in the mapping; data() returns pointers into it.
 */
class GgufFile {
public:
    /** @throws std::runtime_error If the file cannot be opened or is not a valid GGUF file. */
    explicit GgufFile(const std::string& path);
    ~GgufFile();

    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    const std::string& getPath() const { return path; }
    uint32_t getVersion() const { return version; }
    uint64_t getAlignment() const { return alignment; }
    uint64_t getFileSize() const { return fileSize; }

    const std::map<std::string, GgufValue>& getMetadata() const { return metadata; }
    bool has(const std::string& key) const { return metadata.count(key) != 0; }
    /** @throws std::runtime_error If @p key is missing. */
    const GgufValue& get(const std::string& key) const;

    /** @brief Integer value (any integer or bool type). @throws If missing or not an integer. */
    int64_t getInteger(const std::string& key) const;
    int64_t getInteger(const std::string& key, int64_t fallback) const;
    /** @brief Float value (any numeric type). */
    double getNumber(const std::string& key) const;
    double getNumber(const std::string& key, double fallback) const;
    const std::string& getString(const std::string& key) const;
    std::string getString(const std::string& key, const std::string& fallback) const;
    const std::vector<std::string>& getStringArray(const std::string& key) const;
    const std::vector<int64_t>& getIntegerArray(const std::string& key) const;
    const std::vector<double>& getFloatArray(const std::string& key) const;

    const std::vector<GgufTensorInfo>& getTensors() const { return tensors; }
    bool hasTensor(const std::string& name) const { return tensorIndex.count(name) != 0; }
    /** @throws std::runtime_error If the tensor does not exist. */
    const GgufTensorInfo& getTensor(const std::string& name) const;
    /** @brief The tensor's bytes inside the mapped file. */
    const uint8_t* data(const GgufTensorInfo& tensor) const;

private:
    std::string path;
    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t fileSize = 0;
    std::map<std::string, GgufValue> metadata;
    std::vector<GgufTensorInfo> tensors;
    std::map<std::string, size_t> tensorIndex;

    const uint8_t* mapped = nullptr;
#ifdef _WIN32
    void* fileHandle = nullptr;
    void* mappingHandle = nullptr;
#else
    int fileDescriptor = -1;
#endif

    void map();
    void unmap();
    void parse();
};

} // namespace klartraum

#endif // KLARTRAUM_GGUF_GGUF_FILE_HPP
