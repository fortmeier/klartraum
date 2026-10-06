// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gguf/gguf_file.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace klartraum {

namespace {

struct TypeTraits {
    GgmlType type;
    const char* name;
    uint32_t blockSize;
    uint32_t blockBytes;
};

// Block geometry of the GGML tensor types, as documented for the GGUF format.
constexpr TypeTraits kTypes[] = {
    {GgmlType::F32, "F32", 1, 4},       {GgmlType::F16, "F16", 1, 2},       {GgmlType::Q4_0, "Q4_0", 32, 18},
    {GgmlType::Q4_1, "Q4_1", 32, 20},   {GgmlType::Q5_0, "Q5_0", 32, 22},   {GgmlType::Q5_1, "Q5_1", 32, 24},
    {GgmlType::Q8_0, "Q8_0", 32, 34},   {GgmlType::Q8_1, "Q8_1", 32, 36},   {GgmlType::Q2_K, "Q2_K", 256, 84},
    {GgmlType::Q3_K, "Q3_K", 256, 110}, {GgmlType::Q4_K, "Q4_K", 256, 144}, {GgmlType::Q5_K, "Q5_K", 256, 176},
    {GgmlType::Q6_K, "Q6_K", 256, 210}, {GgmlType::Q8_K, "Q8_K", 256, 292}, {GgmlType::I8, "I8", 1, 1},
    {GgmlType::I16, "I16", 1, 2},       {GgmlType::I32, "I32", 1, 4},       {GgmlType::I64, "I64", 1, 8},
    {GgmlType::F64, "F64", 1, 8},       {GgmlType::BF16, "BF16", 1, 2},
};

const TypeTraits* findType(GgmlType type) {
    for (const auto& traits : kTypes) {
        if (traits.type == type)
            return &traits;
    }
    return nullptr;
}

const TypeTraits& requireType(GgmlType type) {
    const auto* traits = findType(type);
    if (!traits)
        throw std::runtime_error("Unsupported GGML tensor type " + std::to_string(uint32_t(type)));
    return *traits;
}

// Bounds-checked little-endian reader over the mapped header.
class Reader {
public:
    Reader(const uint8_t* data, uint64_t size)
        : data(data),
          size(size) {}

    template <typename T>
    T read() {
        require(sizeof(T));
        T value;
        std::memcpy(&value, data + position, sizeof(T));
        position += sizeof(T);
        return value;
    }

    std::string readString() {
        const uint64_t length = read<uint64_t>();
        require(length);
        std::string value(reinterpret_cast<const char*>(data + position), size_t(length));
        position += length;
        return value;
    }

    uint64_t getPosition() const { return position; }

private:
    const uint8_t* data;
    uint64_t size;
    uint64_t position = 0;

    void require(uint64_t bytes) const {
        if (bytes > size || position > size - bytes)
            throw std::runtime_error("GGUF file is truncated");
    }
};

bool isInteger(GgufValueType type) {
    switch (type) {
    case GgufValueType::Uint8:
    case GgufValueType::Int8:
    case GgufValueType::Uint16:
    case GgufValueType::Int16:
    case GgufValueType::Uint32:
    case GgufValueType::Int32:
    case GgufValueType::Uint64:
    case GgufValueType::Int64:
    case GgufValueType::Bool:
        return true;
    default:
        return false;
    }
}

// Reads one scalar (non-array, non-string) value into `integer` and `number`.
void readScalar(Reader& reader, GgufValueType type, int64_t& integer, double& number) {
    switch (type) {
    case GgufValueType::Uint8:
        integer = reader.read<uint8_t>();
        break;
    case GgufValueType::Int8:
        integer = reader.read<int8_t>();
        break;
    case GgufValueType::Uint16:
        integer = reader.read<uint16_t>();
        break;
    case GgufValueType::Int16:
        integer = reader.read<int16_t>();
        break;
    case GgufValueType::Uint32:
        integer = reader.read<uint32_t>();
        break;
    case GgufValueType::Int32:
        integer = reader.read<int32_t>();
        break;
    case GgufValueType::Uint64:
        integer = int64_t(reader.read<uint64_t>());
        break;
    case GgufValueType::Int64:
        integer = reader.read<int64_t>();
        break;
    case GgufValueType::Bool:
        integer = reader.read<uint8_t>() != 0;
        break;
    case GgufValueType::Float32:
        number = reader.read<float>();
        integer = int64_t(number);
        return;
    case GgufValueType::Float64:
        number = reader.read<double>();
        integer = int64_t(number);
        return;
    default:
        throw std::runtime_error("Invalid GGUF scalar type " + std::to_string(uint32_t(type)));
    }
    number = double(integer);
}

GgufValue readValue(Reader& reader, GgufValueType type) {
    GgufValue value;
    value.type = type;
    if (type == GgufValueType::String) {
        value.string = reader.readString();
    } else if (type == GgufValueType::Array) {
        value.arrayType = GgufValueType(reader.read<uint32_t>());
        const uint64_t count = reader.read<uint64_t>();
        if (value.arrayType == GgufValueType::String) {
            value.strings.reserve(size_t(count));
            for (uint64_t i = 0; i < count; ++i)
                value.strings.push_back(reader.readString());
        } else if (value.arrayType == GgufValueType::Array) {
            throw std::runtime_error("Nested GGUF arrays are not supported");
        } else {
            const bool integers = isInteger(value.arrayType);
            (integers ? value.integers.reserve(size_t(count)) : value.floats.reserve(size_t(count)));
            for (uint64_t i = 0; i < count; ++i) {
                int64_t integer = 0;
                double number = 0.0;
                readScalar(reader, value.arrayType, integer, number);
                if (integers)
                    value.integers.push_back(integer);
                else
                    value.floats.push_back(number);
            }
        }
    } else {
        readScalar(reader, type, value.integer, value.number);
    }
    return value;
}

} // namespace

const char* ggmlTypeName(GgmlType type) {
    const auto* traits = findType(type);
    return traits ? traits->name : "unknown";
}

uint32_t ggmlBlockSize(GgmlType type) { return requireType(type).blockSize; }

uint32_t ggmlBlockBytes(GgmlType type) { return requireType(type).blockBytes; }

uint64_t ggmlRowBytes(GgmlType type, uint64_t elements) {
    const auto& traits = requireType(type);
    if (elements % traits.blockSize != 0) {
        throw std::runtime_error(std::string("Row of ") + std::to_string(elements) +
                                 " elements is not a multiple of the " + traits.name + " block size");
    }
    return elements / traits.blockSize * traits.blockBytes;
}

uint64_t GgufTensorInfo::elementCount() const {
    uint64_t count = 1;
    for (const auto dimension : dimensions)
        count *= dimension;
    return count;
}

uint64_t GgufTensorInfo::rowCount() const {
    uint64_t count = 1;
    for (size_t i = 1; i < dimensions.size(); ++i)
        count *= dimensions[i];
    return count;
}

GgufFile::GgufFile(const std::string& path)
    : path(path) {
    map();
    try {
        parse();
    } catch (...) {
        unmap();
        throw;
    }
}

GgufFile::~GgufFile() { unmap(); }

#ifdef _WIN32
void GgufFile::map() {
    fileHandle = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    if (fileHandle == INVALID_HANDLE_VALUE) {
        fileHandle = nullptr;
        throw std::runtime_error("Could not open GGUF file " + path);
    }
    LARGE_INTEGER size;
    GetFileSizeEx(fileHandle, &size);
    fileSize = uint64_t(size.QuadPart);
    if (fileSize == 0)
        throw std::runtime_error("GGUF file is empty: " + path);
    mappingHandle = CreateFileMappingA(fileHandle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mappingHandle)
        throw std::runtime_error("Could not map GGUF file " + path);
    mapped = static_cast<const uint8_t*>(MapViewOfFile(mappingHandle, FILE_MAP_READ, 0, 0, 0));
    if (!mapped)
        throw std::runtime_error("Could not map GGUF file " + path);
}

void GgufFile::unmap() {
    if (mapped)
        UnmapViewOfFile(mapped);
    if (mappingHandle)
        CloseHandle(mappingHandle);
    if (fileHandle)
        CloseHandle(fileHandle);
    mapped = nullptr;
    mappingHandle = nullptr;
    fileHandle = nullptr;
}
#else
void GgufFile::map() {
    fileDescriptor = open(path.c_str(), O_RDONLY);
    if (fileDescriptor < 0)
        throw std::runtime_error("Could not open GGUF file " + path);
    struct stat status{};
    if (fstat(fileDescriptor, &status) != 0 || status.st_size <= 0) {
        unmap();
        throw std::runtime_error("GGUF file is empty or unreadable: " + path);
    }
    fileSize = uint64_t(status.st_size);
    void* address = mmap(nullptr, size_t(fileSize), PROT_READ, MAP_SHARED, fileDescriptor, 0);
    if (address == MAP_FAILED) {
        unmap();
        throw std::runtime_error("Could not map GGUF file " + path);
    }
    mapped = static_cast<const uint8_t*>(address);
}

void GgufFile::unmap() {
    if (mapped)
        munmap(const_cast<uint8_t*>(mapped), size_t(fileSize));
    if (fileDescriptor >= 0)
        close(fileDescriptor);
    mapped = nullptr;
    fileDescriptor = -1;
}
#endif

void GgufFile::parse() {
    Reader reader(mapped, fileSize);
    const uint32_t magic = reader.read<uint32_t>();
    if (magic != 0x46554747u)
        throw std::runtime_error("Not a GGUF file: " + path); // "GGUF"
    version = reader.read<uint32_t>();
    if (version != 2 && version != 3) {
        throw std::runtime_error("Unsupported GGUF version " + std::to_string(version) + " in " + path);
    }
    const uint64_t tensorCount = reader.read<uint64_t>();
    const uint64_t metadataCount = reader.read<uint64_t>();

    for (uint64_t i = 0; i < metadataCount; ++i) {
        std::string key = reader.readString();
        const auto type = GgufValueType(reader.read<uint32_t>());
        metadata[std::move(key)] = readValue(reader, type);
    }
    if (has("general.alignment")) {
        alignment = uint64_t(getInteger("general.alignment"));
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::runtime_error("GGUF alignment must be a power of two");
        }
    }

    tensors.reserve(size_t(tensorCount));
    for (uint64_t i = 0; i < tensorCount; ++i) {
        GgufTensorInfo tensor;
        tensor.name = reader.readString();
        const uint32_t rank = reader.read<uint32_t>();
        if (rank > 8)
            throw std::runtime_error("GGUF tensor " + tensor.name + " has too many dimensions");
        for (uint32_t d = 0; d < rank; ++d)
            tensor.dimensions.push_back(reader.read<uint64_t>());
        tensor.type = GgmlType(reader.read<uint32_t>());
        tensor.offset = reader.read<uint64_t>();
        tensorIndex[tensor.name] = tensors.size();
        tensors.push_back(std::move(tensor));
    }

    // Tensor data starts at the first aligned offset after the header;
    // directory offsets are relative to it.
    const uint64_t dataStart = (reader.getPosition() + alignment - 1) / alignment * alignment;
    for (auto& tensor : tensors) {
        tensor.bytes = ggmlRowBytes(tensor.type, tensor.rowLength()) * tensor.rowCount();
        tensor.offset += dataStart;
        if (tensor.offset > fileSize || tensor.bytes > fileSize - tensor.offset) {
            throw std::runtime_error("GGUF tensor " + tensor.name + " lies outside the file");
        }
    }
}

const GgufValue& GgufFile::get(const std::string& key) const {
    const auto it = metadata.find(key);
    if (it == metadata.end())
        throw std::runtime_error("GGUF metadata key not found: " + key);
    return it->second;
}

int64_t GgufFile::getInteger(const std::string& key) const {
    const auto& value = get(key);
    if (!isInteger(value.type))
        throw std::runtime_error("GGUF metadata " + key + " is not an integer");
    return value.integer;
}

int64_t GgufFile::getInteger(const std::string& key, int64_t fallback) const {
    return has(key) ? getInteger(key) : fallback;
}

double GgufFile::getNumber(const std::string& key) const {
    const auto& value = get(key);
    if (value.type == GgufValueType::String || value.type == GgufValueType::Array) {
        throw std::runtime_error("GGUF metadata " + key + " is not a number");
    }
    return value.number;
}

double GgufFile::getNumber(const std::string& key, double fallback) const {
    return has(key) ? getNumber(key) : fallback;
}

const std::string& GgufFile::getString(const std::string& key) const {
    const auto& value = get(key);
    if (value.type != GgufValueType::String)
        throw std::runtime_error("GGUF metadata " + key + " is not a string");
    return value.string;
}

std::string GgufFile::getString(const std::string& key, const std::string& fallback) const {
    return has(key) ? getString(key) : fallback;
}

const std::vector<std::string>& GgufFile::getStringArray(const std::string& key) const {
    const auto& value = get(key);
    if (value.type != GgufValueType::Array || value.arrayType != GgufValueType::String) {
        throw std::runtime_error("GGUF metadata " + key + " is not a string array");
    }
    return value.strings;
}

const std::vector<int64_t>& GgufFile::getIntegerArray(const std::string& key) const {
    const auto& value = get(key);
    if (value.type != GgufValueType::Array || !isInteger(value.arrayType)) {
        throw std::runtime_error("GGUF metadata " + key + " is not an integer array");
    }
    return value.integers;
}

const std::vector<double>& GgufFile::getFloatArray(const std::string& key) const {
    const auto& value = get(key);
    if (value.type != GgufValueType::Array || value.arrayType == GgufValueType::String || isInteger(value.arrayType)) {
        throw std::runtime_error("GGUF metadata " + key + " is not a float array");
    }
    return value.floats;
}

const GgufTensorInfo& GgufFile::getTensor(const std::string& name) const {
    const auto it = tensorIndex.find(name);
    if (it == tensorIndex.end())
        throw std::runtime_error("GGUF tensor not found: " + name);
    return tensors[it->second];
}

const uint8_t* GgufFile::data(const GgufTensorInfo& tensor) const { return mapped + tensor.offset; }

} // namespace klartraum
