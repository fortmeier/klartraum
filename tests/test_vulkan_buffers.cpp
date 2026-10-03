#include <gtest/gtest.h>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"


TEST(VulkanBuffer, memcopy) {
    klartraum::HeadlessFrontend frontend;

    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();
    auto& device = vulkanContext.getDevice();
    
    klartraum::VulkanBuffer<float> buffer(vulkanContext, 7);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    buffer.memcopyFrom(data);
    std::vector<float> data2(7);
    buffer.memcopyTo(data2);
    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(data[i], data2[i]);
    }
    return;
}
// Device-local buffers uploaded through a BatchedUpload with 1 KiB staging
// chunks: 40 buffers of 256 bytes take 10 submissions, one of 2 KiB (larger
// than a chunk) one more; a host-visible buffer is written right away.
TEST(VulkanBuffer, batchedUploadGathersCopies) {
    klartraum::HeadlessFrontend frontend;
    auto& vulkanContext = frontend.getKlartraumEngine().getVulkanContext();

    constexpr VkBufferUsageFlags kUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto values = [](uint32_t count, float first) {
        std::vector<float> v(count);
        for (uint32_t i = 0; i < count; ++i) v[i] = first + static_cast<float>(i);
        return v;
    };

    klartraum::BatchedUpload batch(vulkanContext, vulkanContext.getBackgroundQueue(), 1024);
    std::vector<klartraum::VulkanBuffer<float>> small;
    small.reserve(40);
    for (int b = 0; b < 40; ++b) {
        small.emplace_back(vulkanContext, 64, kUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const auto data = values(64, 1000.0f * b);
        small.back().memcopyFrom(batch, data.data(), data.size());
    }
    EXPECT_EQ(batch.submissions(), 9u);  // the 40th chunk is still gathered

    klartraum::VulkanBuffer<float> large(vulkanContext, 512, kUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto largeData = values(512, -7.0f);
    large.memcopyFrom(batch, largeData.data(), largeData.size());
    EXPECT_EQ(batch.submissions(), 10u);

    klartraum::VulkanBuffer<float> mapped(vulkanContext, 8);
    const auto mappedData = values(8, 0.5f);
    mapped.memcopyFrom(batch, reinterpret_cast<const char*>(mappedData.data()), mappedData.size() * sizeof(float));
    EXPECT_EQ(batch.submissions(), 10u);

    batch.submit();
    EXPECT_EQ(batch.submissions(), 11u);
    batch.submit();  // nothing left
    EXPECT_EQ(batch.submissions(), 11u);

    for (int b = 0; b < 40; ++b) {
        std::vector<float> readBack(64);
        small[b].memcopyTo(readBack);
        EXPECT_EQ(readBack, values(64, 1000.0f * b)) << "buffer " << b;
    }
    std::vector<float> largeBack(512);
    large.memcopyTo(largeBack);
    EXPECT_EQ(largeBack, largeData);
    std::vector<float> mappedBack(8);
    mapped.memcopyTo(mappedBack);
    EXPECT_EQ(mappedBack, mappedData);
}
