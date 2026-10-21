// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_proxies.hpp"

#include <algorithm>

namespace klartraum {

TreeProxies::TreeProxies(VulkanContext& vulkanContext, TreePlacementTiles& tiles, std::shared_ptr<CameraUboType> camera,
                         std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent,
                         TreeRenderInputs inputs, TreeProxyConfig config)
    : config(config) {
    visible = std::make_shared<BufferElement<VulkanBuffer<TreeSite>>>(vulkanContext, config.capacity);
    visible->setName("TreeProxyVisible");
    counter = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 4);
    counter->setName("TreeProxyCounter");
    counter->setRecordToZero(true);

    cull =
        std::make_shared<GeneralComputation<TreeProxyCullPush>>(vulkanContext, "shaders/tree/tree_place_cull.comp.spv");
    cull->setName("TreeProxyCull");
    tiles.getConfig().connectTo(*cull, 0);
    cull->setInput(tiles.getSlotTable(), 1);
    tiles.getSites().connectTo(*cull, 2);
    tiles.getBounds().connectTo(*cull, 3);
    cull->setInput(camera, 4);
    cull->setInput(settings, 5);
    cull->setInput(visible, 6);
    cull->setInput(counter, 7);
    near = std::make_shared<HostValues<glm::vec4>>(vulkanContext, std::vector<glm::vec4>{glm::vec4(-1.0f)});
    near->setName("TreeProxyNear");
    cull->setInput(near, 8);
    cull->setGroupCountX(tiles.getTileConfig().tiles.y);
    cull->setPushConstants({TreeProxyCullPush{config.capacity, config.minPixels}});

    auto drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(
        vulkanContext, 1u,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    drawArgs->setName("TreeProxyDrawArgs");
    dispatchArgs = std::make_shared<DispatchIndirectCommandBufferElement>(
        vulkanContext, 1u,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    dispatchArgs->setName("TreeProxyDispatchArgs");
    args = std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_place_args.comp.spv");
    args->setName("TreeProxyArgs");
    args->setInput(cull, 0, 7);
    args->setInput(drawArgs, 1);
    args->setInput(dispatchArgs, 2);
    args->setGroupCountX(1);
    args->setPushConstants({glm::uvec2(config.capacity, kVertices)});

    StorageDrawConfig drawConfig;
    drawConfig.vertexShader = "shaders/tree/tree_proxy.vert.spv";
    drawConfig.fragmentShader = "shaders/tree/tree_proxy.frag.spv";
    drawConfig.extent = extent;
    draw = std::make_shared<StorageDrawComponent>(
        drawConfig, std::vector<BufferRef>{{cull, 6},
                                           {settings, -1},
                                           bufferOrPlaceholder(vulkanContext, inputs.sceneDepth),
                                           bufferOrPlaceholder(vulkanContext, inputs.light)});
    StorageDrawComponent::Draw d;
    d.indirect = {args, 1};
    draw->addDraw(std::move(d));
}

std::vector<uint32_t> TreeProxies::readCounter(uint32_t pathId) {
    std::vector<uint32_t> values(4, 0u);
    counter->getBuffer(pathId).memcopyTo(values);
    return values;
}

std::vector<TreeSite> TreeProxies::readVisible(uint32_t pathId) {
    const uint32_t count = std::min(readCounter(pathId)[0], config.capacity);
    std::vector<TreeSite> values(config.capacity);
    visible->getBuffer(pathId).memcopyTo(values);
    values.resize(count);
    return values;
}

TreeProxyGaussians::TreeProxyGaussians(VulkanContext& vulkanContext, TreeProxies& proxies,
                                       const GaussianSoABuffers& source, uint32_t poolOffset, uint32_t poolTrees,
                                       std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                                       std::shared_ptr<CameraUboType> camera, BufferRef light) {
    const uint32_t trees = std::min(poolTrees, proxies.getConfig().capacity);
    clear = std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_lod_clear.comp.spv");
    clear->setName("TreeProxyGaussianClear");
    source.colAlpha.connectTo(*clear, 0);
    clear->setGroupCountX((4u * trees + 255u) / 256u);
    clear->setPushConstants({glm::uvec2(poolOffset, 4u * trees)});

    write =
        std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_proxy_gaussians.comp.spv");
    write->setName("TreeProxyGaussians");
    source.pos.connectTo(*write, 0);
    source.rot.connectTo(*write, 1);
    source.scale.connectTo(*write, 2);
    write->setInput(clear, 3, 0);
    proxies.getVisible().connectTo(*write, 4);
    proxies.getCounter().connectTo(*write, 5);
    write->setInput(settings, 6);
    const BufferRef lightRef = bufferOrPlaceholder(vulkanContext, light);
    write->setInput(lightRef.element, 7, lightRef.slot);
    write->setInput(camera, 8);
    write->addDependency(proxies.getElement());
    write->setDynamicGroupDispatchParams(proxies.getDispatchArgs());
    write->setPushConstants({glm::uvec2(poolOffset, trees)});

    gaussians = source;
    gaussians.pos = BufferRef{write, 0};
    gaussians.rot = BufferRef{write, 1};
    gaussians.scale = BufferRef{write, 2};
    gaussians.colAlpha = BufferRef{write, 3};
}

} // namespace klartraum
