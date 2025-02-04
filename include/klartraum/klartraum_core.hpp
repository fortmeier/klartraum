#ifndef KLARTRAUM_CORE_HPP
#define KLARTRAUM_CORE_HPP

#include <vector>
#include <queue>
#include <optional>
#include <memory>


#include "klartraum/vulkan_kernel.hpp"
#include "klartraum/backend_config.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/camera.hpp"
#include "klartraum/interface_camera.hpp"
#include "klartraum/events.hpp"


namespace klartraum {

class KlartraumCore {
public:
    /*
    KlartraumCore();
    ~KlartraumCore();


    void loop();

    */
    void initialize();
    void shutdown();

    void step();

    std::queue<std::unique_ptr<Event> >& getEventQueue();

    void setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera);


    void addDrawComponent(std::unique_ptr<DrawComponent> drawComponent);

    VulkanKernel& getVulkanKernel();



private:
    VulkanKernel vulkanKernel;

    std::shared_ptr<InterfaceCamera> interfaceCamera;

    std::queue<std::unique_ptr<Event> > eventQueue;

    std::vector<std::unique_ptr<DrawComponent> > drawComponents;



};

} // namespace klartraum

#endif // KLARTRAUM_CORE_HPP