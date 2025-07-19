# klartraum
Klartraum (German for lucid dream) is a real-time neural rendering and inference engine built on top of Vulkan.

# idea
The general idea of Klartraum is to provide an execution environment for pretrained deep learning algorithms that run on a multitude of hardware, given that Vulkan 1.3 is available. Thus, its goal is to enable a wide range of embedded devices that span from single-board computers like Raspberry Pis to Virtual Reality headsets to use algorithms such as
* Gaussian splatting (implemented)
* Convolutional Neural Networks (not implemented yet)
* Diffusion networks (not implemented yet)
* Transformers and LLMs (not implemented yet)

Also it is possible to combine them directly with classical real-time rendering using Vulkan. This way, complex neural rendering pipelines can be created that, for instance, first perform Gaussian Splatting in an embedding space, decode them using a CNN-Decoder, combine it with classic rasterization or raytracing and finally run DLSS to increase resolution and fidelity. 

Alternatively, it will be possible to use Klartraum as a hardware independent neural network inference engine for classical deep learning deployment on edge devices.

# status
Klartraum is in an early stage of development. The core compute graph functionality is implemented, but it is not yet fully optimized or feature-complete.
At this point, it is possible to run Gaussian Splatting in the Klartraum engine, which is the first step towards a fully functional neural rendering engine.
For Gaussian Splatting, there currently exist these limitations that will be addressed in the near future:
- the splatting is not yet fully optimized for performance
- depth values are neither computed nor used, so the splatting is not occluded by other objects
- some rendering bugs occur spontaneously

# design
A major design principle of Klartraum is that all operations shall be prerecorded so that for a single inference only the command buffers have to be submitted, without re-recording them. This is done to free up the CPU as much as possible and reduce latency of the pipeline. For this goal, the Klartraum user creates a directed acyclic graph (DAG) of draw and compute elements, which then gets compiled into a set of Vulkan command buffers. For applications like rendering, it is important to be able to render frames in parallel, so the draw graph directly supports the concept of multiple paths through the graph, for which separate command buffers are recorded.

Klartraum also provides a minimal rendering engine that handles user input for interactive applications.
Nonetheless, it is designed in a way that the render graph can be used without the rendering engine,
so that it can be used in headless applications and in combination with other Vulkan-based rendering engines.
(However, this is not yet implemented and will need another implementation of the `VulkanContext` class)

# example

```cpp
#include <iostream>
#include <filesystem>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"

int main() {
    std::cout << "Wake up, dreamer!" << std::endl;

    klartraum::GlfwFrontend frontend;

    auto& engine = frontend.getKlartraumEngine();

    klartraum::RenderPassPtr renderpass = engine.createRenderPass();

    std::shared_ptr<klartraum::DrawBasics> axes = std::make_shared<klartraum::DrawBasics>(klartraum::DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);

    auto& vulkanContext = engine.getVulkanContext();
    
    auto cameraUBO = renderpass->getCameraUBO();
    cameraUBO->setName("CameraUBO");
    
    std::string spzFile = "./3rdparty/spz/samples/racoonfamily.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting = vulkanContext.create<klartraum::VulkanGaussianSplatting>(renderpass, cameraUBO, spzFile);
    
    engine.add(splatting);

    std::shared_ptr<klartraum::InterfaceCameraOrbit> cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit->setAzimuth(0.9);
    cameraOrbit->setElevation(-0.5);
    cameraOrbit->setPosition({-0.5, 0.0, 0.5});
    cameraOrbit->setDistance(1.0);
    engine.setInterfaceCamera(cameraOrbit);
    engine.setCameraUBO(cameraUBO);

    frontend.loop();

    return 0;
}

```
# Build
Currently, Klartraum can only be build on Windows with Visual Studio 2022 on the x64 architecture.

```bash
git clone https://github.com/fortmeier/klartraum.git
git submodule update --init --recursive

mkdir build
cd build

cmake ..
cmake --build . --target gaussian_splatting_example

cd ..
# on Windows
.\build\examples\Debug\gaussian_splatting_example.exe

# on Linux
# t.b.d.
```



