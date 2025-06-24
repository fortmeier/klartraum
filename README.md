# klartraum
Klartraum (German for lucid dream) is a real-time neural rendering and inference engine build on top of Vulkan.

# idea
The general idea is to provide an execution environment for pretrained deep learning algorithms that runs on a multitude of hardware, given that Vulkan 1.3 is available. Thus, it enables a wide range of embedded devices that span from single-board computers as Raspberry Pies to Virtual Reality headsets to use algorithms as
* Gaussian splatting
* Convolutional Neural Networks
* Diffusion networks
* Transformers and LLMs

Also it is possible to combine them directly with classical real-time rendering using Vulkan. This way, complex neural rendering pipelines can be created that for instance first performs Gaussian Splatting in an embedding space, decodes them using a CNN-Decoder, combines it with classic rasterization or raytracing and finally runs DLSS to increase resolution and fidelity. 

Alternatively, it can be simply used as a hardware independent neural network inference engine for classical deep learning deployment.

# status
Klartraum is in an early stage of development. The core functionality is implemented, but it is not yet fully optimized or feature-complete.
At this point, it is possible to run Gaussian Splatting in the Klartraum engine, which is the first step towards a fully functional neural rendering engine.
For Gaussian Splatting, there currently exists these limitations that will be addressed in the near future:
- computation of 2d gaussian covariance matrices is faulty, which leads to wrong splatting results
- the splatting is not yet optimized for performance, so it is not yet real-time capable for larger datasets
- depth values are neither computed nor used, so the splatting is not occluded by other objects
- color computation based on spherical harmonics is not yet implemented

# design
A major design principle of Klartraum is that all operations shall be as prerecorded as possible so that for a single inference only the command buffers have to be submitted, without re-recording. This is done to free up the CPU as much as possible and reduce latency of the pipeline. For this goal, the Klartraum user creates a directed acyclic graph (DAG) of draw and compute elements, which then gets compiled into a set of VkSubmitInfo blocks with semaphore dependencies between the respective Vulkan command buffers. For applications as rendering, it is important to be able to render frames in parallel, so the draw graph directly supports the concept of multiple paths through the graph, for which separate command buffers are recorded.

Klartraum also provides a minimal rendering engine that handles user input for interactive applications.
Nonetheless, it is design in a way that the render graph can be used without the rendering engine,
so that it can be used in headless applications an in combination with other rendering engines.
(however, this is not yet implemented and will need an other implementation of the `VulkanContext` class)

# example

```cpp
#include <iostream>

#include "klartraum/draw_basics.hpp"
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"

int main() {
    std::cout << "Wake up, dreamer!" << std::endl;

    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumEngine();

    klartraum::RenderPassPtr renderpass = core.createRenderPass();

    std::shared_ptr<klartraum::DrawBasics> axes = std::make_shared<klartraum::DrawBasics>(klartraum::DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);

    auto& vulkanContext = core.getVulkanContext();

    auto& cameraUBO = renderpass->getCameraUBO();

    std::string spzFile = "data/hornedlizard.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting = vulkanContext.create<klartraum::VulkanGaussianSplatting>(renderpass, cameraUBO, spzFile);

    core.add(splatting);

    std::shared_ptr<klartraum::InterfaceCameraOrbit> cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit->setAzimuth(-31.0);
    cameraOrbit->setElevation(-0.5);
    core.setInterfaceCamera(cameraOrbit);
    core.setCameraUBO(cameraUBO);

    frontend.loop();

    return 0;
}
```
# Build
Currently, Klartraum can only be build on Windows with Visual Studio 2022 on the x64 architecture

```bash
git clone https://github.com/fortmeier/klartraum.git
git submodule update --init --recursive

mkdir build
cd build

cmake ..
cmake --build . --target klartraum

cd ..
# on Windows
.\build\Debug\klartraum.exe

# on Linux
# t.b.d.
```




TODO
====
- refactoring run:
-- rename computegraph to ComputeGraph
-- rename vulkancontext to vulkancontext

- get rid of warnings
- implement context factory method
