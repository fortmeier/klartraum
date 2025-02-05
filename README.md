# klartraum
Klartraum is a real-time neural rendering engine build on top of vulkan with Python integration.

# example
```cpp
#include <iostream>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"

int main() {
    std::cout << "Wake up, dreamer!" << std::endl;

    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& kernel = core.getVulkanKernel();

    std::shared_ptr<klartraum::DrawBasics> axes =
        std::make_shared<klartraum::DrawBasics>(klartraum::DrawBasicsType::Axes);

    core.addDrawComponent(axes);

    std::string spzFile = "data/hornedlizard.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting =
        std::make_shared<klartraum::VulkanGaussianSplatting>(spzFile);

    core.addDrawComponent(splatting);

    std::shared_ptr<klartraum::InterfaceCamera> cameraOrbit =
        std::make_shared<klartraum::InterfaceCameraOrbit>();

    core.setInterfaceCamera(cameraOrbit);

    frontend.loop();

    return 0;
}
```
# Build
## Prerequisites
Building needs a development build of python.

### Build Python for Windows Development

To be able to build with Visual Studio 17 2022, python needs to be build from scratch.

Follow the instructions provided in the [Python Developer's Guide](https://devguide.python.org/getting-started/setup-building/). This boils down to:

1. Clone the Python repository:
    ```sh
    cd 3rdparty
    git clone https://github.com/python/cpython.git
    git checkout v3.12.8
    ```

2. Navigate to the `PCbuild` directory and run the build script:
    ```sh
    PCbuild\build.bat -c Debug
    ```

3. Open the solution file (`.sln`) in Visual Studio and compile the project.


