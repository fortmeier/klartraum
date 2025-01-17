# klartraum
Klartraum is a real-time neural rendering engine build on top of vulkan with Python integration.


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


