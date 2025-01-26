cmake_minimum_required(VERSION 3.10)

project(klartraum)
set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(3rdparty/glfw)

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

set(Python3_ROOT_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cpython/)
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_EXECUTABLE ${PROJECT_SOURCE_DIR}/3rdparty/cpython/PCbuild/amd64/python_d.exe)
set(Python3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cpython/Include)
set(Python3_LIBRARY ${PROJECT_SOURCE_DIR}/3rdparty/cpython/PCbuild/amd64/python312_d.lib)
find_package(Python3 3.12.8 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

set(ZLIB_BUILD_EXAMPLES OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/zlib)


include_directories(${PROJECT_SOURCE_DIR}/3rdparty/spz/src/cc/)
set(SPZ_SRC
  3rdparty/spz/src/cc/load-spz.cc
  3rdparty/spz/src/cc/splat-c-types.cc
  3rdparty/spz/src/cc/splat-types.cc
)

set(SRC 
  src/main.cpp
  src/backend_vulkan.cpp
  src/vulkan_gaussian_splatting.cpp
  src/vulkan_helpers.cpp
  src/camera.cpp
  ${SPZ_SRC}
  )

add_executable(klartraum ${SRC})

target_link_libraries(klartraum glfw zlibstatic ${Vulkan_LIBRARIES} ${Python3_LIBRARIES})