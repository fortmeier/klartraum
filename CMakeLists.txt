cmake_minimum_required(VERSION 3.10)

project(klartraum)
set(CMAKE_GENERATOR_PLATFORM x64)

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

set(SRC 
  src/main.cpp
  src/backend_vulkan.cpp
  src/vulkan_gaussian_splatting.cpp
  )

add_executable(klartraum ${SRC})

target_link_libraries(klartraum glfw ${Vulkan_LIBRARIES} ${Python3_LIBRARIES})