cmake_minimum_required(VERSION 3.10)

project(klartraum)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(3rdparty/glfw)

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

set(Python3_ROOT_DIR ${PROJECT_SOURCE_DIR}/3rdparty/python-3.13.1-embed-amd64)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED )
include_directories(${Python3_INCLUDE_DIRS})

set(SRC src/main.cpp src/backend_vulkan.cpp)

add_executable(klartraum ${SRC})

target_link_libraries(klartraum glfw ${Vulkan_LIBRARIES} ${Python3_LIBRARIES})