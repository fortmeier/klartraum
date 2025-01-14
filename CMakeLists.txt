cmake_minimum_required(VERSION 3.10)

project(klartraum)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(3rdparty/glfw)

set(SRC src/main.cpp src/backend_vulkan.cpp)

add_executable(klartraum ${SRC})

target_link_libraries(klartraum glfw)