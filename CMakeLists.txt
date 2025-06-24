cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0135 NEW)

project(klartraum)
set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(3rdparty/glfw)

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

set(ZLIB_BUILD_EXAMPLES OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/zlib)


include_directories(${PROJECT_SOURCE_DIR}/3rdparty/spz/src/cc/)
set(SPZ_SRC
  3rdparty/spz/src/cc/load-spz.cc
  3rdparty/spz/src/cc/splat-c-types.cc
  3rdparty/spz/src/cc/splat-types.cc
)

set(SRC 
  src/glfw_frontend.cpp
  src/vulkan_gaussian_splatting.cpp
  src/vulkan_helpers.cpp
  src/vulkan_kernel.cpp
  src/klartraum_core.cpp
  src/interface_camera_orbit.cpp
  src/draw_basics.cpp
  ${SPZ_SRC}
  )

add_executable(klartraum src/main.cpp ${SRC})

target_link_libraries(klartraum glfw zlibstatic ${Vulkan_LIBRARIES} ${Python3_LIBRARIES})



find_program(GLSLC glslc REQUIRED)

file(GLOB_RECURSE GLSL_SOURCE_FILES "${PROJECT_SOURCE_DIR}/shaders/*.vert" "${PROJECT_SOURCE_DIR}/shaders/*.frag" "${PROJECT_SOURCE_DIR}/shaders/*.comp")

foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSLC} -o ${SPIRV} ${GLSL}
    DEPENDS ${GLSL}
    COMMENT "Compiling ${GLSL} to ${SPIRV}"
  )
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

add_custom_target(
  Shaders
  DEPENDS ${SPIRV_BINARY_FILES} ${GLSL_SOURCE_FILES}
)

add_dependencies(klartraum Shaders)


# Add Google Test
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  klartraum_tests
  tests/test_main.cpp
  tests/test_vulkan_buffers.cpp
  tests/test_computegraph.cpp
  tests/test_buffertransformation.cpp
  tests/test_gaussian_splatting.cpp
  ${SRC}
)

add_dependencies(klartraum_tests Shaders)

target_link_libraries(
  klartraum_tests
  gtest_main
  glfw
  zlibstatic
  ${Vulkan_LIBRARIES}
  )

include(GoogleTest)
gtest_discover_tests(klartraum_tests)