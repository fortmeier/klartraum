# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.24)

# The default minimum macOS version; set CMAKE_OSX_DEPLOYMENT_TARGET on the
# command line to change it. It has to be known before project().
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")
endif()

project(klartraum VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

# Tests and examples are only built by default when klartraum is the top-level
# project, not when it is consumed via add_subdirectory()/FetchContent.
option(KLARTRAUM_BUILD_TESTS "Build the klartraum unit tests" ${PROJECT_IS_TOP_LEVEL})
option(KLARTRAUM_BUILD_EXAMPLES "Build the klartraum examples" ${PROJECT_IS_TOP_LEVEL})

add_subdirectory(3rdparty/glfw)

# Link only the Vulkan loader. The loader selects the driver (ICD) at runtime,
# e.g. KosmicKrisp or MoltenVK on macOS; set VULKAN_SDK to choose the SDK.
find_package(Vulkan REQUIRED)

if(APPLE)
    # macOS system frameworks required for graphics applications
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
    find_library(IOKIT_LIBRARY IOKit REQUIRED)
    find_library(COREVIDEO_LIBRARY CoreVideo REQUIRED)
    find_library(QUARTZCORE_LIBRARY QuartzCore REQUIRED)
    find_library(METAL_LIBRARY Metal REQUIRED)

    set(MACOS_FRAMEWORKS
        ${COCOA_LIBRARY}
        ${IOKIT_LIBRARY}
        ${COREVIDEO_LIBRARY}
        ${QUARTZCORE_LIBRARY}
        ${METAL_LIBRARY}
    )
endif()

set(ZLIB_BUILD_EXAMPLES OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/zlib)

find_package(zstd QUIET)
if(NOT zstd_FOUND)
  message(STATUS "zstd not found, using FetchContent to download it.")
  FetchContent_Declare(
    zstd
    URL https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
    URL_HASH SHA256=8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1
    SOURCE_SUBDIR build/cmake
  )
  set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
  set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(zstd)
  add_library(zstd::zstd ALIAS libzstd_static)
elseif(TARGET zstd::libzstd_static)
  add_library(zstd::zstd ALIAS zstd::libzstd_static)
elseif(NOT TARGET zstd::zstd)
  add_library(zstd::zstd ALIAS zstd::libzstd_shared)
endif()


set(SPZ_SRC
  3rdparty/spz/src/cc/load-spz.cc
  3rdparty/spz/src/cc/splat-c-types.cc
  3rdparty/spz/src/cc/splat-types.cc
)

# Add examples
if(KLARTRAUM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()



# Prefer the glslc that belongs to the Vulkan SDK found above.
if(Vulkan_GLSLC_EXECUTABLE)
    set(GLSLC ${Vulkan_GLSLC_EXECUTABLE})
else()
    find_program(GLSLC glslc REQUIRED)
endif()
message(STATUS "Using glslc: ${GLSLC}")

set(SHADER_BASE_DIR "${PROJECT_SOURCE_DIR}/shaders")
# The compiled shaders go to the build directory; the library finds them there
# through its default asset root (see readFile() in vulkan_helpers.hpp).
set(SHADER_OUTPUT_DIR "${PROJECT_BINARY_DIR}/shaders")

# Shaders compiled into the source tree by older builds would be found first
# when running from the repository root.
file(GLOB_RECURSE STALE_SPIRV_FILES "${SHADER_BASE_DIR}/*.spv")
if(STALE_SPIRV_FILES)
  message(WARNING "shaders/ contains compiled .spv files from an older build. They take "
                  "precedence over the shaders in ${SHADER_OUTPUT_DIR} when running from the "
                  "repository root; delete them with: find shaders -name '*.spv' -delete")
endif()

file(GLOB_RECURSE GLSL_SOURCE_FILES "${SHADER_BASE_DIR}/*.vert" "${SHADER_BASE_DIR}/*.frag" "${SHADER_BASE_DIR}/*.comp")
# Exclude draft/scratch shaders prefixed with underscore
list(FILTER GLSL_SOURCE_FILES EXCLUDE REGEX "/_[^/]+\\.comp$")

foreach(GLSL ${GLSL_SOURCE_FILES})
  file(RELATIVE_PATH FILE_NAME ${SHADER_BASE_DIR} ${GLSL})

  set(SPIRV "${SHADER_OUTPUT_DIR}/${FILE_NAME}.spv")
  get_filename_component(SPIRV_DIRECTORY ${SPIRV} DIRECTORY)
  set(GLSLC_OPTIONS)
  if(FILE_NAME STREQUAL "onnx/fused_attention.comp")
    # Subgroup arithmetic is core in Vulkan 1.1 / SPIR-V 1.3.
    list(APPEND GLSLC_OPTIONS --target-env=vulkan1.1)
  endif()
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SPIRV_DIRECTORY}
    COMMAND ${GLSLC} ${GLSLC_OPTIONS} -o ${SPIRV} ${GLSL}
    DEPENDS ${GLSL}
    COMMENT "Compiling ${GLSL} to ${SPIRV}"
  )
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

# Mesh shaders (VK_EXT_mesh_shader) need a SPIR-V 1.4+ target, so compile them
# separately with --target-env=vulkan1.3 (glslc infers the stage from .mesh).
file(GLOB_RECURSE MESH_SOURCE_FILES "${SHADER_BASE_DIR}/*.mesh")
foreach(GLSL ${MESH_SOURCE_FILES})
  file(RELATIVE_PATH FILE_NAME ${SHADER_BASE_DIR} ${GLSL})
  set(SPIRV "${SHADER_OUTPUT_DIR}/${FILE_NAME}.spv")
  get_filename_component(SPIRV_DIRECTORY ${SPIRV} DIRECTORY)
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SPIRV_DIRECTORY}
    COMMAND ${GLSLC} --target-env=vulkan1.3 -o ${SPIRV} ${GLSL}
    DEPENDS ${GLSL}
    COMMENT "Compiling mesh ${GLSL} to ${SPIRV}"
  )
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

add_custom_target(
  Shaders
  DEPENDS ${SPIRV_BINARY_FILES} ${GLSL_SOURCE_FILES}
)

# Add GLM (OpenGL Mathematics) - header-only library
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 0af55ccecd98d4e5a8d1fad7de25ba429d60e863 # 1.0.1
)
FetchContent_MakeAvailable(glm)

# Add Protobuf for ONNX support
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG 7f94235e552599141950d7a4a3eaf93bc87d1b22 # v25.1
)
FetchContent_MakeAvailable(protobuf)

# Abseil (bundled with protobuf) passes its Randen HWAES flags on Apple as
# "-Xarch_x86_64 -maes -Xarch_x86_64 -msse4.1 ...". CMake de-duplicates the
# repeated -Xarch_* tokens, leaving x86-only flags unguarded on arm64. Re-add
# each flag as an indivisible SHELL: group so it stays bound to its arch.
if(APPLE)
    foreach(_absl_alias absl::random_internal_randen_hwaes absl::random_internal_randen_hwaes_impl)
        if(TARGET ${_absl_alias})
            get_target_property(_absl_target ${_absl_alias} ALIASED_TARGET)
            get_target_property(_absl_opts ${_absl_target} COMPILE_OPTIONS)
            list(REMOVE_ITEM _absl_opts -Xarch_x86_64 -Xarch_arm64 -maes -msse4.1 -march=armv8-a+crypto)
            list(APPEND _absl_opts
                "SHELL:-Xarch_x86_64 -maes"
                "SHELL:-Xarch_x86_64 -msse4.1"
                "SHELL:-Xarch_arm64 -march=armv8-a+crypto")
            set_target_properties(${_absl_target} PROPERTIES COMPILE_OPTIONS "${_absl_opts}")
        endif()
    endforeach()
endif()

# protoc is built as part of protobuf above; use that binary so the generated
# code always matches the linked libprotobuf version.
set(protobuf_PROTOC_EXE $<TARGET_FILE:protobuf::protoc>)

# ONNX Protobuf Generation
set(ONNX_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnx/onnx")
set(ONNX_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Create generated directory
file(MAKE_DIRECTORY ${ONNX_GENERATED_DIR})

# Copy ONNX proto files to project (you'll need to copy them manually first)
# Expected files: onnx.proto, onnx-data.proto, onnx-operators.proto
set(ONNX_PROTO_FILES
    "${ONNX_PROTO_DIR}/onnx.proto"
)

# Generate C++ files from proto files
set(ONNX_GENERATED_SOURCES)
foreach(proto_file ${ONNX_PROTO_FILES})
    get_filename_component(proto_name ${proto_file} NAME_WE)
    set(generated_cpp "${ONNX_GENERATED_DIR}/${proto_name}.pb.cc")
    set(generated_h "${ONNX_GENERATED_DIR}/${proto_name}.pb.h")

    add_custom_command(
        OUTPUT ${generated_cpp} ${generated_h}
        COMMAND ${protobuf_PROTOC_EXE} --cpp_out=${ONNX_GENERATED_DIR} --proto_path=${ONNX_PROTO_DIR} ${proto_file}
        DEPENDS ${proto_file} protobuf::protoc
        COMMENT "Generating C++ code from ${proto_file}"
    )

    list(APPEND ONNX_GENERATED_SOURCES ${generated_cpp} ${generated_h})
endforeach()

# Create ONNX library with generated sources
add_library(onnx_proto STATIC ${ONNX_GENERATED_SOURCES})
target_link_libraries(onnx_proto protobuf::libprotobuf)
target_include_directories(onnx_proto PUBLIC ${ONNX_GENERATED_DIR})



if(KLARTRAUM_BUILD_TESTS)
# Add Google Test
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
  URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  klartraum_tests
  tests/test_main.cpp
  tests/test_vulkan_buffers.cpp
  tests/test_vulkan_context.cpp
  tests/test_vulkan_helpers.cpp
  tests/test_computegraph.cpp
  tests/test_buffertransformation.cpp
  tests/test_drawindirectbuffer.cpp
  tests/test_buffertographicsbarrier.cpp
  tests/test_gaussian_splat_rasterizer.cpp
  tests/test_gaussian_dist.cpp
  tests/test_gaussian_splatting_raster.cpp
  tests/test_gaussian_splatting_factory.cpp
  tests/test_general_computation.cpp
  tests/test_gaussian_splatting.cpp
  tests/test_copybuffer.cpp
  tests/test_tensorelement.cpp
  tests/test_tensor_memory_planner.cpp
  tests/test_clip_tokenizer.cpp
  tests/test_unipc_flow_scheduler.cpp
  tests/test_cosmos3_conditioning_image.cpp
  tests/test_image_resample.cpp
  tests/test_gaussian_transform.cpp
  tests/test_gaussian_elements.cpp

  tests/test_onnx_network.cpp
  tests/test_onnx_external_data.cpp
  tests/test_onnx_conv.cpp
  tests/test_onnx_conv_transpose.cpp
  tests/test_onnx_relu.cpp
  tests/test_onnx_transpose.cpp
  tests/test_onnx_reference.cpp
  tests/test_onnx_sd15_operations.cpp
  tests/test_onnx_cosmos3_operations.cpp
  tests/test_layers.cpp
  tests/test_render_over_targets.cpp

  tests/test_headless_frontend.cpp
  tests/test_multi_viewport_window.cpp
  tests/test_glfw_frontend.cpp
  tests/test_window_resize.cpp
  tests/test_glyph_atlas.cpp
  tests/test_simple_text_renderer.cpp
  tests/test_text_rendering.cpp
)

add_dependencies(klartraum_tests Shaders)
endif()


set(KLARTRAUM_LIB_SRC
  src/glfw_frontend.cpp
  src/headless_frontend.cpp
  src/vulkan_gaussian_splatting.cpp
  src/vulkan_gaussian_splatting_raster.cpp
  src/gaussian_data_standard.cpp
  src/gaussian_transform.cpp
  src/gaussian_splatting_factory.cpp
  src/vulkan_helpers.cpp
  src/vulkan_context.cpp
  src/window.cpp
  src/klartraum_engine.cpp
  src/interface_camera_orbit.cpp
  src/draw_basics.cpp
  src/gaussian_splat_rasterizer.cpp
  src/gaussian_splat_mesh_rasterizer.cpp
  src/onnx_network.cpp
  src/layers.cpp
  src/clip_tokenizer.cpp
  src/unipc_flow_scheduler.cpp
  src/cosmos3_conditioning_image.cpp
  src/glyph_atlas.cpp
  src/simple_text_renderer.cpp
  src/text_draw_component.cpp
  ${SPZ_SRC}
  )

# Create Klartraum library
add_library(klartraum_lib STATIC ${KLARTRAUM_LIB_SRC})

set(LIBRARIES
    Vulkan::Vulkan
    glfw
    zlibstatic
    zstd::zstd
    glm::glm
    protobuf::libprotobuf
    onnx_proto
)

if(APPLE)
  list(APPEND LIBRARIES ${MACOS_FRAMEWORKS})
endif()

target_link_libraries(klartraum_lib PUBLIC ${LIBRARIES})
target_compile_features(klartraum_lib PUBLIC cxx_std_17)

# Consumers only need to link klartraum_lib to get its headers.
target_include_directories(klartraum_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/spz/src/cc
)

target_compile_definitions(klartraum_lib PRIVATE KLARTRAUM_DEFAULT_ASSET_DIR="${PROJECT_BINARY_DIR}")

add_dependencies(klartraum_lib Shaders onnx_proto)

if(KLARTRAUM_BUILD_TESTS)
target_link_libraries(
  klartraum_tests
  gtest
  klartraum_lib
  )

include(GoogleTest)
# The tests read data/ and write build/TestingOutput/ relative to the
# repository root.
gtest_discover_tests(klartraum_tests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
