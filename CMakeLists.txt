cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0135 NEW)

project(klartraum)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Tests and examples are only built by default when klartraum is the top-level
# project, not when it is consumed via add_subdirectory()/FetchContent.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(KLARTRAUM_IS_TOP_LEVEL ON)
else()
    set(KLARTRAUM_IS_TOP_LEVEL OFF)
endif()
option(KLARTRAUM_BUILD_TESTS "Build the klartraum unit tests" ${KLARTRAUM_IS_TOP_LEVEL})
option(KLARTRAUM_BUILD_EXAMPLES "Build the klartraum examples" ${KLARTRAUM_IS_TOP_LEVEL})

# macOS and Apple Silicon (M4) support
if(APPLE)
    message(STATUS "Building for macOS")
    
    # Detect Apple Silicon (ARM64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        message(STATUS "Detected Apple Silicon (ARM64) architecture")
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()

    # Set minimum macOS deployment target
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0")
    
    # Enable modern Objective-C runtime
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")
    
    # Optimization flags for Apple Silicon
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=apple-m1")
    endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(3rdparty/glfw)

# Link only the Vulkan loader. The loader selects the driver (ICD) at runtime,
# e.g. KosmicKrisp or MoltenVK on macOS; set VULKAN_SDK to choose the SDK.
find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

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
  include(FetchContent)
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


include_directories(${PROJECT_SOURCE_DIR}/3rdparty/spz/src/cc/)
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

file(GLOB_RECURSE GLSL_SOURCE_FILES "${SHADER_BASE_DIR}/*.vert" "${SHADER_BASE_DIR}/*.frag" "${SHADER_BASE_DIR}/*.comp")
# Exclude draft/scratch shaders prefixed with underscore
list(FILTER GLSL_SOURCE_FILES EXCLUDE REGEX "/_[^/]+\\.comp$")

foreach(GLSL ${GLSL_SOURCE_FILES})
  file(RELATIVE_PATH FILE_NAME ${SHADER_BASE_DIR} ${GLSL})

  set(SPIRV "${SHADER_BASE_DIR}/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSLC} -o ${SPIRV} ${GLSL}
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
  set(SPIRV "${SHADER_BASE_DIR}/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
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

# Add dependencies using FetchContent
include(FetchContent)

# Add GLM (OpenGL Mathematics) - header-only library
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# Dear ImGui (immediate-mode GUI) with its GLFW and Vulkan backends. It ships
# no CMake project, so the library target is defined here.
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.92.9b
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC ${Vulkan_LIBRARIES} glfw)

# Add Protobuf for ONNX support
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v25.1
  SOURCE_SUBDIR cmake
)
FetchContent_MakeAvailable(protobuf)

# Build protobuf after making it available

add_subdirectory(${protobuf_SOURCE_DIR} ${protobuf_BINARY_DIR})

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
  tests/test_image_resample.cpp

  tests/test_onnx_network.cpp
  tests/test_onnx_conv.cpp
  tests/test_onnx_conv_transpose.cpp
  tests/test_onnx_relu.cpp
  tests/test_onnx_transpose.cpp
  tests/test_onnx_reference.cpp

  tests/test_headless_frontend.cpp
  tests/test_multi_viewport_window.cpp
  tests/test_glfw_frontend.cpp
  tests/test_window_resize.cpp
  tests/test_glyph_atlas.cpp
  tests/test_simple_text_renderer.cpp
  tests/test_text_rendering.cpp
  tests/test_imgui_overlay.cpp
  tests/test_imgui_frontend.cpp
)

add_dependencies(klartraum_tests Shaders)
endif()


set(KLARTRAUM_LIB_SRC 
  src/glfw_frontend.cpp
  src/headless_frontend.cpp
  src/vulkan_gaussian_splatting.cpp
  src/vulkan_gaussian_splatting_raster.cpp
  src/gaussian_data_standard.cpp
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
  src/glyph_atlas.cpp
  src/simple_text_renderer.cpp
  src/text_draw_component.cpp
  src/imgui_overlay.cpp
  src/imgui_frontend.cpp
  ${SPZ_SRC}
  )

# Create Klartraum library
add_library(klartraum_lib STATIC ${KLARTRAUM_LIB_SRC})

set(LIBRARIES 
    ${Vulkan_LIBRARIES}
    glfw
    zlibstatic
    zstd::zstd
    glm::glm
    protobuf::libprotobuf
    onnx_proto
    imgui
)

if(APPLE)
  list(APPEND LIBRARIES ${MACOS_FRAMEWORKS})
endif()

target_link_libraries(klartraum_lib ${LIBRARIES})

# Consumers only need to link klartraum_lib to get its headers.
target_include_directories(klartraum_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/spz/src/cc
    ${Vulkan_INCLUDE_DIRS}
)

add_dependencies(klartraum_lib Shaders onnx_proto)

if(KLARTRAUM_BUILD_TESTS)
target_link_libraries(
  klartraum_tests
  gtest
  klartraum_lib
  )

include(GoogleTest)
gtest_discover_tests(klartraum_tests)
endif()
