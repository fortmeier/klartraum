cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0135 NEW)

project(klartraum)
set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

# macOS Vulkan support via MoltenVK
if(APPLE)
    # Try to find MoltenVK
    find_library(MOLTENVK_LIBRARY MoltenVK PATHS
        /usr/local/lib
        /opt/homebrew/lib
        $ENV{VULKAN_SDK}/lib
        $ENV{VULKAN_SDK}/MoltenVK/dylib/macOS
    )
    
    if(MOLTENVK_LIBRARY)
        message(STATUS "Found MoltenVK: ${MOLTENVK_LIBRARY}")
        list(APPEND Vulkan_LIBRARIES ${MOLTENVK_LIBRARY})
    else()
        message(WARNING "MoltenVK not found. Please install Vulkan SDK for macOS.")
    endif()
    
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


include_directories(${PROJECT_SOURCE_DIR}/3rdparty/spz/src/cc/)
set(SPZ_SRC
  3rdparty/spz/src/cc/load-spz.cc
  3rdparty/spz/src/cc/splat-c-types.cc
  3rdparty/spz/src/cc/splat-types.cc
)

# Add examples
add_subdirectory(examples)



find_program(GLSLC glslc REQUIRED)

set(SHADER_BASE_DIR "${PROJECT_SOURCE_DIR}/shaders")

file(GLOB_RECURSE GLSL_SOURCE_FILES "${SHADER_BASE_DIR}/*.vert" "${SHADER_BASE_DIR}/*.frag" "${SHADER_BASE_DIR}/*.comp")

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

# find protoc
find_program(protobuf_PROTOC_EXE 
    NAMES protoc protoc.exe
    PATHS build/_deps/protobuf-build/Debug
)

# ONNX Protobuf Generation
set(ONNX_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
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
        DEPENDS ${proto_file}
        COMMENT "Generating C++ code from ${proto_file}"
    )
    
    list(APPEND ONNX_GENERATED_SOURCES ${generated_cpp} ${generated_h})
endforeach()

# Create ONNX library with generated sources
add_library(onnx_proto STATIC ${ONNX_GENERATED_SOURCES})
target_link_libraries(onnx_proto protobuf::libprotobuf)
target_include_directories(onnx_proto PUBLIC ${ONNX_GENERATED_DIR})



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
  tests/test_computegraph.cpp
  tests/test_buffertransformation.cpp
  tests/test_gaussian_splatting.cpp
)

add_dependencies(klartraum_tests Shaders)


set(KLARTRAUM_LIB_SRC 
  src/glfw_frontend.cpp
  src/vulkan_gaussian_splatting.cpp
  src/vulkan_helpers.cpp
  src/vulkan_context.cpp
  src/klartraum_engine.cpp
  src/interface_camera_orbit.cpp
  src/draw_basics.cpp
  ${SPZ_SRC}
  )

# Create Klartraum library
add_library(klartraum_lib STATIC ${KLARTRAUM_LIB_SRC})

set(LIBRARIES 
    ${Vulkan_LIBRARIES}
    glfw
    zlibstatic
    glm::glm
    protobuf::libprotobuf
    onnx_proto
)

if(APPLE)
  list(APPEND LIBRARIES ${MACOS_FRAMEWORKS})
endif()

target_link_libraries(klartraum_lib ${LIBRARIES})

add_dependencies(klartraum_lib Shaders onnx_proto)

target_link_libraries(
  klartraum_tests
  gtest_main
  klartraum_lib
  )

include(GoogleTest)
gtest_discover_tests(klartraum_tests)