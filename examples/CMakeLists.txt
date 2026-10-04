# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# DrawBasics Example
add_executable(draw_basics_example
    draw_basics_example.cpp
)

target_link_libraries(draw_basics_example
    klartraum_lib
)

add_dependencies(draw_basics_example Shaders)

# Gaussian Splatting Example
add_executable(gaussian_splatting_example
    gaussian_splatting_example.cpp
)

target_link_libraries(gaussian_splatting_example
    klartraum_lib
)

add_dependencies(gaussian_splatting_example Shaders)

# Gaussian splatting through the ONNX autoencoder
add_executable(gaussian_autoencoder_example
    gaussian_autoencoder_example.cpp
)

target_link_libraries(gaussian_autoencoder_example
    klartraum_lib
)

add_dependencies(gaussian_autoencoder_example Shaders)

# Multi-viewport (single camera) Example
add_executable(multi_viewports_single_camera_example
    multi_viewports_single_camera_example.cpp
)

target_link_libraries(multi_viewports_single_camera_example
    klartraum_lib
)

add_dependencies(multi_viewports_single_camera_example Shaders)

# ONNX Example
add_executable(onnx_example
    onnx_example.cpp
)

target_link_libraries(onnx_example
    klartraum_lib
)

# Headless turntable: renders a scene from a camera orbiting it once
add_executable(turntable_example
    turntable_example.cpp
)

target_link_libraries(turntable_example
    klartraum_lib
)

add_dependencies(turntable_example Shaders)

# Stable Diffusion 1.5 VAE encoder/decoder example
add_executable(sd15_vae_example
    sd15_vae_example.cpp
)

target_link_libraries(sd15_vae_example
    klartraum_lib
)

add_dependencies(sd15_vae_example Shaders)

# Stable Diffusion 1.5 fixed-size UNet denoiser example
add_executable(sd15_denoiser_example
    sd15_denoiser_example.cpp
)

target_link_libraries(sd15_denoiser_example
    klartraum_lib
)

add_dependencies(sd15_denoiser_example Shaders)

# NVIDIA Cosmos3-Edge image-to-video example (graphs from scripts/cosmos3)
add_executable(cosmos3_example
    cosmos3_example.cpp
)

target_link_libraries(cosmos3_example
    klartraum_lib
)

add_dependencies(cosmos3_example Shaders)
