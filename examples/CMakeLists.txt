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

# Headless OpenPose pose maps of a 3D skeleton seen from an orbiting camera
add_executable(openpose_skeleton_example
    openpose_skeleton_example.cpp
)

target_link_libraries(openpose_skeleton_example
    klartraum_lib
)

add_dependencies(openpose_skeleton_example Shaders)

# Neural character animation rendered as a video with ControlNet-OpenPose and SD1.5 LCM
add_executable(pose_to_video_example
    pose_to_video_example.cpp
)

target_link_libraries(pose_to_video_example
    klartraum_lib
)

add_dependencies(pose_to_video_example Shaders)

# A neural animated character walking around the lantern, drawn as an OpenPose skeleton over the scene
add_executable(lantern_walk_example
    lantern_walk_example.cpp
)

target_link_libraries(lantern_walk_example
    klartraum_lib
)

add_dependencies(lantern_walk_example Shaders)

# The promo title as Gaussians, rendered once per candidate font
add_executable(title_font_example
    title_font_example.cpp
)

target_link_libraries(title_font_example
    klartraum_lib
)

add_dependencies(title_font_example Shaders)

# The opening shot of the promo video: the title morphs into the rotating lantern
add_executable(title_morph_example
    title_morph_example.cpp
)

target_link_libraries(title_morph_example
    klartraum_lib
)

add_dependencies(title_morph_example Shaders)

# klartraum tech demo 000: the promo video in one continuous shot around the lantern
add_executable(techdemo000_example
    techdemo000_example.cpp
)

target_link_libraries(techdemo000_example
    klartraum_lib
)

add_dependencies(techdemo000_example Shaders)
