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

# The title examples set text with FreeType (KLARTRAUM_WITH_FREETYPE).
if(KLARTRAUM_WITH_FREETYPE)
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

  # tech demo 001: a sphere morphed into a pumpkin by differentiable rendering, dropped onto the bonsai
  add_executable(techdemo001_example
      techdemo001_example.cpp
  )

  target_link_libraries(techdemo001_example
      klartraum_lib
  )

  add_dependencies(techdemo001_example Shaders)
endif()

# Training a network with gradients from Slang's automatic differentiation
add_executable(autodiff_training_example
    autodiff_training_example.cpp
)

target_link_libraries(autodiff_training_example
    klartraum_lib
)

add_dependencies(autodiff_training_example Shaders)

# Fitting a mesh and its texture to masked images with differentiable rasterization
add_executable(mesh_texture_fitting_example
    mesh_texture_fitting_example.cpp
)

target_link_libraries(mesh_texture_fitting_example
    klartraum_lib
)

add_dependencies(mesh_texture_fitting_example Shaders)

# Training 3D Gaussian splats on masked photographs, one submit per step
add_executable(splat_training_example
    splat_training_example.cpp
)

target_link_libraries(splat_training_example
    klartraum_lib
)

add_dependencies(splat_training_example Shaders)

# Bonsai forest of GPU-generated trees, rendered classically and as Gaussians
add_executable(tree_bonsai_example
    tree_bonsai_example.cpp
)

target_link_libraries(tree_bonsai_example
    klartraum_lib
)

add_dependencies(tree_bonsai_example Shaders)

# Walking or flying through the bonsai landscape in a window
add_executable(tree_bonsai_explore_example
    tree_bonsai_explore_example.cpp
)

target_link_libraries(tree_bonsai_explore_example
    klartraum_lib
)

add_dependencies(tree_bonsai_explore_example Shaders)

# Popping of a single Gaussian tree's levels of detail, measured along a flight away from it and back
add_executable(tree_lod_metrics_example
    tree_lod_metrics_example.cpp
)

target_link_libraries(tree_lod_metrics_example
    klartraum_lib
)

add_dependencies(tree_lod_metrics_example Shaders)

# A database of view pairs (Gaussian renderer, classical reference) for training a neural enhancer of distant trees
add_executable(tree_neural_dataset_example
    tree_neural_dataset_example.cpp
)

target_link_libraries(tree_neural_dataset_example
    klartraum_lib
)

add_dependencies(tree_neural_dataset_example Shaders)
