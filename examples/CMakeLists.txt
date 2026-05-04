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

# ONNX Example
add_executable(onnx_example
    onnx_example.cpp
)

target_link_libraries(onnx_example 
    klartraum_lib
)