# Gaussian Splatting Example
add_executable(gaussian_splatting_example 
    gaussian_splatting_example.cpp
)

target_link_libraries(gaussian_splatting_example 
    klartraum_lib
)

add_dependencies(gaussian_splatting_example Shaders)