#version 450

// Screen size in pixels, used to map pixel-space positions to NDC.
layout(push_constant) uniform PushConstants {
    vec2 screenSize;
} pushConstants;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;

void main() {
    // Pixel-space (origin top-left, y down) maps directly to Vulkan NDC
    // (origin top-left, y down) by normalizing to [0, 2] and shifting by -1.
    vec2 ndc = (inPosition / pushConstants.screenSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    fragUV = inUV;
    fragColor = inColor;
}
