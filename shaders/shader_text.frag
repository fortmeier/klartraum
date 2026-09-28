#version 450

layout(binding = 0) uniform sampler2D glyphAtlas;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    // The atlas is a single-channel coverage mask: red holds glyph alpha.
    float coverage = texture(glyphAtlas, fragUV).r;
    outColor = vec4(fragColor.rgb, fragColor.a * coverage);
}
