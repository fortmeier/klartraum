#version 450

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inQuadUV;

layout(location = 0) out vec4 outFragColor;

void main() {
    // Gaussian falloff over the quad, parameterized on the unit disk; the
    // quad is sized to ~3sigma so corners (|uv| == sqrt(2)) fall well below
    // the discard threshold (guide §3.4 / §5C).
    float d2 = dot(inQuadUV, inQuadUV);
    float falloff = exp(-0.5 * d2 * 9.0);
    if (falloff < 1.0 / 255.0) {
        discard;
    }

    // inColor is premultiplied (rgb already scaled by alpha); multiplying by
    // the falloff keeps it premultiplied for the hardware blender.
    outFragColor = inColor * falloff;
}
