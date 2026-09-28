#version 450

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec3 inConic;
layout(location = 2) in vec2 inCenterPx;

layout(location = 0) out vec4 outFragColor;

void main() {
    // Exact per-pixel Gaussian weight from the conic (inverse projected
    // covariance): exp(-0.5 * diff^T * conic * diff), the same evaluation
    // gsplat_binned_splatting.comp performs — conic is symmetric
    // (inConic = (a, b, c) for [[a,b],[b,c]]), so the cross term is 2*b*dx*dy.
    vec2  d = gl_FragCoord.xy - inCenterPx;
    float q = inConic.x * d.x * d.x + 2.0 * inConic.y * d.x * d.y + inConic.z * d.y * d.y;
    float falloff = exp(-0.5 * q);
    if (falloff < 1.0 / 255.0) {
        discard;
    }

    // inColor is premultiplied (rgb already scaled by alpha); multiplying by
    // the falloff keeps it premultiplied for the hardware blender.
    outFragColor = inColor * falloff;
}
