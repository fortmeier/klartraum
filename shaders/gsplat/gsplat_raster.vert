#version 450
#extension GL_EXT_scalar_block_layout : enable

// set 0: camera UBO, as in the existing draw components
layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraWorldPos;
} ubo;

// set 1: per-splat SoA storage buffers + sorted-index buffer, in the order
// supplied to GaussianSplatRasterizer's constructor (binding == array index).
// Mirrors gsplat_projection.comp's Gaussian3D SoA inputs (band-major SH:
// shX[band * numSplats + idx]) plus the sorted compaction permutation.
layout(scalar, set = 1, binding = 0) readonly buffer Positions    { vec3  positions[];    };
layout(scalar, set = 1, binding = 1) readonly buffer Rotations    { vec4  rotations[];    };
layout(scalar, set = 1, binding = 2) readonly buffer Scales       { vec3  scales[];       };
layout(scalar, set = 1, binding = 3) readonly buffer ColorsAlpha  { vec4  colorsAlpha[];  };
layout(scalar, set = 1, binding = 4) readonly buffer ShR          { float shR[];          };
layout(scalar, set = 1, binding = 5) readonly buffer ShG          { float shG[];          };
layout(scalar, set = 1, binding = 6) readonly buffer ShB          { float shB[];          };
layout(scalar, set = 1, binding = 7) readonly buffer SortedIndices{ uint  sortedIndices[];};

layout(push_constant) uniform PushConstants {
    vec2  resolution;
    vec2  focal;
    float splatScale;
    int   shDegree;
    uint  numSplats;
} pc;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec3 outConic;
layout(location = 2) out vec2 outCenterPx;

// {(-1,-1), (1,-1), (-1,1), (1,1)} — a triangle-strip quad from gl_VertexIndex,
// drawn with no vertex buffer (guide §3.4). Axes below are the covariance's
// eigenvector directions, not screen x/y, so this quad is oriented per-splat.
vec2 quadCorner(uint vertexIndex) {
    const vec2 corners[4] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    return corners[vertexIndex];
}

mat3 quatToMat3(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    return mat3(
        1.0 - 2.0*(yy+zz),  2.0*(xy-wz),        2.0*(xz+wy),
        2.0*(xy+wz),        1.0 - 2.0*(xx+zz),  2.0*(yz-wx),
        2.0*(xz-wy),        2.0*(yz+wx),        1.0 - 2.0*(xx+yy)
    );
}

// EWA covariance projection (guide §5C / plan §2.1), identical to
// gsplat_projection.comp's validated implementation — the clamp bound is
// derived from 1/abs(focal) since Vulkan's Y-flip makes focalY negative,
// while the Jacobian J keeps the signed focal lengths for correct direction.
mat2 computeCovarianceMatrix2D(vec3 p, vec4 rotation, vec3 gscale) {
    mat3 S = mat3(gscale.x, 0.0, 0.0,
                  0.0, gscale.y, 0.0,
                  0.0, 0.0, gscale.z);
    mat3 R = quatToMat3(rotation);
    mat3 M = S * R;
    mat3 cov3d = transpose(M) * M;

    vec4 t4 = ubo.model * ubo.view * vec4(p, 1.0);
    vec3 t = t4.xyz;

    float focalX  = ubo.proj[0][0];
    float focalY  = ubo.proj[1][1];
    float tanFovX = 1.0 / abs(focalX);
    float tanFovY = 1.0 / abs(focalY);

    const float limx = 1.3 * tanFovX;
    const float limy = 1.3 * tanFovY;
    t.x = min(limx,  max(-limx,  t.x / t.z)) * t.z;
    t.y = min(limy,  max(-limy,  t.y / t.z)) * t.z;

    mat3 J = mat3(
        focalX / t.z, 0.0, -(focalX * t.x) / (t.z * t.z),
        0.0, focalY / t.z, -(focalY * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0);

    mat4 W4 = ubo.model * ubo.view;
    mat3 W = mat3(
        W4[0][0], W4[1][0], W4[2][0],
        W4[0][1], W4[1][1], W4[2][1],
        W4[0][2], W4[1][2], W4[2][2]);

    mat3 T = W * J;
    mat3 cov3dCam = transpose(T) * transpose(cov3d) * T;

    return mat2(cov3dCam[0][0], cov3dCam[0][1],
                cov3dCam[1][0], cov3dCam[1][1]);
}

// World-space SH evaluation, identical to gsplat_projection.comp's validated
// implementation (degree-3 basis evaluated unconditionally; coeffs[15] holds
// bands 1-3, the DC term is folded into `base`).
float computeSH(float base, float coeffs[15], vec3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x*x, y2 = y*y, z2 = z*z;
    float sh[16];
    sh[0]  = 0.282095;
    sh[1]  = -0.488603 * y;
    sh[2]  = -0.488603 * z;
    sh[3]  = -0.488603 * x;
    sh[4]  = 1.092548 * x * y;
    sh[5]  = -1.092548 * y * z;
    sh[6]  = 0.315392 * (2.0*z2 - x2 - y2);
    sh[7]  = -1.092548 * x * z;
    sh[8]  = 0.546274 * (x2 - y2);
    sh[9]  = -0.590044 * y * (3.0*x2 - y2);
    sh[10] = 2.890611 * x * y * z;
    sh[11] = -0.457046 * y * (4.0*z2 - x2 - y2);
    sh[12] = 0.373176 * z * (2.0*z2 - 3.0*x2 - 3.0*y2);
    sh[13] = -0.457046 * x * (4.0*z2 - x2 - y2);
    sh[14] = 1.445306 * z * (x2 - y2);
    sh[15] = -0.590044 * x * (x2 - 3.0*y2);
    float value = 0.5 + base * sh[0];
    for (int i = 0; i < 15; i++) value += coeffs[i] * sh[i + 1];
    return value;
}

void main() {
    uint splatId = sortedIndices[gl_InstanceIndex];
    vec3 position   = positions[splatId];
    vec4 rotation   = rotations[splatId];
    vec3 scale      = scales[splatId];
    vec4 colorAlpha = colorsAlpha[splatId];

    vec4 clip = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
    vec2 ndcCenter = clip.xy / clip.w;

    float swh = pc.resolution.x * 0.5;
    float shh = pc.resolution.y * 0.5;

    // Project the 3D covariance to screen-space pixels (same Jacobian scaling
    // and dilation term as gsplat_projection.comp, so both backends agree).
    mat2 cov = computeCovarianceMatrix2D(position, rotation, scale);
    float covA = cov[0][0] * swh * swh + 0.3;
    float covB = cov[0][1] * swh * shh;
    float covD = cov[1][1] * shh * shh + 0.3;

    // Closed-form eigendecomposition of the symmetric 2x2 covariance: its
    // eigenvectors are the ellipse's major/minor axes, sqrt(eigenvalue) the
    // axis sigma — this orients and sizes the quad to the splat's actual
    // projected footprint instead of an axis-aligned circle.
    float trace = covA + covD;
    float det   = covA * covD - covB * covB;
    float disc  = sqrt(max(trace * trace * 0.25 - det, 0.0));
    float lambda1 = trace * 0.5 + disc;
    float lambda2 = max(trace * 0.5 - disc, 0.0);

    vec2 majorDir = (abs(covB) > 1e-6)
        ? normalize(vec2(covB, lambda1 - covA))
        : ((covA >= covD) ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 minorDir = vec2(-majorDir.y, majorDir.x);

    // Quad half-extent = splatScale (sigma multiplier, ~3 => 99.7% coverage)
    // times the axis sigma, in pixels.
    float majorRadiusPx = pc.splatScale * sqrt(lambda1);
    float minorRadiusPx = pc.splatScale * sqrt(max(lambda2, 0.0));

    vec2 corner = quadCorner(gl_VertexIndex);
    vec2 offsetPx = majorDir * (corner.x * majorRadiusPx) + minorDir * (corner.y * minorRadiusPx);
    vec2 ndcOffset = offsetPx / vec2(swh, shh);

    gl_Position = vec4(ndcCenter + ndcOffset, clip.z / clip.w, 1.0);

    // Conic = inverse of the projected covariance, packed as the symmetric
    // matrix's distinct entries (a, b, c) = [[a, b], [b, c]] — the fragment
    // shader evaluates the exact per-pixel Gaussian weight from this, the same
    // "diff^T * covInv * diff" the compute backend's binned splatting performs.
    float invDet = 1.0 / max(det, 1e-12);
    outConic    = vec3(covD, -covB, covA) * invDet;
    outCenterPx = vec2((ndcCenter.x + 1.0) * swh, (ndcCenter.y + 1.0) * shh);

    // Spherical harmonics colour — coefficients are defined in world space, so
    // the view direction must be the world-space camera->Gaussian vector.
    float shRc[15], shGc[15], shBc[15];
    for (int i = 0; i < 15; i++) {
        shRc[i] = shR[i * pc.numSplats + splatId];
        shGc[i] = shG[i * pc.numSplats + splatId];
        shBc[i] = shB[i * pc.numSplats + splatId];
    }
    vec3 dir = normalize(position - ubo.cameraWorldPos.xyz);
    vec3 color = vec3(
        computeSH(colorAlpha.r, shRc, dir),
        computeSH(colorAlpha.g, shGc, dir),
        computeSH(colorAlpha.b, shBc, dir));

    // premultiplied: rgb already scaled by alpha so the hardware blender
    // composites back-to-front correctly (guide §3.4 "premultiplied over")
    float alpha = colorAlpha.a;
    outColor = vec4(color * alpha, alpha);
}
