// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import occlusion;
import openpose_common;

[shader("fragment")]
float4 main(OpenPoseVertex input) : SV_Target0 {
    float2 offset = input.position.xy - input.center;
    if (pc.drawJoints != 0u) {
        // OpenCV's filled circle: the pixels whose offset lies within the radius.
        if (dot(offset, offset) > input.axes.x * input.axes.x + 1e-3) {
            discard;
        }
    } else {
        float2 local = float2(dot(offset, input.direction), dot(offset, float2(-input.direction.y, input.direction.x))) /
                       input.axes;
        if (dot(local, local) > 1.0) {
            discard;
        }
    }
    if (pc.useOcclusion != 0u &&
        occluded(input.position.xy, input.bone, input.boneDepth, input.projection, pc.occlusionBias, pc.resolution)) {
        discard;
    }
    return float4(input.color, 1.0);
}
