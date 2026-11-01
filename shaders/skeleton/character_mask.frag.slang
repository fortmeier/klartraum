// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The pixels within a tapered capsule around the segment: closer to the nearest
// point of the bone than the radius there, interpolated between the ends.
import character_mask_common;
import occlusion;

[shader("fragment")]
float4 main(CharacterMaskVertex input) : SV_Target0 {
    float2 fragCoord = input.position.xy;
    float2 first = input.bone.xy, along = input.bone.zw - input.bone.xy;
    float lengthSquared = dot(along, along);
    float t = lengthSquared > 0.0 ? clamp(dot(fragCoord - first, along) / lengthSquared, 0.0, 1.0) : 0.0;
    if (length(fragCoord - (first + t * along)) > lerp(input.radii.x, input.radii.y, t)) {
        discard;
    }
    if (pc.useOcclusion != 0u &&
        occluded(fragCoord, input.bone, input.boneDepth, input.projection, pc.occlusionBias, pc.resolution)) {
        discard;
    }
    return float4(1.0);
}
