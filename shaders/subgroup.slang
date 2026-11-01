// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Subgroup built-ins that Slang's wave intrinsics do not expose.
module subgroup;

// Returns the index of the invocation's subgroup within its workgroup
// (SPIR-V SubgroupId, GLSL gl_SubgroupID).
public uint subgroupId() {
    return spirv_asm {
        result:$$uint = OpLoad builtin(SubgroupId:uint);
    };
}

// Returns the number of subgroups in the workgroup (SPIR-V NumSubgroups).
public uint subgroupCount() {
    return spirv_asm {
        result:$$uint = OpLoad builtin(NumSubgroups:uint);
    };
}

// Returns `value` of the lane whose index is this lane's index XOR `mask`.
public float subgroupShuffleXor(float value, uint mask) {
    return WaveReadLaneAt(value, WaveGetLaneIndex() ^ mask);
}
