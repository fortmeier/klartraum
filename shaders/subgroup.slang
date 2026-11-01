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
