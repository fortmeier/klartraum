// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Push constants shared by the Transpose shaders.
module transpose_common;

public struct TransposePushConstants {
    public uint dims;
    public uint dimInput[8];
    public uint dimOutput[8];
    public uint dimPerm[8];
};
