// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Storage buffers with scalar block layout (VK_EXT_scalar_block_layout): a
// float3 element takes 12 bytes, as in the tightly packed C++ arrays.
module scalar_buffer;

public typealias ScalarBuffer<T> = StructuredBuffer<T, ScalarDataLayout>;
public typealias RWScalarBuffer<T> = RWStructuredBuffer<T, ScalarDataLayout>;
