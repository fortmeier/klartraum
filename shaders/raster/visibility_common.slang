// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Interface between visibility.vert.slang and visibility.frag.slang.
module visibility_common;

// The vertex outputs of the visibility pass.
public struct VisibilityVertex {
    public float4 position : SV_Position;
    // The index of the triangle the vertex belongs to.
    [[vk::location(0)]] public nointerpolation uint triangle;
};
