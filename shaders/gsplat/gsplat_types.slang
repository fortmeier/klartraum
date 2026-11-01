// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Types shared by the Gaussian splatting shaders.
module gsplat_types;

public struct Gaussian {
    public float3 position;
    public float4 rotation;
    public float3 scale;
    public float3 color;
    public float alpha;
    public float shR[15];
    public float shG[15];
    public float shB[15];
};

public struct Gaussian2D {
    public float2 position;        // position in screen space
    public float z;                // depth value
    public uint binMask;           // bitmask for which bins this gaussian is in, only one bit should be set
    public float2x2 covarianceInv; // inverse covariance matrix for the gaussian
    public float3 color;           // color of the gaussian
    public float alpha;
};

public struct RadixContainer {
    public uint value;
    public uint originalIndex;
};

public struct StartAndEnd {
    public uint start;
    public uint end;
};

public struct DispatchIndirectCommand {
    public uint x;
    public uint y;
    public uint z;
};
