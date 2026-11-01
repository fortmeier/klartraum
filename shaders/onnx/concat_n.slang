// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Concatenation of 3 to 8 tensors along one axis, shared by concat<N>.comp.slang.
// One invocation per output element: it finds the input whose axis range holds
// its axis index and copies the element.
module concat_n;

public struct ConcatPushConstants {
    public uint outerCount;
    public uint innerSize;
    public uint elementCount;
    public uint totalAxis;
    public uint axisSize[8];
};

// Reads element `index` of input tensor `input`; implemented by each shader
// over its own input bindings.
public interface IConcatInputs {
    static const uint count;
    static float load(uint input, uint index);
}

// Returns output element `index`, or nothing for an invocation past the end.
public Optional<float> concatElement<Inputs : IConcatInputs>(uint index, ConcatPushConstants pc) {
    if (index >= pc.elementCount) return none;
    uint block = pc.totalAxis * pc.innerSize;
    uint outer = index / block;
    uint within = index % block;
    uint axisIndex = within / pc.innerSize;
    uint innerIndex = within % pc.innerSize;
    uint start = 0;
    [ForceUnroll]
    for (uint input = 0; input < Inputs.count; ++input) {
        if (axisIndex < start + pc.axisSize[input]) {
            return Inputs.load(input, (outer * pc.axisSize[input] + axisIndex - start) * pc.innerSize + innerIndex);
        }
        start += pc.axisSize[input];
    }
    return none;
}
