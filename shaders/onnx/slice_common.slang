// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Index mapping shared by the Slice shaders. The starts, ends, axes and steps
// tensors are bound for ONNX slot compatibility; the slice itself comes in
// through the push constants.
module slice_common;

public struct SlicePushConstants {
    public uint rank;
    public uint axis;
    public uint start;
    public uint elementCount;
    public uint inputDims[4];
    public uint outputDims[4];
};

// Returns the input index that output element `outputIndex` copies.
public uint sliceInputIndex(uint outputIndex, SlicePushConstants pc) {
    uint coordinates[4];
    uint remaining = outputIndex;
    for (int dimension = int(pc.rank) - 1; dimension >= 0; --dimension) {
        coordinates[dimension] = remaining % pc.outputDims[dimension];
        remaining /= pc.outputDims[dimension];
    }
    coordinates[pc.axis] += pc.start;
    uint inputIndex = 0;
    uint stride = 1;
    for (int dimension = int(pc.rank) - 1; dimension >= 0; --dimension) {
        inputIndex += coordinates[dimension] * stride;
        stride *= pc.inputDims[dimension];
    }
    return inputIndex;
}
