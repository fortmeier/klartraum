// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Index helpers for ONNX operators with numpy-style broadcasting over four axes.
module broadcast;

// Returns the index into a tensor of shape `dims` that broadcasts to element
// `linear` of a tensor of shape `outputDims`. Axes of size 1 repeat.
public uint broadcastIndex(uint linear, uint dims[4], uint outputDims[4]) {
    uint coordinates[4];
    uint remaining = linear;
    for (int axis = 3; axis >= 0; --axis) {
        coordinates[axis] = remaining % outputDims[axis];
        remaining /= outputDims[axis];
    }
    uint index = 0;
    uint stride = 1;
    for (int axis = 3; axis >= 0; --axis) {
        index += (dims[axis] == 1 ? 0 : coordinates[axis]) * stride;
        stride *= dims[axis];
    }
    return index;
}
