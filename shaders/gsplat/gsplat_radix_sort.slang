// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Bindings, push constants and helpers shared by the three radix sort passes
// (gsplat_radix_sort_histogram, _hist_prefix_sum and _hist_scatter), which
// share one descriptor set.
module gsplat_radix_sort;

__exported import scalar_buffer;

public struct RadixSortPushConstants {
    public uint pass;
    public uint numElements;
    public uint numBins;
    public uint useCountBuffer; // 1 = take the active element count from countBuffer
                                // (binding 6) instead of numElements, so the sort
                                // processes only a GPU-determined visible count
};
[[vk::push_constant]] public ConstantBuffer<RadixSortPushConstants> pushConstants;

[[vk::binding(0, 0)]] public RWScalarBuffer<uint> radixValuesA;
[[vk::binding(1, 0)]] public RWScalarBuffer<uint> indexesA;
[[vk::binding(2, 0)]] public RWScalarBuffer<uint> radixValuesB;
[[vk::binding(3, 0)]] public RWScalarBuffer<uint> indexesB;
[[vk::binding(4, 0)]] public RWScalarBuffer<uint> counts;
[[vk::binding(5, 0)]] public RWScalarBuffer<uint> offsets;
// One uint: the number of elements to sort, written on the GPU.
[[vk::binding(6, 0)]] public RWScalarBuffer<uint> numberTotalGaussians;
[[vk::binding(7, 0)]] public RWScalarBuffer<uint> histogram;

// Active element count for this dispatch. When useCountBuffer is set the count
// comes from numberTotalGaussians (written on the GPU before the sort runs);
// otherwise it is the static push-constant numElements. Keeping the
// work-distribution math driven by this value while the dispatch group count
// stays fixed preserves the histogram's bin-major stride (= numWorkGroups).
public uint activeNumElements() {
    return (pushConstants.useCountBuffer != 0u) ? numberTotalGaussians[0] : pushConstants.numElements;
}

// Helper function to get the bin at a specific position
// this uses 16 bins (= 4 bits of the float value for each pass)
public uint getBin(uint value, uint binMask, uint pass) {
    // Invert the sign bit for correct float sorting
    value ^= 0x80000000u;

    // depending on the pass, we need to switch between the depth value and the binMask
    uint switched = pass >= (32 / 4) ? binMask : value;
    // for pass >= 32, we would need to either subtract 32 or
    // use the modulo operator
    pass = pass % (32 / 4);
    // shift the value to the right by 4 * pass bits
    // and mask it to get the last 4 bits
    return (switched >> (pass * 4)) & 0xF;
}
