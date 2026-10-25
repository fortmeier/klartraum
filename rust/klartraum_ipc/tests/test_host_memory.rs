// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - shared_memory_is_visible_through_a_second_mapping: a region opened by name sees the writes of
 *   the creating mapping, and its size is rounded up to the region alignment
 * - computation_writes_into_imported_shared_memory: operator_add from SPIR-V over three
 *   shared-memory regions imported as Vulkan buffers writes A + B straight into the result
 *   mapping (no copy), and a second run sees new inputs written into the input mappings
 * - raw_push_constants_reach_the_shader: push constants given as raw bytes reach the shader: the
 *   radix sort's histogram pass, told the element count and bin count that way, counts n/16 keys per bin
 **/
use std::rc::Rc;

use klartraum::computegraph::{
    BufferElement, ComputeGraph, ComputeGraphElement, ElementPtr, GeneralComputation, HostMemoryBufferElement,
};
use klartraum::vulkan_context::{VulkanContext, validation_error_count};
use klartraum::vulkan_helpers::read_file;
use klartraum_ipc::shared_memory::{REGION_ALIGNMENT, SharedMemoryRegion};

fn spirv(path: &str) -> Vec<u32> {
    bytemuck::pod_collect_to_vec(&read_file(path).unwrap())
}

#[test]
fn shared_memory_is_visible_through_a_second_mapping() {
    let mut a = SharedMemoryRegion::create(100).unwrap();
    assert_eq!(a.as_slice().len(), REGION_ALIGNMENT);
    let b = SharedMemoryRegion::open(a.name(), REGION_ALIGNMENT).unwrap();
    a.as_mut_slice()[7] = 42;
    assert_eq!(b.as_slice()[7], 42);
}

#[test]
fn computation_writes_into_imported_shared_memory() {
    let vc = VulkanContext::new_headless().unwrap();
    if !HostMemoryBufferElement::is_supported(&vc) {
        eprintln!("skipped: VK_EXT_external_memory_host is not supported");
        return;
    }
    let errors = validation_error_count();
    let n = 1000usize;
    let bytes = n * 4;
    let mut a = SharedMemoryRegion::create(bytes).unwrap();
    let mut b = SharedMemoryRegion::create(bytes).unwrap();
    let r = SharedMemoryRegion::create(bytes).unwrap();
    let a_ptr = a.as_mut_slice().as_mut_ptr() as *mut f32;
    let b_ptr = b.as_mut_slice().as_mut_ptr() as *mut f32;
    let r_name = r.name().to_owned();
    let write = |a_ptr: *mut f32, b_ptr: *mut f32, offset: f32| unsafe {
        for i in 0..n {
            *a_ptr.add(i) = i as f32 + offset;
            *b_ptr.add(i) = 2.0 * i as f32;
        }
    };
    write(a_ptr, b_ptr, 0.0);

    let import = |region: SharedMemoryRegion| HostMemoryBufferElement::import(&vc, Rc::new(region), bytes).unwrap();
    let (ea, eb, er) = (import(a), import(b), import(r));
    let add = GeneralComputation::<()>::from_spirv(vec![spirv("shaders/operator_add.comp.spv")]);
    add.set_input(ea, 0).unwrap();
    add.set_input(eb, 1).unwrap();
    add.set_input(er, 2).unwrap();
    add.set_group_count_x(n as u32);
    let root: ElementPtr = add;
    let mut graph = ComputeGraph::new(&vc, 1);
    graph.compile_from(&root, &vc).unwrap();

    // The result is read through a separate mapping of the result object, as the client would.
    let result = SharedMemoryRegion::open(&r_name, REGION_ALIGNMENT).unwrap();
    let read = |region: &SharedMemoryRegion| -> Vec<f32> { bytemuck::pod_collect_to_vec(&region.as_slice()[..bytes]) };
    graph.submit_and_wait(vc.graphics_queue(), 0).unwrap();
    assert_eq!(read(&result), (0..n).map(|i| 3.0 * i as f32).collect::<Vec<_>>());

    write(a_ptr, b_ptr, 100.0);
    graph.submit_and_wait(vc.graphics_queue(), 0).unwrap();
    assert_eq!(
        read(&result),
        (0..n).map(|i| 3.0 * i as f32 + 100.0).collect::<Vec<_>>()
    );
    assert_eq!(validation_error_count(), errors);
}

#[test]
fn raw_push_constants_reach_the_shader() {
    let vc = VulkanContext::new_headless().unwrap();
    let errors = validation_error_count();
    let n = 256u32;
    // operator_multiply_scalar reads its scalar from a uniform buffer, so a raw push-constant
    // shader of the engine is used instead: the radix sort histogram (pass, count, bins, flag).
    let keys = BufferElement::<u32>::new(&vc, n);
    let indices = BufferElement::<u32>::new(&vc, n);
    let keys_out = BufferElement::<u32>::new(&vc, n);
    let indices_out = BufferElement::<u32>::new(&vc, n);
    let counts = BufferElement::<u32>::new(&vc, 16);
    let offsets = BufferElement::<u32>::new(&vc, 2);
    let total = BufferElement::<u32>::new(&vc, 1);
    let hist = BufferElement::<u32>::new(&vc, 16);
    let histogram =
        GeneralComputation::<()>::from_spirv(vec![spirv("shaders/gsplat/gsplat_radix_sort_histogram.comp.spv")]);
    histogram.set_input(keys.clone(), 0).unwrap();
    histogram.set_input(indices.clone(), 1).unwrap();
    histogram.set_input(keys_out.clone(), 2).unwrap();
    histogram.set_input(indices_out.clone(), 3).unwrap();
    histogram.add_scratch_buffer_element(counts.clone(), true);
    histogram.add_scratch_buffer_element(offsets.clone(), true);
    histogram.add_scratch_buffer_element(total.clone(), false);
    histogram.add_scratch_buffer_element(hist.clone(), true);
    // SortPushConstants { pass: 0, numElements: n, numBins: 16, useCountBuffer: 0 }.
    let pc: Vec<u8> = [0u32, n, 16, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    histogram.set_raw_push_constants(vec![pc]).unwrap();
    histogram.set_group_count_x(1);
    let root: ElementPtr = histogram;
    let mut graph = ComputeGraph::new(&vc, 1);
    graph.compile_from(&root, &vc).unwrap();
    // Keys 0..n: the low 4 bits spread evenly over the 16 bins.
    keys.buffer(0).memcopy_from(&(0..n).collect::<Vec<_>>()).unwrap();
    graph.submit_and_wait(vc.graphics_queue(), 0).unwrap();
    let hist = hist.buffer(0).read().unwrap();
    assert_eq!(hist.iter().sum::<u32>(), n, "histogram {hist:?}");
    assert!(hist.iter().all(|&c| c == n / 16), "histogram {hist:?}");
    assert_eq!(validation_error_count(), errors);
}
