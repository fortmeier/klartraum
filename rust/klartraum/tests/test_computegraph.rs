// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - order_puts_predecessors_first: in a diamond of elements every element is ordered after its
 *   inputs, the shared input once, and the root last
 * - cycles_are_rejected: two elements that are each other's input cannot be ordered, nor can a
 *   cycle further up the graph
 * - execution_only_dependencies_are_unique: a dependency added twice is one predecessor, and an
 *   element cannot depend on itself
 * - slot_inputs_resolve_to_the_slot_element: an input given as a slot of another element resolves
 *   to that element's input, while the graph orders the other element first
 * - general_computation_adds_on_every_path: operator_add over per-path buffers computes A + B on
 *   every path with submit_and_wait, without validation errors
 * - chained_computations_run_in_order: a doubling feeding an addition through a slot input
 *   computes 2A + B, so the graph records the producer first
 * - submit_requires_a_compiled_graph: submitting a graph that was not compiled is an error
 * - profiling_reports_every_element: a profiled graph reports one time per element and a GPU span
 **/
mod common;

use std::rc::Rc;

use ash::vk;
use klartraum::computegraph::{
    BufferElement, ComputeGraph, ComputeGraphElement, ElementPtr, GeneralComputation, NoOp, element_key,
};
use klartraum::vulkan_context::{VulkanContext, validation_error_count};

fn position(order: &[ElementPtr], element: &ElementPtr) -> usize {
    order
        .iter()
        .position(|e| element_key(e) == element_key(element))
        .expect("element not in the order")
}

#[test]
fn order_puts_predecessors_first() {
    let (a, b, c, root) = (NoOp::new(), NoOp::new(), NoOp::new(), NoOp::new());
    b.set_input(a.clone(), 0).unwrap();
    c.set_input(a.clone(), 0).unwrap();
    root.set_input(b.clone(), 0).unwrap();
    root.set_input(c.clone(), 1).unwrap();
    let root: ElementPtr = root;
    let order = ComputeGraph::topological_order(&root).unwrap();
    assert_eq!(order.len(), 4);
    let (a, b, c): (ElementPtr, ElementPtr, ElementPtr) = (a, b, c);
    assert!(position(&order, &a) < position(&order, &b));
    assert!(position(&order, &a) < position(&order, &c));
    assert_eq!(position(&order, &root), 3);
}

#[test]
fn cycles_are_rejected() {
    let (a, b) = (NoOp::new(), NoOp::new());
    a.set_input(b.clone(), 0).unwrap();
    b.set_input(a.clone(), 0).unwrap();
    let root = NoOp::new();
    root.set_input(a.clone(), 0).unwrap();
    let (a_ptr, root): (ElementPtr, ElementPtr) = (a.clone(), root);
    assert!(ComputeGraph::topological_order(&a_ptr).is_err());
    assert!(ComputeGraph::topological_order(&root).is_err());
    // Break the reference cycle so the elements are freed.
    b.set_input(NoOp::new(), 0).unwrap();
}

#[test]
fn execution_only_dependencies_are_unique() {
    let (a, b) = (NoOp::new(), NoOp::new());
    a.add_dependency(b.clone()).unwrap();
    a.add_dependency(b.clone()).unwrap();
    a.set_input(b.clone(), 0).unwrap();
    let a: ElementPtr = a;
    assert_eq!(ComputeGraph::predecessors(&a).len(), 1);
    assert!(a.add_dependency(a.clone()).is_err());
}

#[test]
fn slot_inputs_resolve_to_the_slot_element() {
    let (buffer, producer, consumer) = (NoOp::new(), NoOp::new(), NoOp::new());
    producer.set_input(buffer.clone(), 2).unwrap();
    consumer.set_input_slot(producer.clone(), 0, 2).unwrap();
    let resolved = consumer.input_element(0).unwrap();
    let buffer: ElementPtr = buffer;
    assert_eq!(element_key(&resolved), element_key(&buffer));
    let consumer: ElementPtr = consumer;
    let order = ComputeGraph::topological_order(&consumer).unwrap();
    let producer: ElementPtr = producer;
    assert!(position(&order, &producer) < position(&order, &consumer));
}

fn add_graph(
    vc: &VulkanContext,
    n: u32,
) -> (
    Rc<BufferElement<f32>>,
    Rc<BufferElement<f32>>,
    Rc<BufferElement<f32>>,
    ElementPtr,
) {
    let a = BufferElement::<f32>::new(vc, n);
    let b = BufferElement::<f32>::new(vc, n);
    let r = BufferElement::<f32>::new(vc, n);
    let add = GeneralComputation::<()>::new("shaders/operator_add.comp.spv");
    add.set_input(a.clone(), 0).unwrap();
    add.set_input(b.clone(), 1).unwrap();
    add.set_input(r.clone(), 2).unwrap();
    add.set_group_count_x(n);
    (a, b, r, add)
}

#[test]
fn general_computation_adds_on_every_path() {
    let _gpu = common::gpu_lock();
    let errors = validation_error_count();
    let vc = VulkanContext::new_headless().unwrap();
    let n = 256;
    let (a, b, r, add) = add_graph(&vc, n);
    let mut graph = ComputeGraph::new(&vc, 2);
    graph.compile_from(&add, &vc).unwrap();
    for path in 0..2u32 {
        let av: Vec<f32> = (0..n).map(|i| i as f32 + path as f32).collect();
        let bv: Vec<f32> = (0..n).map(|i| 1000.0 - i as f32).collect();
        a.buffer(path).memcopy_from(&av).unwrap();
        b.buffer(path).memcopy_from(&bv).unwrap();
        graph.submit_and_wait(vc.graphics_queue(), path).unwrap();
        let expected: Vec<f32> = av.iter().zip(&bv).map(|(x, y)| x + y).collect();
        assert_eq!(r.buffer(path).read().unwrap(), expected);
    }
    assert_eq!(validation_error_count(), errors);
}

#[test]
fn chained_computations_run_in_order() {
    let _gpu = common::gpu_lock();
    let errors = validation_error_count();
    let vc = VulkanContext::new_headless().unwrap();
    let n = 64;
    let a = BufferElement::<f32>::new(&vc, n);
    let unused = BufferElement::<f32>::new(&vc, n);
    let doubled = BufferElement::<f32>::new(&vc, n);
    let b = BufferElement::<f32>::new(&vc, n);
    let r = BufferElement::<f32>::new(&vc, n);

    let double = GeneralComputation::<()>::new("shaders/operator_double.comp.spv");
    double.set_input(a.clone(), 0).unwrap();
    double.set_input(unused.clone(), 1).unwrap();
    double.set_input(doubled.clone(), 2).unwrap();
    double.set_group_count_x(n);

    let add = GeneralComputation::<()>::new("shaders/operator_add.comp.spv");
    add.set_input_slot(double.clone(), 0, 2).unwrap();
    add.set_input(b.clone(), 1).unwrap();
    add.set_input(r.clone(), 2).unwrap();
    add.set_group_count_x(n);

    let mut graph = ComputeGraph::new(&vc, 1);
    let root: ElementPtr = add;
    graph.compile_from(&root, &vc).unwrap();
    let av: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let bv: Vec<f32> = vec![0.5; n as usize];
    a.buffer(0).memcopy_from(&av).unwrap();
    b.buffer(0).memcopy_from(&bv).unwrap();
    graph.submit_and_wait(vc.graphics_queue(), 0).unwrap();
    let expected: Vec<f32> = av.iter().map(|x| 2.0 * x + 0.5).collect();
    assert_eq!(r.buffer(0).read().unwrap(), expected);
    assert_eq!(validation_error_count(), errors);
}

#[test]
fn submit_requires_a_compiled_graph() {
    let _gpu = common::gpu_lock();
    let vc = VulkanContext::new_headless().unwrap();
    let mut graph = ComputeGraph::new(&vc, 1);
    assert!(graph.submit_to(vc.graphics_queue(), 0, vk::Fence::null()).is_err());
}

#[test]
fn profiling_reports_every_element() {
    let _gpu = common::gpu_lock();
    let vc = VulkanContext::new_headless().unwrap();
    let (_a, _b, _r, add) = add_graph(&vc, 1024);
    add.set_name("Add");
    let mut graph = ComputeGraph::new(&vc, 1);
    graph.enable_profiling();
    graph.compile_from(&add, &vc).unwrap();
    for _ in 0..3 {
        graph.submit_and_wait(vc.graphics_queue(), 0).unwrap();
    }
    let results = graph.profiling_results();
    assert_eq!(results.len(), graph.elements().len());
    assert_eq!(results.last().unwrap().0, "Add");
    assert!(graph.gpu_span_ms() > 0.0);
}
