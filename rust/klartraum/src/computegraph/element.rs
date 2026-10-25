// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! The element trait every node of a compute graph implements.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use anyhow::{Result, anyhow, bail};
use ash::vk;

use crate::computegraph::buffer_element::BufferElementInterface;
use crate::computegraph::image_view_src::ImageViewSrcInterface;
use crate::computegraph::uniform_buffer_object::UniformBufferObjectInterface;
use crate::vulkan_context::VulkanContext;

/// A shared handle to a graph element; graphs and elements share their inputs.
pub type ElementPtr = Rc<dyn ComputeGraphElement>;

/// The state every element has: its inputs, dependencies, wait semaphores and name.
///
/// Elements are shared (`Rc`) and configured after they are connected, so the
/// state uses interior mutability, as the C++ elements are mutated through `shared_ptr`s.
#[derive(Default)]
pub struct ElementBase {
    /// The inputs by index. Each input is also the element's output of the same index.
    inputs: RefCell<BTreeMap<i32, ElementPtr>>,
    /// Per input index, the slot of the input element that is the actual input, or -1 for the element itself.
    src_output_slots: RefCell<BTreeMap<i32, i32>>,
    /// Execution-only predecessors that are not shader inputs.
    dependencies: RefCell<Vec<ElementPtr>>,
    /// Per path, an external semaphore the path's submission waits for (e.g. swapchain image acquisition).
    render_wait_semaphores: RefCell<BTreeMap<u32, vk::Semaphore>>,
    /// Set by `setup`; recording an element that was not set up is an error.
    initialized: Cell<bool>,
    name: RefCell<String>,
}

impl ElementBase {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.get()
    }

    pub fn set_initialized(&self, initialized: bool) {
        self.initialized.set(initialized);
    }

    /// The inputs as stored, without resolving slots.
    pub fn inputs(&self) -> BTreeMap<i32, ElementPtr> {
        self.inputs.borrow().clone()
    }

    pub fn number_of_inputs(&self) -> usize {
        self.inputs.borrow().len()
    }

    pub fn render_wait_semaphore(&self, path_id: u32) -> Option<vk::Semaphore> {
        self.render_wait_semaphores.borrow().get(&path_id).copied()
    }
}

/// The address of an element, which identifies it in a graph.
pub fn element_key(element: &ElementPtr) -> usize {
    Rc::as_ptr(element) as *const () as usize
}

/// `element` as its concrete type, as `std::dynamic_pointer_cast` in C++.
pub fn downcast<T: ComputeGraphElement>(element: &ElementPtr) -> Option<Rc<T>> {
    let any: Rc<dyn Any> = element.clone();
    any.downcast::<T>().ok()
}

/// A node of a compute graph.
///
/// Each element can have several inputs; every input is passed through and is
/// also the element's output of the same index. A graph compiled from an element
/// ([`ComputeGraph`](crate::computegraph::ComputeGraph)) records every element it
/// reaches through the inputs into one command buffer per path.
///
/// The `setup`, `record` and `update` hooks correspond to the C++ `_setup`,
/// `_record` and `_update`.
pub trait ComputeGraphElement: Any {
    /// The element's shared state.
    fn base(&self) -> &ElementBase;

    /// The element type, used for messages and profiling labels.
    fn type_name(&self) -> &'static str;

    /// Checks that `input` may be connected at `index`.
    fn check_input(&self, _input: &ElementPtr, _index: i32) -> Result<()> {
        bail!("check_input not implemented for {}", self.type_name())
    }

    /// Creates the element's resources for `number_paths` paths.
    fn setup(&self, _context: &VulkanContext, _number_paths: u32) -> Result<()> {
        self.base().set_initialized(true);
        Ok(())
    }

    /// Records the element's GPU work for a path.
    fn record(&self, _command_buffer: vk::CommandBuffer, _path_id: u32) -> Result<()> {
        if !self.base().is_initialized() {
            bail!("{} not initialized", self.type_name());
        }
        Ok(())
    }

    /// Host-side work right before a path is submitted, once the path's previous
    /// submission has finished; only called when [`is_updatable`](Self::is_updatable).
    fn update(&self, _path_id: u32) -> Result<()> {
        Ok(())
    }

    /// Whether the element has a host-side update.
    fn is_updatable(&self) -> bool {
        false
    }

    /// The elements this one reads; a graph traverses them. Groups return their outputs here.
    fn inputs(&self) -> BTreeMap<i32, ElementPtr> {
        self.base().inputs()
    }

    /// Execution-only predecessors added with [`add_dependency`](Self::add_dependency).
    fn dependencies(&self) -> Vec<ElementPtr> {
        self.base().dependencies.borrow().clone()
    }

    /// The element as a buffer, if it is one.
    fn as_buffer(&self) -> Option<&dyn BufferElementInterface> {
        None
    }

    /// The element as an image source, if it is one.
    fn as_image_view_src(&self) -> Option<&dyn ImageViewSrcInterface> {
        None
    }

    /// The element as a uniform buffer, if it is one.
    fn as_uniform_buffer(&self) -> Option<&dyn UniformBufferObjectInterface> {
        None
    }

    /// The name, or empty.
    fn name(&self) -> String {
        self.base().name.borrow().clone()
    }

    fn set_name(&self, name: &str) {
        *self.base().name.borrow_mut() = name.to_owned();
    }

    /// Makes `input` the input `index`.
    fn set_input(&self, input: ElementPtr, index: i32) -> Result<()> {
        self.set_input_slot(input, index, -1)
    }

    /// Makes input `slot` of `input` the input `index` of this element, or `input`
    /// itself for slot -1. With a slot this element depends on `input`, so it runs
    /// after it — e.g. the buffer at a slot is written by `input`.
    fn set_input_slot(&self, input: ElementPtr, index: i32, slot: i32) -> Result<()> {
        let checked = if slot == -1 {
            input.clone()
        } else {
            input.input_element(slot)?
        };
        self.check_input(&checked, index)?;
        self.base().inputs.borrow_mut().insert(index, input);
        self.base().src_output_slots.borrow_mut().insert(index, slot);
        Ok(())
    }

    /// The actual input `index`, resolving slots.
    fn input_element(&self, index: i32) -> Result<ElementPtr> {
        let input = self
            .base()
            .inputs
            .borrow()
            .get(&index)
            .cloned()
            .ok_or_else(|| anyhow!("{}: no input {index}", self.type_name()))?;
        let slot = self.base().src_output_slots.borrow().get(&index).copied().unwrap_or(-1);
        if slot != -1 {
            input.input_element(slot)
        } else {
            Ok(input)
        }
    }

    /// The output `index`, which is the input `index` as stored.
    fn output_element(&self, index: i32) -> Result<ElementPtr> {
        self.base()
            .inputs
            .borrow()
            .get(&index)
            .cloned()
            .ok_or_else(|| anyhow!("{}: no output {index}", self.type_name()))
    }

    /// The submission of path `path_id` waits for `semaphore`.
    fn set_wait_for(&self, path_id: u32, semaphore: vk::Semaphore) {
        self.base()
            .render_wait_semaphores
            .borrow_mut()
            .insert(path_id, semaphore);
    }

    /// Adds an execution-only dependency: `dependency` executes before this element.
    fn add_dependency(&self, dependency: ElementPtr) -> Result<()> {
        if std::ptr::addr_eq(Rc::as_ptr(&dependency), self as *const Self) {
            bail!("ComputeGraphElement: element cannot depend on itself");
        }
        let mut dependencies = self.base().dependencies.borrow_mut();
        if !dependencies.iter().any(|d| element_key(d) == element_key(&dependency)) {
            dependencies.push(dependency);
        }
        Ok(())
    }
}

/// A plain element that only orders others, e.g. for tests of graph ordering.
#[derive(Default)]
pub struct NoOp {
    base: ElementBase,
}

impl NoOp {
    pub fn new() -> Rc<Self> {
        Rc::new(Self::default())
    }
}

impl ComputeGraphElement for NoOp {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "NoOp"
    }

    fn check_input(&self, _input: &ElementPtr, _index: i32) -> Result<()> {
        Ok(())
    }
}
