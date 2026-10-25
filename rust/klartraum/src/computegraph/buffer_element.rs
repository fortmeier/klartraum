// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Buffers as graph elements: one buffer per path, or one shared by all paths.

use std::cell::{Cell, Ref, RefCell};
use std::rc::Rc;

use anyhow::{Result, bail};
use ash::vk;
use bytemuck::Pod;

use crate::computegraph::element::{ComputeGraphElement, ElementBase};
use crate::vulkan_buffer::{DEFAULT_BUFFER_USAGE, HOST_COHERENT, VulkanBuffer};
use crate::vulkan_context::{Device, VulkanContext};

/// Untyped access to a buffer element's per-path Vulkan buffers.
pub trait BufferElementInterface {
    /// Byte size of one buffer; valid after setup.
    fn buffer_mem_size(&self) -> vk::DeviceSize;

    /// The Vulkan buffer of a path; valid after setup.
    fn vk_buffer(&self, path_id: u32) -> vk::Buffer;
}

/// What a [`BufferElement`] records each time its path runs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum RecordReset {
    #[default]
    None,
    Zero,
    ZeroRange {
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    },
    Fill(u32),
}

/// Owns one independent buffer of `T` per compute-graph path.
///
/// The graph's setup creates the allocations, so each path can update its data
/// while other frames are in flight. The default memory is host-visible and
/// coherent, for direct CPU updates of a path's buffer.
pub struct BufferElement<T: Pod> {
    base: ElementBase,
    device: Rc<Device>,
    /// Capacity of each buffer in elements.
    number_elements: u32,
    usage: vk::BufferUsageFlags,
    memory: vk::MemoryPropertyFlags,
    /// One buffer per path, created by the first graph that sets the element up.
    buffers: RefCell<Vec<VulkanBuffer<T>>>,
    reset: Cell<RecordReset>,
}

/// Buffer element holding the arguments of an indirect draw (`vkCmdDrawIndirect`).
pub type DrawIndirectCommandBufferElement = BufferElement<DrawIndirectCommand>;

/// `VkDrawIndirectCommand` as a plain-data type.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

impl<T: Pod> BufferElement<T> {
    /// Per-path storage buffers of `number_elements` elements in host-coherent memory.
    pub fn new(context: &VulkanContext, number_elements: u32) -> Rc<Self> {
        Self::with_usage(context, number_elements, DEFAULT_BUFFER_USAGE, HOST_COHERENT)
    }

    /// Per-path buffers with explicit usage and memory properties; pass DEVICE_LOCAL
    /// without HOST_VISIBLE for GPU-only data (CPU access then goes through staging).
    pub fn with_usage(
        context: &VulkanContext,
        number_elements: u32,
        usage: vk::BufferUsageFlags,
        memory: vk::MemoryPropertyFlags,
    ) -> Rc<Self> {
        Rc::new(Self {
            base: ElementBase::new(),
            device: context.device().clone(),
            number_elements,
            usage,
            memory,
            buffers: RefCell::new(Vec::new()),
            reset: Cell::new(RecordReset::None),
        })
    }

    /// The buffer of a path, after setup.
    pub fn buffer(&self, path_id: u32) -> Ref<'_, VulkanBuffer<T>> {
        Ref::map(self.buffers.borrow(), |b| &b[path_id as usize])
    }

    /// Number of paths with a buffer (0 before setup).
    pub fn number_of_buffers(&self) -> usize {
        self.buffers.borrow().len()
    }

    /// Clears every path's buffer now.
    pub fn zero(&self) -> Result<()> {
        self.buffers.borrow().iter().try_for_each(|b| b.zero())
    }

    /// Zeroes the whole buffer on the GPU each time the path runs; set before compiling.
    pub fn set_record_to_zero(&self, zero: bool) {
        if matches!(self.reset.get(), RecordReset::ZeroRange { .. }) {
            // Range zeroing takes precedence.
        } else if zero {
            self.reset.set(RecordReset::Zero);
        } else if self.reset.get() == RecordReset::Zero {
            self.reset.set(RecordReset::None);
        }
    }

    /// Zeroes the bytes `[offset, offset + size)` each time the path runs, leaving the rest;
    /// takes precedence over whole-buffer zeroing and filling. Set before compiling.
    pub fn set_record_to_zero_range(&self, offset: vk::DeviceSize, size: vk::DeviceSize) {
        self.reset.set(RecordReset::ZeroRange { offset, size });
    }

    /// Fills the buffer with a repeating 32-bit pattern each time the path runs; set before compiling.
    pub fn set_record_to_fill(&self, value: u32) {
        if !matches!(self.reset.get(), RecordReset::Zero | RecordReset::ZeroRange { .. }) {
            self.reset.set(RecordReset::Fill(value));
        }
    }
}

impl<T: Pod> ComputeGraphElement for BufferElement<T> {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "BufferElement"
    }

    // A buffer element may be part of several graphs; the first one allocates the
    // buffers and the others, which must have as many paths, reuse them.
    fn setup(&self, _context: &VulkanContext, number_paths: u32) -> Result<()> {
        let mut buffers = self.buffers.borrow_mut();
        if !buffers.is_empty() {
            if buffers.len() != number_paths as usize {
                bail!("BufferElement: shared by graphs with differing path counts");
            }
        } else {
            for _ in 0..number_paths {
                buffers.push(VulkanBuffer::with_usage(
                    &self.device,
                    self.number_elements,
                    self.usage,
                    self.memory,
                )?);
            }
        }
        self.base.set_initialized(true);
        Ok(())
    }

    fn record(&self, cmd: vk::CommandBuffer, path_id: u32) -> Result<()> {
        let buffer = self.buffer(path_id);
        match self.reset.get() {
            RecordReset::None => {}
            RecordReset::Zero => buffer.record_zero(cmd),
            RecordReset::ZeroRange { offset, size } => buffer.record_zero_range(cmd, offset, size),
            RecordReset::Fill(value) => buffer.record_fill(cmd, value),
        }
        Ok(())
    }

    fn as_buffer(&self) -> Option<&dyn BufferElementInterface> {
        Some(self)
    }
}

impl<T: Pod> BufferElementInterface for BufferElement<T> {
    fn buffer_mem_size(&self) -> vk::DeviceSize {
        self.buffers.borrow()[0].buffer_mem_size()
    }

    fn vk_buffer(&self, path_id: u32) -> vk::Buffer {
        self.buffers.borrow()[path_id as usize].buffer()
    }
}

/// Owns one buffer of `T` shared by every path, allocated at construction,
/// e.g. for immutable model data. Writes must be synchronized with its GPU use by all paths.
pub struct BufferElementSinglePath<T: Pod> {
    base: ElementBase,
    buffer: VulkanBuffer<T>,
}

impl<T: Pod> BufferElementSinglePath<T> {
    pub fn new(
        context: &VulkanContext,
        number_elements: u32,
        usage: vk::BufferUsageFlags,
        memory: vk::MemoryPropertyFlags,
    ) -> Result<Rc<Self>> {
        Ok(Rc::new(Self {
            base: ElementBase::new(),
            buffer: VulkanBuffer::with_usage(context.device(), number_elements, usage, memory)?,
        }))
    }

    /// The shared buffer.
    pub fn buffer(&self) -> &VulkanBuffer<T> {
        &self.buffer
    }
}

impl<T: Pod> ComputeGraphElement for BufferElementSinglePath<T> {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "BufferElementSinglePath"
    }

    fn setup(&self, _context: &VulkanContext, _number_paths: u32) -> Result<()> {
        Ok(())
    }

    fn record(&self, _cmd: vk::CommandBuffer, _path_id: u32) -> Result<()> {
        Ok(())
    }

    fn as_buffer(&self) -> Option<&dyn BufferElementInterface> {
        Some(self)
    }
}

impl<T: Pod> BufferElementInterface for BufferElementSinglePath<T> {
    fn buffer_mem_size(&self) -> vk::DeviceSize {
        self.buffer.buffer_mem_size()
    }

    fn vk_buffer(&self, _path_id: u32) -> vk::Buffer {
        self.buffer.buffer()
    }
}
