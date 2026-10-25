// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`BufferToGraphicsBarrier`], which orders an indirect draw after the compute work writing its buffers.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use anyhow::Result;
use ash::vk;

use crate::computegraph::element::{ComputeGraphElement, ElementBase, ElementPtr};
use crate::vulkan_context::{Device, VulkanContext};

/// Orders an indirect draw after the compute work that wrote its buffers.
///
/// An indirect draw that reads buffers written by compute shaders (sorted indices,
/// draw arguments) needs a buffer barrier whose destination is the indirect-draw
/// and vertex-shader stages. Placed between those compute elements (its inputs, so
/// they run first) and the render pass (which depends on it), this element records it.
#[derive(Default)]
pub struct BufferToGraphicsBarrier {
    base: ElementBase,
    /// The buffers the render pass reads: draw arguments and/or vertex-shader storage buffers.
    buffers: RefCell<Vec<ElementPtr>>,
    include_mesh_shader_stage: Cell<bool>,
    device: RefCell<Option<Rc<Device>>>,
}

impl BufferToGraphicsBarrier {
    pub fn new() -> Rc<Self> {
        Rc::new(Self::default())
    }

    /// Adds a buffer element the barrier covers.
    pub fn add_buffer(&self, buffer: ElementPtr) {
        assert!(
            buffer.as_buffer().is_some(),
            "BufferToGraphicsBarrier::add_buffer: not a buffer"
        );
        self.buffers.borrow_mut().push(buffer);
    }

    /// Also makes the writes visible to the mesh-shader stage (VK_EXT_mesh_shader must be enabled).
    pub fn set_include_mesh_shader_stage(&self, include: bool) {
        self.include_mesh_shader_stage.set(include);
    }
}

impl ComputeGraphElement for BufferToGraphicsBarrier {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "BufferToGraphicsBarrier"
    }

    // Any producer is accepted as an input, only so the graph schedules it before this barrier.
    fn check_input(&self, _input: &ElementPtr, _index: i32) -> Result<()> {
        Ok(())
    }

    fn setup(&self, context: &VulkanContext, _number_paths: u32) -> Result<()> {
        *self.device.borrow_mut() = Some(context.device().clone());
        self.base.set_initialized(true);
        Ok(())
    }

    fn record(&self, cmd: vk::CommandBuffer, path_id: u32) -> Result<()> {
        let device = self.device.borrow();
        let device = device.as_ref().expect("BufferToGraphicsBarrier recorded before setup");
        let barriers: Vec<_> = self
            .buffers
            .borrow()
            .iter()
            .map(|buffer| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer.as_buffer().unwrap().vk_buffer(path_id))
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();
        if barriers.is_empty() {
            return Ok(());
        }
        let mut dst_stage = vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::VERTEX_SHADER;
        if self.include_mesh_shader_stage.get() {
            dst_stage |= vk::PipelineStageFlags::MESH_SHADER_EXT;
        }
        unsafe {
            device.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            )
        };
        Ok(())
    }
}
