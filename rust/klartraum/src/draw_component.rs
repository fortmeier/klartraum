// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`DrawComponent`], something a render pass draws with its own graphics pipeline.

use std::rc::Rc;

use anyhow::Result;
use ash::vk;

use crate::computegraph::uniform_buffer_object::CameraUbo;
use crate::vulkan_context::VulkanContext;

/// Something a [`RenderPass`](crate::computegraph::RenderPass) draws with its own graphics pipeline.
///
/// When the graph is compiled, the render pass calls [`initialize`](Self::initialize)
/// with its Vulkan render pass, its camera uniform buffer and the graph's number of
/// paths (per-path resources are sized by it), and then
/// [`record_command_buffer`](Self::record_command_buffer) once per path, inside the render pass.
pub trait DrawComponent {
    fn initialize(
        &self,
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        camera_ubo: Rc<CameraUbo>,
        number_paths: u32,
    ) -> Result<()>;

    fn record_command_buffer(&self, cmd: vk::CommandBuffer, framebuffer: vk::Framebuffer, path_id: u32) -> Result<()>;
}
