// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! The compute graph framework: elements and the graph that compiles them into command buffers.

pub mod buffer_element;
pub mod buffer_to_graphics_barrier;
pub mod element;
pub mod general_computation;
pub mod graph;
pub mod image_view_src;
pub mod render_pass;
pub mod uniform_buffer_object;

pub use buffer_element::{
    BufferElement, BufferElementInterface, BufferElementSinglePath, DrawIndirectCommand,
    DrawIndirectCommandBufferElement,
};
pub use buffer_to_graphics_barrier::BufferToGraphicsBarrier;
pub use element::{ComputeGraphElement, ElementBase, ElementPtr, NoOp, downcast, element_key};
pub use general_computation::{GeneralComputation, ImageLayoutTransition};
pub use graph::ComputeGraph;
pub use image_view_src::{ImageViewSrc, ImageViewSrcInterface, record_clear_image};
pub use render_pass::RenderPass;
pub use uniform_buffer_object::{CameraUbo, UniformBufferObject, UniformBufferObjectInterface};
