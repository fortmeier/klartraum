// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Images a graph renders into or reads, one per path.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use anyhow::Result;
use ash::vk;

use crate::computegraph::element::{ComputeGraphElement, ElementBase};
use crate::vulkan_context::{Device, VulkanContext, color_subresource_range};

/// Records a clear of `image` to `color` and leaves it in `GENERAL` for whatever reads
/// or writes it next. The previous contents are discarded; the clear waits for all
/// earlier work on the queue (e.g. the previous frame's reads of a swapchain image).
pub fn record_clear_image(device: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image, color: vk::ClearColorValue) {
    let range = color_subresource_range();
    let to_transfer = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
    let to_general = to_transfer
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer],
        );
        device.cmd_clear_color_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &color, &[range]);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_general],
        );
    }
}

/// Access to the per-path images of an image-providing element.
pub trait ImageViewSrcInterface {
    fn image_view(&self, path_id: u32) -> vk::ImageView;
    fn image(&self, path_id: u32) -> vk::Image;
    fn image_extent(&self, path_id: u32) -> vk::Extent2D;

    /// The layout an image-writing consumer should leave the image in, or None for the
    /// consumer's default (PRESENT_SRC for a presentable swapchain image, GENERAL otherwise).
    fn final_layout_override(&self) -> Option<vk::ImageLayout> {
        None
    }
}

/// The images a graph renders into or reads, one per path.
///
/// By default the source clears its image to opaque black every time its path runs,
/// before any element that uses it, and leaves it in `GENERAL`: renderers such as
/// the Gaussian splatting backends draw over what the image holds, so a fresh frame
/// starts from black. `set_clear(false)` keeps the previous contents; the image
/// must then already be in `GENERAL`. Clearing needs TRANSFER_DST usage.
pub struct ImageViewSrc {
    base: ElementBase,
    image_views: Vec<vk::ImageView>,
    images: Vec<vk::Image>,
    image_extents: Vec<vk::Extent2D>,
    clear_enabled: Cell<bool>,
    clear_color: Cell<[f32; 4]>,
    /// Set by setup, for recording the clear.
    device: RefCell<Option<Rc<Device>>>,
}

impl ImageViewSrc {
    pub fn new(image_views: Vec<vk::ImageView>, images: Vec<vk::Image>, image_extents: Vec<vk::Extent2D>) -> Rc<Self> {
        Rc::new(Self {
            base: ElementBase::new(),
            image_views,
            images,
            image_extents,
            clear_enabled: Cell::new(true),
            clear_color: Cell::new([0.0, 0.0, 0.0, 1.0]),
            device: RefCell::new(None),
        })
    }

    /// An image source for the swapchain images of `context`, whose paths wait for the images to be acquired.
    pub fn for_swap_chain(context: &VulkanContext) -> Rc<Self> {
        let n = context.number_of_swap_chain_images();
        let source = Self::new(
            (0..n).map(|i| context.image_view(i)).collect(),
            (0..n).map(|i| context.swap_chain_image(i)).collect(),
            vec![context.swap_chain_extent(); n as usize],
        );
        for (i, &semaphore) in context.image_available_semaphores_per_image().iter().enumerate() {
            source.set_wait_for(i as u32, semaphore);
        }
        source
    }

    /// Whether the image is cleared each time its path runs (default: yes).
    pub fn set_clear(&self, clear: bool) {
        self.clear_enabled.set(clear);
    }

    pub fn clears(&self) -> bool {
        self.clear_enabled.get()
    }

    /// The colour the image is cleared to (default: opaque black).
    pub fn set_clear_color(&self, color: [f32; 4]) {
        self.clear_color.set(color);
    }
}

impl ComputeGraphElement for ImageViewSrc {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "ImageViewSrc"
    }

    fn setup(&self, context: &VulkanContext, _number_paths: u32) -> Result<()> {
        *self.device.borrow_mut() = Some(context.device().clone());
        self.base.set_initialized(true);
        Ok(())
    }

    fn record(&self, cmd: vk::CommandBuffer, path_id: u32) -> Result<()> {
        if self.clear_enabled.get() && (path_id as usize) < self.images.len() {
            let device = self.device.borrow();
            let device = device.as_ref().expect("ImageViewSrc recorded before setup");
            let color = vk::ClearColorValue {
                float32: self.clear_color.get(),
            };
            record_clear_image(&device.device, cmd, self.images[path_id as usize], color);
        }
        Ok(())
    }

    fn as_image_view_src(&self) -> Option<&dyn ImageViewSrcInterface> {
        Some(self)
    }
}

impl ImageViewSrcInterface for ImageViewSrc {
    fn image_view(&self, path_id: u32) -> vk::ImageView {
        self.image_views[path_id as usize]
    }

    fn image(&self, path_id: u32) -> vk::Image {
        self.images[path_id as usize]
    }

    fn image_extent(&self, path_id: u32) -> vk::Extent2D {
        self.image_extents[path_id as usize]
    }
}
