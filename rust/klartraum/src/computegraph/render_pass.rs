// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`RenderPass`], a Vulkan render pass over the image of its input 0 that runs draw components.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use anyhow::{Result, bail};
use ash::vk;

use crate::computegraph::element::{ComputeGraphElement, ElementBase, ElementPtr, downcast};
use crate::computegraph::image_view_src::ImageViewSrcInterface;
use crate::computegraph::uniform_buffer_object::CameraUbo;
use crate::draw_component::DrawComponent;
use crate::vulkan_context::{Device, VulkanContext, color_subresource_range};

/// The Vulkan objects of a set-up [`RenderPass`].
struct RenderPassResources {
    device: Rc<Device>,
    render_pass: vk::RenderPass,
    /// One framebuffer per path, over that path's image.
    framebuffers: Vec<vk::Framebuffer>,
}

impl Drop for RenderPassResources {
    fn drop(&mut self) {
        unsafe {
            for &framebuffer in &self.framebuffers {
                self.device.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

/// A render pass over the image of input 0 (an image source) with the camera
/// uniform buffer at input 1, drawing its draw components in order.
///
/// It stands for its target image: as an image source it forwards input 0's
/// images, so later elements can use what it rendered.
pub struct RenderPass {
    base: ElementBase,
    format: vk::Format,
    extent: vk::Extent2D,
    /// Draw over the image's contents instead of clearing it.
    load_existing: Cell<bool>,
    draw_components: RefCell<Vec<Rc<dyn DrawComponent>>>,
    /// Elements scheduled before the pass that are not inputs, e.g. compute work an indirect draw consumes.
    compute_dependencies: RefCell<Vec<ElementPtr>>,
    resources: RefCell<Option<RenderPassResources>>,
}

impl RenderPass {
    pub fn new(format: vk::Format, extent: vk::Extent2D) -> Rc<Self> {
        Rc::new(Self {
            base: ElementBase::new(),
            format,
            extent,
            load_existing: Cell::new(false),
            draw_components: RefCell::new(Vec::new()),
            compute_dependencies: RefCell::new(Vec::new()),
            resources: RefCell::new(None),
        })
    }

    /// Draw over the image's contents instead of clearing it first; the image must
    /// be in `GENERAL` when the pass starts, as an image source leaves it. Set before compiling.
    pub fn set_load_existing(&self, load: bool) {
        self.load_existing.set(load);
    }

    pub fn add_draw_component(&self, component: Rc<dyn DrawComponent>) {
        self.draw_components.borrow_mut().push(component);
    }

    /// Makes `element` run before this pass: graph ordering puts it earlier in the
    /// command buffer, e.g. a sort and cull feeding an indirect draw (through a
    /// [`BufferToGraphicsBarrier`](crate::computegraph::BufferToGraphicsBarrier)).
    pub fn add_compute_dependency(&self, element: ElementPtr) {
        self.compute_dependencies.borrow_mut().push(element);
    }

    /// The camera uniform buffer among the inputs.
    pub fn camera_ubo(&self) -> Result<Rc<CameraUbo>> {
        self.base
            .inputs()
            .values()
            .find_map(downcast::<CameraUbo>)
            .ok_or_else(|| anyhow::anyhow!("No CameraUbo found in inputs!"))
    }

    fn target(&self) -> ElementPtr {
        self.input_element(0).expect("RenderPass: no input!")
    }
}

impl ComputeGraphElement for RenderPass {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "RenderPass"
    }

    fn check_input(&self, input: &ElementPtr, index: i32) -> Result<()> {
        match index {
            0 if input.as_image_view_src().is_none() => bail!("input is not an ImageViewSrc!"),
            1 if downcast::<CameraUbo>(input).is_none() => bail!("input is not a CameraUbo!"),
            0 | 1 => Ok(()),
            _ => bail!("input index out of range!"),
        }
    }

    fn inputs(&self) -> BTreeMap<i32, ElementPtr> {
        let mut all = self.base.inputs();
        let mut index = all.len() as i32;
        for dependency in self.compute_dependencies.borrow().iter() {
            all.insert(index, dependency.clone());
            index += 1;
        }
        all
    }

    fn setup(&self, context: &VulkanContext, number_paths: u32) -> Result<()> {
        let device = context.vk();
        let load_existing = self.load_existing.get();
        let default_final = if context.has_surface() {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::GENERAL
        };
        let final_layout = self.final_layout_override().unwrap_or(default_final);

        // Drawing over the image needs its contents, which its source left in GENERAL.
        let attachment = [vk::AttachmentDescription::default()
            .format(self.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(if load_existing {
                vk::AttachmentLoadOp::LOAD
            } else {
                vk::AttachmentLoadOp::CLEAR
            })
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(if load_existing {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            })
            .final_layout(final_layout)];
        let color_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let subpass = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref)];
        // Loading the contents reads the attachment after its layout transition, so the
        // transition must be visible to reads, too.
        let dependency = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(if load_existing {
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            })];
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment)
            .subpasses(&subpass)
            .dependencies(&dependency);
        let mut resources = RenderPassResources {
            device: context.device().clone(),
            render_pass: unsafe { device.create_render_pass(&info, None) }?,
            framebuffers: Vec::new(),
        };

        for path_id in 0..number_paths {
            let views = [self.image_view(path_id)];
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(resources.render_pass)
                .attachments(&views)
                .width(self.extent.width)
                .height(self.extent.height)
                .layers(1);
            resources
                .framebuffers
                .push(unsafe { device.create_framebuffer(&info, None) }?);
        }

        let camera_ubo = self.camera_ubo()?;
        for component in self.draw_components.borrow().iter() {
            component.initialize(context, resources.render_pass, camera_ubo.clone(), number_paths)?;
        }
        *self.resources.borrow_mut() = Some(resources);
        self.base.set_initialized(true);
        Ok(())
    }

    fn record(&self, cmd: vk::CommandBuffer, path_id: u32) -> Result<()> {
        let resources = self.resources.borrow();
        let Some(resources) = resources.as_ref() else {
            bail!("RenderPass not initialized")
        };
        let device = &resources.device.device;
        let load_existing = self.load_existing.get();

        // Keep the contents when drawing over them; else they are discarded.
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(if load_existing {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            })
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image(path_id))
            .subresource_range(color_subresource_range())
            .src_access_mask(if load_existing {
                vk::AccessFlags::MEMORY_WRITE
            } else {
                vk::AccessFlags::SHADER_WRITE
            })
            .dst_access_mask(if load_existing {
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags::MEMORY_READ
            });
        let (src_stage, dst_stage) = if load_existing {
            (
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            )
        } else {
            (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            )
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

        let clear = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let framebuffer = resources.framebuffers[path_id as usize];
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(resources.render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .clear_values(&clear);
        unsafe { device.cmd_begin_render_pass(cmd, &begin, vk::SubpassContents::INLINE) };
        for component in self.draw_components.borrow().iter() {
            component.record_command_buffer(cmd, framebuffer, path_id)?;
        }
        unsafe { device.cmd_end_render_pass(cmd) };
        Ok(())
    }

    fn as_image_view_src(&self) -> Option<&dyn ImageViewSrcInterface> {
        Some(self)
    }
}

impl ImageViewSrcInterface for RenderPass {
    fn image_view(&self, path_id: u32) -> vk::ImageView {
        self.target()
            .as_image_view_src()
            .expect("input is not an ImageViewSrc!")
            .image_view(path_id)
    }

    fn image(&self, path_id: u32) -> vk::Image {
        self.target()
            .as_image_view_src()
            .expect("input is not an ImageViewSrc!")
            .image(path_id)
    }

    fn image_extent(&self, path_id: u32) -> vk::Extent2D {
        self.target()
            .as_image_view_src()
            .expect("input is not an ImageViewSrc!")
            .image_extent(path_id)
    }

    fn final_layout_override(&self) -> Option<vk::ImageLayout> {
        self.base
            .inputs()
            .get(&0)
            .and_then(|_| self.target().as_image_view_src()?.final_layout_override())
    }
}
