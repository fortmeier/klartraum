// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`GaussianSplatRasterizer`], the draw component of the raster Gaussian splatting backend.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use anyhow::{Context, Result, bail};
use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::computegraph::{CameraUbo, ElementPtr};
use crate::draw_component::DrawComponent;
use crate::vulkan_context::{Device, VulkanContext};
use crate::vulkan_helpers::load_shader_module;

/// Push constants of `gsplat_raster.vert` / `.frag` (scalar layout, 28 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GaussianSplatRasterPushConstants {
    pub resolution: [f32; 2],
    pub focal: [f32; 2],
    pub splat_scale: f32,
    pub sh_degree: i32,
    pub num_splats: u32,
}

/// The Vulkan objects of an initialized [`GaussianSplatRasterizer`].
struct RasterizerResources {
    device: Rc<Device>,
    camera_ubo: Rc<CameraUbo>,
    splat_descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    /// Set 1 per path: the splat buffers.
    splat_descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
}

impl Drop for RasterizerResources {
    fn drop(&mut self) {
        let device = &self.device.device;
        unsafe {
            device.destroy_pipeline(self.graphics_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.splat_descriptor_set_layout, None);
        }
    }
}

/// Draws the sorted splats as instanced quads (vertex and fragment shader path).
///
/// It binds set 0, the camera uniform buffer, and set 1, the splat storage buffers
/// in the order given (binding = index); generates the quad corners from
/// `gl_VertexIndex` (no vertex buffer); and issues one `vkCmdDrawIndirect` that
/// reads the instance count from the draw-arguments buffer, which the compute
/// culling fills (ordered before the render pass by a `BufferToGraphicsBarrier`).
/// Splats are blended with premultiplied "over", so they composite back to front.
pub struct GaussianSplatRasterizer {
    /// Per-splat storage buffers followed by the sorted-index buffer, bound as set 1.
    splat_buffers: Vec<ElementPtr>,
    /// `VkDrawIndirectCommand` written by the compute stage.
    draw_args_buffer: ElementPtr,
    push_constants: Cell<GaussianSplatRasterPushConstants>,
    resources: RefCell<Option<RasterizerResources>>,
}

impl GaussianSplatRasterizer {
    pub fn new(splat_buffers: Vec<ElementPtr>, draw_args_buffer: ElementPtr) -> Rc<Self> {
        Rc::new(Self {
            splat_buffers,
            draw_args_buffer,
            push_constants: Cell::new(GaussianSplatRasterPushConstants::default()),
            resources: RefCell::new(None),
        })
    }

    pub fn set_push_constants(&self, push_constants: GaussianSplatRasterPushConstants) {
        self.push_constants.set(push_constants);
    }

    pub fn push_constants(&self) -> GaussianSplatRasterPushConstants {
        self.push_constants.get()
    }

    fn create(
        &self,
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        camera_ubo: Rc<CameraUbo>,
        number_paths: u32,
    ) -> Result<RasterizerResources> {
        let device = context.vk();
        let mut r = RasterizerResources {
            device: context.device().clone(),
            camera_ubo,
            splat_descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            splat_descriptor_sets: Vec::new(),
            pipeline_layout: vk::PipelineLayout::null(),
            graphics_pipeline: vk::Pipeline::null(),
        };

        let bindings: Vec<_> = (0..self.splat_buffers.len())
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
            })
            .collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        r.splat_descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
            .context("failed to create splat descriptor set layout!")?;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: number_paths * self.splat_buffers.len() as u32,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(number_paths);
        r.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .context("failed to create splat descriptor pool!")?;
        let layouts = vec![r.splat_descriptor_set_layout; number_paths as usize];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(r.descriptor_pool)
            .set_layouts(&layouts);
        r.splat_descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&alloc) }.context("failed to allocate splat descriptor sets!")?;
        for path_id in 0..number_paths {
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = self
                .splat_buffers
                .iter()
                .map(|b| {
                    let b = b.as_buffer().expect("splat buffers are buffer elements");
                    [vk::DescriptorBufferInfo {
                        buffer: b.vk_buffer(path_id),
                        offset: 0,
                        range: b.buffer_mem_size(),
                    }]
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(r.splat_descriptor_sets[path_id as usize])
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        let vert = load_shader_module(device, "shaders/gsplat/gsplat_raster.vert.spv")?;
        let frag = match load_shader_module(device, "shaders/gsplat/gsplat_raster.frag.spv") {
            Ok(frag) => frag,
            Err(e) => {
                unsafe { device.destroy_shader_module(vert, None) };
                return Err(e);
            }
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(c"main"),
        ];
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        // No vertex buffer: the quad corners come from gl_VertexIndex.
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
        let extent = context.swap_chain_extent();
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        }];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0);
        // Premultiplied "over": the fragment shader outputs rgb already multiplied by alpha.
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op(vk::LogicOp::COPY)
            .attachments(&blend_attachments);

        let push_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<GaussianSplatRasterPushConstants>() as u32,
        }];
        let set_layouts = [r.camera_ubo.descriptor_set_layout(), r.splat_descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let created = unsafe { device.create_pipeline_layout(&layout_info, None) };
        let pipeline = created.map_err(anyhow::Error::from).and_then(|layout| {
            r.pipeline_layout = layout;
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization)
                .multisample_state(&multisample)
                .color_blend_state(&color_blend)
                .dynamic_state(&dynamic_state)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0);
            unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
                .map(|p| p[0])
                .map_err(|(_, e)| anyhow::anyhow!("failed to create graphics pipeline: {e}"))
        });
        unsafe {
            device.destroy_shader_module(frag, None);
            device.destroy_shader_module(vert, None);
        }
        r.graphics_pipeline = pipeline?;
        Ok(r)
    }
}

impl DrawComponent for GaussianSplatRasterizer {
    fn initialize(
        &self,
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        camera_ubo: Rc<CameraUbo>,
        number_paths: u32,
    ) -> Result<()> {
        let resources = self.create(context, render_pass, camera_ubo, number_paths)?;
        *self.resources.borrow_mut() = Some(resources);
        Ok(())
    }

    fn record_command_buffer(&self, cmd: vk::CommandBuffer, _framebuffer: vk::Framebuffer, path_id: u32) -> Result<()> {
        let resources = self.resources.borrow();
        let Some(r) = resources.as_ref() else {
            bail!("GaussianSplatRasterizer not initialized")
        };
        let device = &r.device.device;
        // The viewport is the render target's extent, carried in the push constants'
        // resolution; it differs from the swapchain's for an offscreen viewport.
        let pc = self.push_constants.get();
        let extent = vk::Extent2D {
            width: pc.resolution[0] as u32,
            height: pc.resolution[1] as u32,
        };
        let camera_sets = r.camera_ubo.descriptor_sets();
        let sets = [camera_sets[path_id as usize], r.splat_descriptor_sets[path_id as usize]];
        let draw_args = self
            .draw_args_buffer
            .as_buffer()
            .expect("draw arguments are a buffer")
            .vk_buffer(path_id);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, r.graphics_pipeline);
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, r.pipeline_layout, 0, &sets, &[]);
            device.cmd_push_constants(
                cmd,
                r.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&pc),
            );
            device.cmd_draw_indirect(
                cmd,
                draw_args,
                0,
                1,
                std::mem::size_of::<vk::DrawIndirectCommand>() as u32,
            );
        }
        Ok(())
    }
}
