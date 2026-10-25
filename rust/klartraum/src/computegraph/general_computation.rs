// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`GeneralComputation`], a compute shader (or a sequence of them) over the element's inputs.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use anyhow::{Result, bail, ensure};
use ash::vk;
use bytemuck::Pod;

use crate::computegraph::element::{ComputeGraphElement, ElementBase, ElementPtr};
use crate::vulkan_context::{Device, VulkanContext, color_subresource_range};
use crate::vulkan_helpers::load_shader_module;

/// A layout transition recorded before the dispatches for an image input.
#[derive(Clone, Copy, Debug)]
pub struct ImageLayoutTransition {
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_stage_mask: vk::PipelineStageFlags,
    pub dst_stage_mask: vk::PipelineStageFlags,
    pub src_access_mask: vk::AccessFlags,
    pub dst_access_mask: vk::AccessFlags,
}

impl Default for ImageLayoutTransition {
    fn default() -> Self {
        Self {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            src_stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
        }
    }
}

/// The Vulkan objects of a set-up [`GeneralComputation`].
struct Pipelines {
    device: Rc<Device>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    /// One pipeline per shader, dispatched in order.
    pipelines: Vec<vk::Pipeline>,
    /// One descriptor set per path.
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        let device = &self.device.device;
        unsafe {
            for &pipeline in &self.pipelines {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

/// Runs one or more compute shaders over the element's inputs.
///
/// Input `i` is bound as binding `i` of set 0 — a storage buffer, a storage image
/// (an image source, in `GENERAL`) or a uniform buffer — followed by the scratch
/// buffers added with [`add_scratch_buffer_element`](Self::add_scratch_buffer_element).
/// The inputs must be numbered 0, 1, 2, ... without gaps.
///
/// With a push-constant type `P` (not `()`), the element records, for each push
/// constant set with [`set_push_constants`](Self::set_push_constants), all shaders
/// in order with a compute-to-compute memory barrier after each dispatch; scratch
/// buffers marked for it are zeroed before each round. With `P = ()` every shader
/// runs once.
pub struct GeneralComputation<P: Pod = ()> {
    base: ElementBase,
    shader_paths: Vec<String>,
    group_count: Cell<[u32; 3]>,
    push_constants: RefCell<Vec<P>>,
    /// Indirect dispatch arguments (`VkDispatchIndirectCommand`), replacing the group count.
    dynamic_group_dispatch_params: RefCell<Option<ElementPtr>>,
    /// Scratch buffers that are not part of the graph, set up by this element, and
    /// whether each is zeroed before every round of dispatches.
    scratch_buffers: RefCell<Vec<(ElementPtr, bool)>>,
    image_layout_transitions: RefCell<BTreeMap<i32, ImageLayoutTransition>>,
    pipelines: RefCell<Option<Pipelines>>,
}

impl<P: Pod> GeneralComputation<P> {
    /// A computation with one shader.
    pub fn new(shader_path: &str) -> Rc<Self> {
        Self::with_shaders(&[shader_path])
    }

    /// A computation running several shaders in sequence with the same bindings.
    pub fn with_shaders(shader_paths: &[&str]) -> Rc<Self> {
        Rc::new(Self {
            base: ElementBase::new(),
            shader_paths: shader_paths.iter().map(|s| s.to_string()).collect(),
            group_count: Cell::new([1, 1, 1]),
            push_constants: RefCell::new(Vec::new()),
            dynamic_group_dispatch_params: RefCell::new(None),
            scratch_buffers: RefCell::new(Vec::new()),
            image_layout_transitions: RefCell::new(BTreeMap::new()),
            pipelines: RefCell::new(None),
        })
    }

    fn has_push_constants() -> bool {
        std::mem::size_of::<P>() > 0
    }

    pub fn set_group_count(&self, x: u32, y: u32, z: u32) {
        self.group_count.set([x, y, z]);
    }

    pub fn set_group_count_x(&self, x: u32) {
        let [_, y, z] = self.group_count.get();
        self.group_count.set([x, y, z]);
    }

    pub fn group_count(&self) -> [u32; 3] {
        self.group_count.get()
    }

    /// Dispatches indirectly with the arguments in `params` instead of the group count.
    pub fn set_dynamic_group_dispatch_params(&self, params: ElementPtr) {
        *self.dynamic_group_dispatch_params.borrow_mut() = Some(params);
    }

    /// One round of dispatches per push constant.
    pub fn set_push_constants(&self, push_constants: Vec<P>) {
        *self.push_constants.borrow_mut() = push_constants;
    }

    /// Records a layout transition of the image at input `index` before the dispatches.
    pub fn set_image_layout_transition(&self, index: i32, transition: ImageLayoutTransition) {
        self.image_layout_transitions.borrow_mut().insert(index, transition);
    }

    /// Adds a buffer bound after the inputs; with `record_set_to_zero` it is zeroed before each round.
    pub fn add_scratch_buffer_element(&self, buffer: ElementPtr, record_set_to_zero: bool) {
        self.scratch_buffers.borrow_mut().push((buffer, record_set_to_zero));
    }

    fn descriptor_type(input: &ElementPtr) -> Result<vk::DescriptorType> {
        if input.as_image_view_src().is_some() {
            Ok(vk::DescriptorType::STORAGE_IMAGE)
        } else if input.as_buffer().is_some() {
            Ok(vk::DescriptorType::STORAGE_BUFFER)
        } else if input.as_uniform_buffer().is_some() {
            Ok(vk::DescriptorType::UNIFORM_BUFFER)
        } else {
            bail!(
                "input type not supported for a descriptor set layout: {}",
                input.type_name()
            )
        }
    }

    fn create_pipelines(&self, context: &VulkanContext, number_paths: u32) -> Result<Pipelines> {
        let device = context.vk();
        let number_inputs = self.base.number_of_inputs();
        let scratch = self.scratch_buffers.borrow();
        let inputs: Vec<ElementPtr> = (0..number_inputs as i32)
            .map(|i| self.input_element(i))
            .collect::<Result<_>>()?;
        let mut types = inputs.iter().map(Self::descriptor_type).collect::<Result<Vec<_>>>()?;
        types.extend(std::iter::repeat_n(vk::DescriptorType::STORAGE_BUFFER, scratch.len()));

        let mut resources = Pipelines {
            device: context.device().clone(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipelines: Vec::new(),
            descriptor_sets: Vec::new(),
        };

        let pool_sizes: Vec<_> = types
            .iter()
            .map(|&ty| vk::DescriptorPoolSize {
                ty,
                descriptor_count: number_paths,
            })
            .collect();
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(number_paths);
        resources.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;

        let bindings: Vec<_> = types
            .iter()
            .enumerate()
            .map(|(i, &ty)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(ty)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        resources.descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;

        let set_layouts = [resources.descriptor_set_layout];
        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<P>() as u32,
        }];
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if Self::has_push_constants() {
            layout_info = layout_info.push_constant_ranges(&ranges);
        }
        resources.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;

        for path in &self.shader_paths {
            let module = load_shader_module(device, path)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default()
                .layout(resources.pipeline_layout)
                .stage(stage);
            let created = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
            unsafe { device.destroy_shader_module(module, None) };
            let pipeline =
                created.map_err(|(_, e)| anyhow::anyhow!("failed to create compute pipeline {path}: {e}"))?;
            resources.pipelines.push(pipeline[0]);
        }

        let layouts = vec![resources.descriptor_set_layout; number_paths as usize];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(resources.descriptor_pool)
            .set_layouts(&layouts);
        resources.descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc) }?;

        for path_id in 0..number_paths {
            let set = resources.descriptor_sets[path_id as usize];
            // The infos must outlive update_descriptor_sets; one per binding.
            let mut buffer_infos = Vec::with_capacity(types.len());
            let mut image_infos = Vec::with_capacity(types.len());
            let mut kinds = Vec::with_capacity(types.len());
            for input in &inputs {
                if let Some(image) = input.as_image_view_src() {
                    image_infos.push([vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: image.image_view(path_id),
                        image_layout: vk::ImageLayout::GENERAL,
                    }]);
                    kinds.push((vk::DescriptorType::STORAGE_IMAGE, image_infos.len() - 1));
                } else if let Some(buffer) = input.as_buffer() {
                    buffer_infos.push([vk::DescriptorBufferInfo {
                        buffer: buffer.vk_buffer(path_id),
                        offset: 0,
                        range: buffer.buffer_mem_size(),
                    }]);
                    kinds.push((vk::DescriptorType::STORAGE_BUFFER, buffer_infos.len() - 1));
                } else if let Some(ubo) = input.as_uniform_buffer() {
                    buffer_infos.push([vk::DescriptorBufferInfo {
                        buffer: ubo.vk_buffer(path_id),
                        offset: 0,
                        range: ubo.buffer_mem_size(),
                    }]);
                    kinds.push((vk::DescriptorType::UNIFORM_BUFFER, buffer_infos.len() - 1));
                }
            }
            for (buffer, _) in scratch.iter() {
                let buffer = buffer.as_buffer().expect("scratch elements are buffers");
                buffer_infos.push([vk::DescriptorBufferInfo {
                    buffer: buffer.vk_buffer(path_id),
                    offset: 0,
                    range: buffer.buffer_mem_size(),
                }]);
                kinds.push((vk::DescriptorType::STORAGE_BUFFER, buffer_infos.len() - 1));
            }
            let writes: Vec<_> = kinds
                .iter()
                .enumerate()
                .map(|(binding, &(ty, k))| {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(ty);
                    if ty == vk::DescriptorType::STORAGE_IMAGE {
                        write.image_info(&image_infos[k])
                    } else {
                        write.buffer_info(&buffer_infos[k])
                    }
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        Ok(resources)
    }

    fn record_scratch_to_zero(&self, device: &ash::Device, cmd: vk::CommandBuffer, path_id: u32) {
        for (buffer, zero) in self.scratch_buffers.borrow().iter() {
            if *zero {
                let buffer = buffer.as_buffer().unwrap();
                unsafe { device.cmd_fill_buffer(cmd, buffer.vk_buffer(path_id), 0, buffer.buffer_mem_size(), 0) };
            }
        }
    }

    fn dispatch(&self, device: &ash::Device, cmd: vk::CommandBuffer, path_id: u32) {
        match self.dynamic_group_dispatch_params.borrow().as_ref() {
            None => {
                let [x, y, z] = self.group_count.get();
                unsafe { device.cmd_dispatch(cmd, x, y, z) };
            }
            Some(params) => {
                let buffer = params
                    .as_buffer()
                    .expect("dispatch params are a buffer")
                    .vk_buffer(path_id);
                unsafe { device.cmd_dispatch_indirect(cmd, buffer, 0) };
            }
        }
    }
}

impl<P: Pod> ComputeGraphElement for GeneralComputation<P> {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "GeneralComputation"
    }

    fn check_input(&self, input: &ElementPtr, _index: i32) -> Result<()> {
        ensure!(
            input.as_buffer().is_some() || input.as_image_view_src().is_some() || input.as_uniform_buffer().is_some(),
            "input is not a buffer, an image source or a uniform buffer: {}",
            input.type_name()
        );
        Ok(())
    }

    fn setup(&self, context: &VulkanContext, number_paths: u32) -> Result<()> {
        // The scratch buffers are not part of the graph, so they are set up here.
        for (buffer, _) in self.scratch_buffers.borrow().iter() {
            buffer.setup(context, number_paths)?;
        }
        ensure!(
            self.base.number_of_inputs() > 0 || !self.scratch_buffers.borrow().is_empty(),
            "GeneralComputation: no inputs provided!"
        );
        let pipelines = self.create_pipelines(context, number_paths)?;
        *self.pipelines.borrow_mut() = Some(pipelines);
        self.base.set_initialized(true);
        Ok(())
    }

    fn record(&self, cmd: vk::CommandBuffer, path_id: u32) -> Result<()> {
        let resources = self.pipelines.borrow();
        let Some(resources) = resources.as_ref() else {
            bail!("GeneralComputation not initialized")
        };
        let device = &resources.device.device;

        // Image inputs get a layout barrier (by default UNDEFINED -> GENERAL).
        for i in 0..self.base.number_of_inputs() as i32 {
            let input = self.input_element(i)?;
            if let Some(image) = input.as_image_view_src() {
                let t = self
                    .image_layout_transitions
                    .borrow()
                    .get(&i)
                    .copied()
                    .unwrap_or_default();
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(t.old_layout)
                    .new_layout(t.new_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.image(path_id))
                    .subresource_range(color_subresource_range())
                    .src_access_mask(t.src_access_mask)
                    .dst_access_mask(t.dst_access_mask);
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        t.src_stage_mask,
                        t.dst_stage_mask,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };
            }
        }

        let bind = |pipeline: vk::Pipeline| unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                resources.pipeline_layout,
                0,
                &[resources.descriptor_sets[path_id as usize]],
                &[],
            );
        };

        if !Self::has_push_constants() {
            self.record_scratch_to_zero(device, cmd, path_id);
            for &pipeline in &resources.pipelines {
                bind(pipeline);
                self.dispatch(device, cmd, path_id);
            }
            return Ok(());
        }

        let push_constants = self.push_constants.borrow();
        ensure!(!push_constants.is_empty(), "push constants are empty!");
        for push_constant in push_constants.iter() {
            self.record_scratch_to_zero(device, cmd, path_id);
            for &pipeline in &resources.pipelines {
                bind(pipeline);
                unsafe {
                    device.cmd_push_constants(
                        cmd,
                        resources.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(push_constant),
                    )
                };
                self.dispatch(device, cmd, path_id);
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    )
                };
            }
        }
        Ok(())
    }
}
