// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Uniform buffers as graph elements, with their own descriptor sets.

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use anyhow::{Result, bail};
use ash::vk;
use bytemuck::Pod;

use crate::camera::CameraMVP;
use crate::computegraph::element::{ComputeGraphElement, ElementBase};
use crate::vulkan_buffer::HOST_COHERENT;
use crate::vulkan_context::{Device, VulkanContext};

/// Untyped access to a uniform buffer element's per-path buffers.
pub trait UniformBufferObjectInterface {
    fn buffer_mem_size(&self) -> vk::DeviceSize;
    fn vk_buffer(&self, path_id: u32) -> vk::Buffer;
}

/// The camera uniform buffer the renderers read.
pub type CameraUbo = UniformBufferObject<CameraMVP>;

/// The Vulkan objects of a set-up [`UniformBufferObject`].
struct UboResources {
    device: Rc<Device>,
    /// Layout of set 0 binding 0, visible to the vertex and compute stages.
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    buffers: Vec<vk::Buffer>,
    memories: Vec<vk::DeviceMemory>,
    /// Persistently mapped memory of each path's buffer.
    mapped: Vec<*mut u8>,
}

impl Drop for UboResources {
    fn drop(&mut self) {
        let device = &self.device.device;
        unsafe {
            for (&buffer, &memory) in self.buffers.iter().zip(&self.memories) {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// A uniform buffer with one host-coherent, mapped buffer and descriptor set per path.
///
/// The CPU edits [`ubo_mut`](Self::ubo_mut) and copies it into a path's buffer with
/// [`update`](Self::update) when that path's previous submission has finished. A UBO
/// may feed several graphs (e.g. one camera for several viewports); the first graph
/// creates the buffers and later ones, which must have as many paths, reuse them.
pub struct UniformBufferObject<U: Pod> {
    base: ElementBase,
    ubo: RefCell<U>,
    resources: RefCell<Option<UboResources>>,
}

impl<U: Pod + Default> UniformBufferObject<U> {
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            base: ElementBase::new(),
            ubo: RefCell::new(U::default()),
            resources: RefCell::new(None),
        })
    }
}

impl<U: Pod> UniformBufferObject<U> {
    /// The CPU-side contents.
    pub fn ubo(&self) -> Ref<'_, U> {
        self.ubo.borrow()
    }

    /// The CPU-side contents, for editing before [`update`](Self::update).
    pub fn ubo_mut(&self) -> RefMut<'_, U> {
        self.ubo.borrow_mut()
    }

    /// Copies the CPU-side contents into the buffer of a path.
    pub fn update(&self, path_id: u32) {
        let resources = self.resources.borrow();
        let resources = resources.as_ref().expect("UniformBufferObject::update before setup");
        let ubo = self.ubo.borrow();
        let bytes = bytemuck::bytes_of(&*ubo);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), resources.mapped[path_id as usize], bytes.len()) };
    }

    /// The layout of the UBO's descriptor set (binding 0); valid after setup.
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.resources
            .borrow()
            .as_ref()
            .map_or(vk::DescriptorSetLayout::null(), |r| r.descriptor_set_layout)
    }

    /// The descriptor set of each path; valid after setup.
    pub fn descriptor_sets(&self) -> Vec<vk::DescriptorSet> {
        self.resources
            .borrow()
            .as_ref()
            .map_or_else(Vec::new, |r| r.descriptor_sets.clone())
    }

    fn create_resources(&self, context: &VulkanContext, number_paths: u32) -> Result<UboResources> {
        let device = context.vk();
        let size = std::mem::size_of::<U>() as vk::DeviceSize;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE)];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding);
        let mut resources = UboResources {
            device: context.device().clone(),
            descriptor_set_layout: unsafe { device.create_descriptor_set_layout(&layout_info, None) }?,
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            buffers: Vec::new(),
            memories: Vec::new(),
            mapped: Vec::new(),
        };

        for _ in 0..number_paths {
            let (buffer, memory) =
                context
                    .device()
                    .create_buffer(size, vk::BufferUsageFlags::UNIFORM_BUFFER, HOST_COHERENT)?;
            resources.buffers.push(buffer);
            resources.memories.push(memory);
            let mapped = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }?;
            resources.mapped.push(mapped as *mut u8);
        }

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: number_paths,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(number_paths);
        resources.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;
        let layouts = vec![resources.descriptor_set_layout; number_paths as usize];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(resources.descriptor_pool)
            .set_layouts(&layouts);
        resources.descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc) }?;

        for (&set, &buffer) in resources.descriptor_sets.iter().zip(&resources.buffers) {
            let info = [vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: size,
            }];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }
        Ok(resources)
    }
}

impl<U: Pod> ComputeGraphElement for UniformBufferObject<U> {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "UniformBufferObject"
    }

    fn setup(&self, context: &VulkanContext, number_paths: u32) -> Result<()> {
        if let Some(resources) = self.resources.borrow().as_ref() {
            if resources.buffers.len() != number_paths as usize {
                bail!("UniformBufferObject shared across graphs with differing path counts");
            }
            return Ok(());
        }
        let resources = self.create_resources(context, number_paths)?;
        *self.resources.borrow_mut() = Some(resources);
        self.base.set_initialized(true);
        Ok(())
    }

    fn as_uniform_buffer(&self) -> Option<&dyn UniformBufferObjectInterface> {
        Some(self)
    }
}

impl<U: Pod> UniformBufferObjectInterface for UniformBufferObject<U> {
    fn buffer_mem_size(&self) -> vk::DeviceSize {
        std::mem::size_of::<U>() as vk::DeviceSize
    }

    fn vk_buffer(&self, path_id: u32) -> vk::Buffer {
        self.resources
            .borrow()
            .as_ref()
            .expect("UniformBufferObject used before setup")
            .buffers[path_id as usize]
    }
}
