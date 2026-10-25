// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`HostMemoryBufferElement`], a buffer whose memory is host memory the application provides.

use std::rc::Rc;

use anyhow::{Context, Result, anyhow, ensure};
use ash::vk;

use crate::computegraph::buffer_element::BufferElementInterface;
use crate::computegraph::element::{ComputeGraphElement, ElementBase};
use crate::vulkan_context::{Device, VulkanContext};

/// Host memory that can back a Vulkan buffer, e.g. a shared-memory mapping.
///
/// The memory must stay mapped at the same address for as long as the value lives.
pub trait HostMemory {
    /// The start of the memory.
    fn ptr(&self) -> *mut u8;
    /// The size in bytes.
    fn len(&self) -> usize;
}

/// A buffer shared by all paths whose device memory *is* the given host memory,
/// imported with `VK_EXT_external_memory_host`: shaders read and write the host
/// pages directly, so the host sees results without any copy (after the GPU work
/// finished), and data written by the host is read by the next submission.
///
/// The pointer and the buffer size must be multiples of the device's import
/// alignment ([`Device::external_memory_host`]).
pub struct HostMemoryBufferElement {
    base: ElementBase,
    device: Rc<Device>,
    /// Keeps the host memory mapped while the buffer uses it.
    memory_owner: Rc<dyn HostMemory>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    byte_size: vk::DeviceSize,
}

impl HostMemoryBufferElement {
    /// Whether the device of `context` can import host memory.
    pub fn is_supported(context: &VulkanContext) -> bool {
        context.device().external_memory_host.is_some()
    }

    /// Imports the first `byte_size` bytes of `memory` as a storage buffer.
    pub fn import(context: &VulkanContext, memory: Rc<dyn HostMemory>, byte_size: usize) -> Result<Rc<Self>> {
        let device = context.device().clone();
        let (loader, alignment) = device
            .external_memory_host
            .as_ref()
            .ok_or_else(|| anyhow!("VK_EXT_external_memory_host is not supported"))?;
        let alignment = *alignment as usize;
        let ptr = memory.ptr();
        ensure!(
            byte_size > 0 && byte_size <= memory.len(),
            "byte size {byte_size} outside the host memory"
        );
        ensure!(
            ptr as usize % alignment == 0,
            "host pointer is not aligned to {alignment} bytes"
        );
        // The imported allocation covers whole alignment units of the host memory.
        let allocation_size = byte_size.div_ceil(alignment) * alignment;
        ensure!(
            allocation_size <= memory.len(),
            "host memory is not padded to the import alignment {alignment}"
        );

        let mut external_info = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = vk::BufferCreateInfo::default()
            .size(byte_size as vk::DeviceSize)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);
        let buffer = unsafe { device.device.create_buffer(&info, None) }.context("failed to create host buffer")?;

        let import = || -> Result<vk::DeviceMemory> {
            let mut pointer_properties = vk::MemoryHostPointerPropertiesEXT::default();
            unsafe {
                (loader.fp().get_memory_host_pointer_properties_ext)(
                    device.device.handle(),
                    vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                    ptr as *const std::ffi::c_void,
                    &mut pointer_properties,
                )
            }
            .result()
            .context("vkGetMemoryHostPointerPropertiesEXT failed")?;
            let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
            let type_bits = pointer_properties.memory_type_bits & requirements.memory_type_bits;
            let type_index = device
                .find_memory_type(
                    type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .or_else(|_| device.find_memory_type(type_bits, vk::MemoryPropertyFlags::empty()))?;
            let mut import_info = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .host_pointer(ptr as *mut std::ffi::c_void);
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_size.max(requirements.size as usize) as vk::DeviceSize)
                .memory_type_index(type_index)
                .push_next(&mut import_info);
            let memory =
                unsafe { device.device.allocate_memory(&allocate, None) }.context("failed to import host memory")?;
            if let Err(e) = unsafe { device.device.bind_buffer_memory(buffer, memory, 0) } {
                unsafe { device.device.free_memory(memory, None) };
                return Err(anyhow!("failed to bind host memory: {e}"));
            }
            Ok(memory)
        };
        match import() {
            Ok(device_memory) => Ok(Rc::new(Self {
                base: ElementBase::new(),
                device,
                memory_owner: memory,
                buffer,
                memory: device_memory,
                byte_size: byte_size as vk::DeviceSize,
            })),
            Err(e) => {
                unsafe { device.device.destroy_buffer(buffer, None) };
                Err(e)
            }
        }
    }

    /// The host memory the buffer uses.
    pub fn host_memory(&self) -> &Rc<dyn HostMemory> {
        &self.memory_owner
    }
}

impl Drop for HostMemoryBufferElement {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

impl ComputeGraphElement for HostMemoryBufferElement {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "HostMemoryBufferElement"
    }

    fn setup(&self, _context: &VulkanContext, _number_paths: u32) -> Result<()> {
        self.base.set_initialized(true);
        Ok(())
    }

    fn as_buffer(&self) -> Option<&dyn BufferElementInterface> {
        Some(self)
    }
}

impl BufferElementInterface for HostMemoryBufferElement {
    fn buffer_mem_size(&self) -> vk::DeviceSize {
        self.byte_size
    }

    fn vk_buffer(&self, _path_id: u32) -> vk::Buffer {
        self.buffer
    }
}
