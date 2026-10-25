// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`VulkanBuffer`], a typed Vulkan buffer with its own memory.

use std::marker::PhantomData;
use std::rc::Rc;

use anyhow::{Result, ensure};
use ash::vk;
use bytemuck::Pod;

use crate::vulkan_context::Device;

/// Storage buffer and transfer destination: the default usage of a [`VulkanBuffer`].
pub const DEFAULT_BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw() | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// Host-visible, host-coherent: the default memory of a [`VulkanBuffer`].
pub const HOST_COHERENT: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);

/// A Vulkan buffer of `size` elements of type `T`.
///
/// Host-visible memory is mapped for copies; other memory (e.g. device-local) is
/// written and read through a temporary staging buffer and an immediate submission,
/// which is meant for initialization and explicit readback only. Host-visible
/// memory must be coherent: mapped copies do not flush or invalidate.
pub struct VulkanBuffer<T: Pod> {
    device: Rc<Device>,
    /// Number of elements.
    size: u32,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Whether copies can map the memory directly or need a staging buffer.
    host_visible: bool,
    /// Properties of the memory type actually backing the buffer.
    memory_properties: vk::MemoryPropertyFlags,
    _marker: PhantomData<T>,
}

impl<T: Pod> VulkanBuffer<T> {
    /// A host-coherent storage buffer of `size` elements.
    pub fn new(device: &Rc<Device>, size: u32) -> Result<Self> {
        Self::with_usage(device, size, DEFAULT_BUFFER_USAGE, HOST_COHERENT)
    }

    /// A buffer of `size` elements with the given usage and memory properties.
    ///
    /// Every buffer can also be a transfer source and destination, for
    /// initialization, GPU-to-GPU copies and readback.
    pub fn with_usage(
        device: &Rc<Device>,
        size: u32,
        usage: vk::BufferUsageFlags,
        memory: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        ensure!(size > 0, "Buffer size must be greater than 0");
        let bytes = std::mem::size_of::<T>() as vk::DeviceSize * size as vk::DeviceSize;
        let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.device.create_buffer(&info, None) }?;
        let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
        let type_index = match device.find_memory_type(requirements.memory_type_bits, memory) {
            Ok(index) => index,
            Err(e) => {
                unsafe { device.device.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(type_index);
        let device_memory = match unsafe { device.device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.device.destroy_buffer(buffer, None) };
                return Err(anyhow::anyhow!("failed to allocate buffer memory: {e}"));
            }
        };
        unsafe { device.device.bind_buffer_memory(buffer, device_memory, 0) }?;
        Ok(Self {
            device: device.clone(),
            size,
            buffer,
            memory: device_memory,
            host_visible: memory.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
            memory_properties: device.memory_properties.memory_types[type_index as usize].property_flags,
            _marker: PhantomData,
        })
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Number of elements.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Size in bytes.
    pub fn buffer_mem_size(&self) -> vk::DeviceSize {
        std::mem::size_of::<T>() as vk::DeviceSize * self.size as vk::DeviceSize
    }

    /// Properties of the memory type backing the buffer.
    pub fn memory_properties(&self) -> vk::MemoryPropertyFlags {
        self.memory_properties
    }

    /// Runs a transfer and makes its writes visible to later GPU work and host reads.
    fn commands(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let device = &self.device.device;
        self.device.submit_immediate(
            |cmd| {
                record(cmd);
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ,
                    );
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    )
                };
            },
            vk::Queue::null(),
        )
    }

    /// Copies `src` (at most `size` elements) to the start of the buffer.
    ///
    /// Earlier GPU use of the buffer must have finished.
    pub fn memcopy_from(&self, src: &[T]) -> Result<()> {
        self.memcopy_from_bytes(bytemuck::cast_slice(src))
    }

    /// Copies raw bytes (at most the buffer's byte size) to the start of the buffer.
    pub fn memcopy_from_bytes(&self, src: &[u8]) -> Result<()> {
        let n = src.len().min(self.buffer_mem_size() as usize);
        if n == 0 {
            return Ok(());
        }
        if !self.host_visible {
            let staging = VulkanBuffer::<u8>::new(&self.device, n as u32)?;
            staging.memcopy_from_bytes(&src[..n])?;
            return self.commands(|cmd| unsafe {
                self.device.device.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    self.buffer,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: n as vk::DeviceSize,
                    }],
                )
            });
        }
        unsafe {
            let mapped =
                self.device
                    .device
                    .map_memory(self.memory, 0, self.buffer_mem_size(), vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(src.as_ptr(), mapped as *mut u8, n);
            self.device.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    /// Copies the start of the buffer into `dst` (at most `size` elements); the
    /// GPU work writing it must have finished.
    pub fn memcopy_to(&self, dst: &mut [T]) -> Result<()> {
        let n = dst.len().min(self.size as usize) * std::mem::size_of::<T>();
        if n == 0 {
            return Ok(());
        }
        if !self.host_visible {
            let staging = VulkanBuffer::<u8>::new(&self.device, n as u32)?;
            self.commands(|cmd| unsafe {
                self.device.device.cmd_copy_buffer(
                    cmd,
                    self.buffer,
                    staging.buffer,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: n as vk::DeviceSize,
                    }],
                )
            })?;
            return staging.memcopy_to(&mut bytemuck::cast_slice_mut::<T, u8>(dst)[..n]);
        }
        unsafe {
            let mapped =
                self.device
                    .device
                    .map_memory(self.memory, 0, self.buffer_mem_size(), vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(
                mapped as *const u8,
                bytemuck::cast_slice_mut::<T, u8>(dst).as_mut_ptr(),
                n,
            );
            self.device.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    /// Reads the whole buffer.
    pub fn read(&self) -> Result<Vec<T>> {
        let mut out = vec![T::zeroed(); self.size as usize];
        self.memcopy_to(&mut out)?;
        Ok(out)
    }

    /// Sets the whole buffer to zero now.
    pub fn zero(&self) -> Result<()> {
        if !self.host_visible {
            return self.commands(|cmd| unsafe {
                self.device
                    .device
                    .cmd_fill_buffer(cmd, self.buffer, 0, vk::WHOLE_SIZE, 0)
            });
        }
        self.memcopy_from_bytes(&vec![0u8; self.buffer_mem_size() as usize])
    }

    /// Records zeroing the whole buffer.
    pub fn record_zero(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device
                .device
                .cmd_fill_buffer(cmd, self.buffer, 0, self.buffer_mem_size(), 0)
        };
    }

    /// Records filling the whole buffer with a repeating 32-bit pattern.
    pub fn record_fill(&self, cmd: vk::CommandBuffer, value: u32) {
        unsafe {
            self.device
                .device
                .cmd_fill_buffer(cmd, self.buffer, 0, self.buffer_mem_size(), value)
        };
    }

    /// Records zeroing the bytes `[byte_offset, byte_offset + byte_size)`, e.g. one field of a struct buffer.
    pub fn record_zero_range(&self, cmd: vk::CommandBuffer, byte_offset: vk::DeviceSize, byte_size: vk::DeviceSize) {
        unsafe {
            self.device
                .device
                .cmd_fill_buffer(cmd, self.buffer, byte_offset, byte_size, 0)
        };
    }

    /// Copies the first `count` elements of `source` (bounded by this buffer's size) now.
    pub fn copy_from(&self, source: vk::Buffer, count: u32) -> Result<()> {
        let bytes = std::mem::size_of::<T>() as vk::DeviceSize * count.min(self.size) as vk::DeviceSize;
        if bytes == 0 {
            return Ok(());
        }
        self.commands(|cmd| unsafe {
            self.device.device.cmd_copy_buffer(
                cmd,
                source,
                self.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes,
                }],
            )
        })
    }
}

impl<T: Pod> Drop for VulkanBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}
