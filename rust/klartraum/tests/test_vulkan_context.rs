// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - headless_context_has_two_images: a headless context creates two 512x384 BGRA images with views
 * - shaders_are_embedded: the raster shaders are found through read_file without files on disk
 * - host_visible_buffer_round_trip: data copied into a host-coherent buffer reads back unchanged
 * - device_local_buffer_round_trip: likewise for a device-local buffer, through staging copies
 * - device_local_buffer_zero_and_copy: zero() clears a device-local buffer and copy_from() copies another one
 * - frames_cycle_through_images: begin/end of four headless frames uses images 0, 1, 0, 1 without
 *   validation errors
 **/
mod common;

use ash::vk;
use klartraum::vulkan_buffer::{DEFAULT_BUFFER_USAGE, VulkanBuffer};
use klartraum::vulkan_context::{VulkanContext, validation_error_count};
use klartraum::vulkan_helpers::read_file;

#[test]
fn headless_context_has_two_images() {
    let _gpu = common::gpu_lock();
    let vc = VulkanContext::new_headless().unwrap();
    assert!(!vc.has_surface());
    assert_eq!(vc.number_of_swap_chain_images(), 2);
    assert_eq!(
        vc.swap_chain_extent(),
        vk::Extent2D {
            width: 512,
            height: 384
        }
    );
    assert_eq!(vc.swap_chain_image_format(), vk::Format::B8G8R8A8_UNORM);
    assert_ne!(vc.image_view(1), vk::ImageView::null());
}

#[test]
fn shaders_are_embedded() {
    for name in [
        "shaders/gsplat/gsplat_raster.vert.spv",
        "shaders/gsplat/gsplat_raster_project.comp.spv",
    ] {
        let code = read_file(name).unwrap();
        assert_eq!(&code[..4], &0x0723_0203u32.to_le_bytes(), "{name} is not SPIR-V");
    }
}

#[test]
fn host_visible_buffer_round_trip() {
    let _gpu = common::gpu_lock();
    let vc = VulkanContext::new_headless().unwrap();
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
    let buffer = VulkanBuffer::<f32>::new(vc.device(), 1000).unwrap();
    buffer.memcopy_from(&data).unwrap();
    assert_eq!(buffer.read().unwrap(), data);
}

#[test]
fn device_local_buffer_round_trip() {
    let _gpu = common::gpu_lock();
    let vc = VulkanContext::new_headless().unwrap();
    let data: Vec<u32> = (0..4096).collect();
    let buffer = VulkanBuffer::<u32>::with_usage(
        vc.device(),
        4096,
        DEFAULT_BUFFER_USAGE,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .unwrap();
    buffer.memcopy_from(&data).unwrap();
    assert_eq!(buffer.read().unwrap(), data);
}

#[test]
fn device_local_buffer_zero_and_copy() {
    let _gpu = common::gpu_lock();
    let vc = VulkanContext::new_headless().unwrap();
    let local = |n| {
        VulkanBuffer::<u32>::with_usage(
            vc.device(),
            n,
            DEFAULT_BUFFER_USAGE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap()
    };
    let a = local(64);
    a.memcopy_from(&[7; 64]).unwrap();
    a.zero().unwrap();
    assert_eq!(a.read().unwrap(), vec![0; 64]);

    let b = local(32);
    b.memcopy_from(&(0..32).collect::<Vec<_>>()).unwrap();
    a.copy_from(b.buffer(), 32).unwrap();
    let read = a.read().unwrap();
    assert_eq!(&read[..32], &(0..32).collect::<Vec<_>>()[..]);
    assert_eq!(&read[32..], &[0; 32]);
}

#[test]
fn frames_cycle_through_images() {
    let _gpu = common::gpu_lock();
    let errors = validation_error_count();
    let mut vc = VulkanContext::new_headless().unwrap();
    let mut images = Vec::new();
    for _ in 0..4 {
        let (index, fence) = vc.begin_render().unwrap();
        images.push(index);
        // A frame without graphs: the image-available semaphore is handed straight to
        // end_render, and a fence-only submit stands for the frame's last submission.
        let wait = [vc.image_available_semaphores_per_image()[index as usize]];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages);
        vc.device().queue_submit(vc.graphics_queue(), &[submit], fence).unwrap();
        let semaphore = unsafe { vc.vk().create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.unwrap();
        let signal = [semaphore];
        let signal_submit = vk::SubmitInfo::default().signal_semaphores(&signal);
        vc.device()
            .queue_submit(vc.graphics_queue(), &[signal_submit], vk::Fence::null())
            .unwrap();
        vc.end_render(index, semaphore).unwrap();
        vc.device().queue_wait_idle(vc.graphics_queue()).unwrap();
        unsafe { vc.vk().destroy_semaphore(semaphore, None) };
    }
    assert_eq!(images, vec![0, 1, 0, 1]);
    assert_eq!(validation_error_count(), errors);
}
