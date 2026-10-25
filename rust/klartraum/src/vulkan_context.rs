// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! The Vulkan instance, device, swapchain (or headless images) and frame synchronization.
//!
//! [`VulkanInstance`] and [`Device`] are shared (`Rc`) by everything that owns
//! Vulkan objects; each destroys its handle when the last owner is gone, so
//! buffers and pipelines are always destroyed before the device, and the device
//! before the instance, whatever order the owners are dropped in.
//! [`VulkanContext`] owns the swapchain and the per-frame synchronization; it does
//! not create render passes, pipelines or framebuffers — that is done by the
//! compute graph elements.

use std::ffi::{CStr, CString, c_char, c_void};
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

/// Engine-wide constants, as the C++ `BackendConfig`.
pub struct BackendConfig;

impl BackendConfig {
    /// Frames the CPU may record ahead of the GPU.
    pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
    /// Window width, and the width of the headless images.
    pub const WIDTH: u32 = 512;
    /// Window height, and the height of the headless images.
    pub const HEIGHT: u32 = 384;
    pub const ENGINE_VERSION: &'static str = "Klartraum Engine v0.0.1";
}

static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);

/// The number of validation-layer errors reported so far in this process.
///
/// The C++ engine throws from the debug callback; a Rust panic cannot unwind out
/// of a callback called by C code, so errors are counted instead and the tests
/// check that the count does not change.
pub fn validation_error_count() -> usize {
    VALIDATION_ERRORS.load(Ordering::SeqCst)
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        if data.is_null() || (*data).p_message.is_null() {
            String::new()
        } else {
            CStr::from_ptr((*data).p_message).to_string_lossy().into_owned()
        }
    };
    // Verbose and info messages are the loader's chatter, except shader debugPrintf output.
    let printf = message.contains("DEBUG-PRINTF") || message.contains("Debug Printf");
    if severity
        .intersects(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        || printf
    {
        eprintln!("validation layer: {message}\n");
    }
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::SeqCst);
    }
    vk::FALSE
}

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Whether validation is requested: in debug builds (as the C++ engine without
/// `NDEBUG`), unless `KLARTRAUM_VALIDATION` says otherwise (`0` or `1`).
fn validation_requested() -> bool {
    match std::env::var("KLARTRAUM_VALIDATION").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => cfg!(debug_assertions),
    }
}

/// The Vulkan loader and instance, with the debug messenger when validation is on.
pub struct VulkanInstance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_loader: ash::khr::surface::Instance,
    debug_utils: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    validation: bool,
}

impl VulkanInstance {
    /// Creates the instance with the extensions a surface for `display` needs, if given.
    pub fn new(display: Option<RawDisplayHandle>) -> Result<Rc<Self>> {
        let entry = unsafe { ash::Entry::load() }.context("failed to load the Vulkan loader")?;

        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?;
        let mut validation = validation_requested();
        if validation
            && !available_layers
                .iter()
                .any(|l| l.layer_name_as_c_str().ok() == Some(VALIDATION_LAYER))
        {
            eprintln!("validation layers requested, but not available; continuing without");
            validation = false;
        }
        println!("use validation layers: {validation}");

        let mut extensions: Vec<*const c_char> = Vec::new();
        if let Some(display) = display {
            extensions.extend_from_slice(ash_window::enumerate_required_extensions(display)?);
        }
        if validation {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        // Portability drivers (e.g. MoltenVK) are only enumerated when the instance
        // opts in; conformant drivers are listed either way.
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }?;
        let mut flags = vk::InstanceCreateFlags::empty();
        if available_extensions
            .iter()
            .any(|e| e.extension_name_as_c_str().ok() == Some(ash::khr::portability_enumeration::NAME))
        {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Hello Dreamer")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"Klartraum Engine")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);
        let layers = [VALIDATION_LAYER.as_ptr()];
        let printf_features = [vk::ValidationFeatureEnableEXT::DEBUG_PRINTF];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&printf_features);
        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .flags(flags);
        if validation {
            create_info = create_info
                .enabled_layer_names(&layers)
                .push_next(&mut validation_features);
        }
        let instance = unsafe { entry.create_instance(&create_info, None) }.context("failed to create instance")?;

        let debug_utils = if validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None) }
                .context("failed to set up debug messenger")?;
            Some((loader, messenger))
        } else {
            None
        };

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        Ok(Rc::new(Self {
            entry,
            instance,
            surface_loader,
            debug_utils,
            validation,
        }))
    }

    /// Whether the validation layers are enabled.
    pub fn validation_enabled(&self) -> bool {
        self.validation
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            if let Some((loader, messenger)) = self.debug_utils.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// The logical device, its queues and the physical device's properties.
pub struct Device {
    pub instance: Rc<VulkanInstance>,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub present_queue: vk::Queue,
    pub present_family: u32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub properties: vk::PhysicalDeviceProperties,
    pub pipeline_statistics_query_supported: bool,
    /// Serializes submissions to the queues, which Vulkan requires to be externally synchronized.
    queue_lock: Mutex<()>,
}

struct QueueFamilyIndices {
    graphics_and_compute: Option<u32>,
    present: Option<u32>,
}

fn find_queue_families(
    instance: &VulkanInstance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> QueueFamilyIndices {
    let families = unsafe {
        instance
            .instance
            .get_physical_device_queue_family_properties(physical_device)
    };
    let mut indices = QueueFamilyIndices {
        graphics_and_compute: None,
        present: None,
    };
    for (i, family) in families.iter().enumerate() {
        let i = i as u32;
        if family
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        {
            indices.graphics_and_compute = Some(i);
        }
        let present_support = surface == vk::SurfaceKHR::null()
            || unsafe {
                instance
                    .surface_loader
                    .get_physical_device_surface_support(physical_device, i, surface)
            }
            .unwrap_or(false);
        if present_support {
            indices.present = Some(i);
        }
        if indices.graphics_and_compute.is_some() && indices.present.is_some() {
            break;
        }
    }
    indices
}

fn device_extension_names(instance: &VulkanInstance, physical_device: vk::PhysicalDevice) -> Vec<CString> {
    unsafe { instance.instance.enumerate_device_extension_properties(physical_device) }
        .unwrap_or_default()
        .iter()
        .filter_map(|e| e.extension_name_as_c_str().ok().map(CStr::to_owned))
        .collect()
}

fn contains(names: &[CString], name: &CStr) -> bool {
    names.iter().any(|n| n.as_c_str() == name)
}

impl Device {
    /// Picks a physical device (discrete > integrated > virtual > other > CPU) that can
    /// present to `surface` (if not null) and creates the logical device.
    pub fn new(instance: Rc<VulkanInstance>, surface: vk::SurfaceKHR) -> Result<Rc<Self>> {
        let mut required: Vec<&CStr> = vec![ash::khr::shader_non_semantic_info::NAME];
        if surface != vk::SurfaceKHR::null() {
            required.push(ash::khr::swapchain::NAME);
        }

        let physical_devices = unsafe { instance.instance.enumerate_physical_devices() }?;
        if physical_devices.is_empty() {
            bail!("failed to find GPUs with Vulkan support!");
        }
        let score = |t: vk::PhysicalDeviceType| match t {
            vk::PhysicalDeviceType::DISCRETE_GPU => 4,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::OTHER => 1,
            _ => 0,
        };
        let mut best: Option<(i32, vk::PhysicalDevice)> = None;
        for &pd in &physical_devices {
            let props = unsafe { instance.instance.get_physical_device_properties(pd) };
            let indices = find_queue_families(&instance, pd, surface);
            let available = device_extension_names(&instance, pd);
            let extensions_supported = required.iter().all(|r| contains(&available, r));
            let swap_chain_adequate = surface == vk::SurfaceKHR::null()
                || (extensions_supported
                    && unsafe {
                        !instance
                            .surface_loader
                            .get_physical_device_surface_formats(pd, surface)?
                            .is_empty()
                            && !instance
                                .surface_loader
                                .get_physical_device_surface_present_modes(pd, surface)?
                                .is_empty()
                    });
            let suitable = indices.graphics_and_compute.is_some()
                && indices.present.is_some()
                && extensions_supported
                && swap_chain_adequate;
            if suitable && best.is_none_or(|(s, _)| score(props.device_type) > s) {
                best = Some((score(props.device_type), pd));
            }
        }
        let physical_device = best.ok_or_else(|| anyhow!("failed to find a suitable GPU!"))?.1;
        let properties = unsafe { instance.instance.get_physical_device_properties(physical_device) };
        let device_name = properties
            .device_name_as_c_str()
            .unwrap_or(c"?")
            .to_string_lossy()
            .into_owned();
        println!("Selected Vulkan device: {device_name}");

        let indices = find_queue_families(&instance, physical_device, surface);
        let graphics_family = indices.graphics_and_compute.unwrap();
        let present_family = indices.present.unwrap();
        let priorities = [1.0f32];
        let mut unique_families = vec![graphics_family];
        if present_family != graphics_family {
            unique_families.push(present_family);
        }
        let queue_infos: Vec<_> = unique_families
            .iter()
            .map(|&f| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(f)
                    .queue_priorities(&priorities)
            })
            .collect();

        let supported = unsafe { instance.instance.get_physical_device_features(physical_device) };
        let features = vk::PhysicalDeviceFeatures::default()
            .pipeline_statistics_query(supported.pipeline_statistics_query == vk::TRUE)
            // ONNX INT64 tensors use 64-bit integer shader types.
            .shader_int64(supported.shader_int64 == vk::TRUE)
            // The raster Gaussian splatting backend writes its depth output from the fragment shader.
            .fragment_stores_and_atomics(supported.fragment_stores_and_atomics == vk::TRUE);

        let mut extensions: Vec<*const c_char> = required.iter().map(|e| e.as_ptr()).collect();
        // The spec requires VK_KHR_portability_subset to be enabled whenever the device exposes it (MoltenVK).
        let available = device_extension_names(&instance, physical_device);
        if contains(&available, ash::khr::portability_subset::NAME) {
            extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        let mut scalar_block_layout = vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&features)
            .enabled_extension_names(&extensions)
            .push_next(&mut scalar_block_layout);
        let device = unsafe { instance.instance.create_device(physical_device, &create_info, None) }
            .context("failed to create logical device!")?;

        let version = properties.api_version;
        println!(
            "[VulkanContext] device: {device_name} (apiVersion={}.{}.{})",
            vk::api_version_major(version),
            vk::api_version_minor(version),
            vk::api_version_patch(version)
        );
        println!(
            "[VulkanContext] validation layers: {}",
            if instance.validation { "ON" } else { "OFF" }
        );
        let names: Vec<_> = extensions
            .iter()
            .map(|&e| unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned())
            .collect();
        println!("[VulkanContext] enabled device extensions: {}", names.join(" "));

        let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family, 0) };
        let memory_properties = unsafe { instance.instance.get_physical_device_memory_properties(physical_device) };

        Ok(Rc::new(Self {
            instance,
            physical_device,
            device,
            graphics_queue,
            graphics_family,
            present_queue,
            present_family,
            memory_properties,
            properties,
            pipeline_statistics_query_supported: supported.pipeline_statistics_query == vk::TRUE,
            queue_lock: Mutex::new(()),
        }))
    }

    /// Nanoseconds per GPU timestamp unit.
    pub fn timestamp_period(&self) -> f32 {
        self.properties.limits.timestamp_period
    }

    /// The first memory type allowed by `type_filter` that has all of `properties`.
    pub fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32> {
        (0..self.memory_properties.memory_type_count)
            .find(|&i| {
                type_filter & (1 << i) != 0
                    && self.memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
            })
            .ok_or_else(|| anyhow!("failed to find suitable memory type!"))
    }

    /// Creates a buffer with its own allocation of memory with `properties`.
    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&info, None) }.context("failed to create buffer!")?;
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let allocate = || -> Result<vk::DeviceMemory> {
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(self.find_memory_type(requirements.memory_type_bits, properties)?);
            let memory =
                unsafe { self.device.allocate_memory(&info, None) }.context("failed to allocate buffer memory!")?;
            if let Err(e) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
                unsafe { self.device.free_memory(memory, None) };
                return Err(anyhow!("failed to bind buffer memory: {e}"));
            }
            Ok(memory)
        };
        match allocate() {
            Ok(memory) => Ok((buffer, memory)),
            Err(e) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                Err(e)
            }
        }
    }

    /// `vkQueueSubmit`, serialized with the other queue operations.
    pub fn queue_submit(&self, queue: vk::Queue, submits: &[vk::SubmitInfo], fence: vk::Fence) -> Result<()> {
        let _lock = self.queue_lock.lock().unwrap();
        unsafe { self.device.queue_submit(queue, submits, fence) }.context("vkQueueSubmit failed")
    }

    /// `vkQueueWaitIdle`, serialized with the other queue operations.
    pub fn queue_wait_idle(&self, queue: vk::Queue) -> Result<()> {
        let _lock = self.queue_lock.lock().unwrap();
        unsafe { self.device.queue_wait_idle(queue) }.context("vkQueueWaitIdle failed")
    }

    fn queue_present(
        &self,
        loader: &ash::khr::swapchain::Device,
        queue: vk::Queue,
        info: &vk::PresentInfoKHR,
    ) -> Result<bool, vk::Result> {
        let _lock = self.queue_lock.lock().unwrap();
        unsafe { loader.queue_present(queue, info) }
    }

    /// The queue family of `queue`, which must be the graphics or the present queue.
    pub fn queue_family(&self, queue: vk::Queue) -> Result<u32> {
        if queue == self.graphics_queue {
            Ok(self.graphics_family)
        } else if queue == self.present_queue {
            Ok(self.present_family)
        } else {
            bail!("queue is neither the graphics nor the present queue of this device")
        }
    }

    /// Records commands into a one-time command buffer, submits it to `queue` (the
    /// graphics queue if null) and waits for it to finish.
    pub fn submit_immediate(&self, record: impl FnOnce(vk::CommandBuffer), queue: vk::Queue) -> Result<()> {
        let queue = if queue == vk::Queue::null() {
            self.graphics_queue
        } else {
            queue
        };
        let device = &self.device;
        // A pool per call: pools are externally synchronized.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.queue_family(queue)?);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }
            .context("failed to create an immediate command pool")?;
        let result = (|| -> Result<()> {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { device.allocate_command_buffers(&alloc) }?[0];
            let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { device.begin_command_buffer(cmd, &begin) }?;
            record(cmd);
            unsafe { device.end_command_buffer(cmd) }?;
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let waited = self
                .queue_submit(queue, &[submit], fence)
                .and_then(|_| unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(Into::into));
            unsafe { device.destroy_fence(fence, None) };
            waited.context("failed to run immediate commands")
        })();
        // Destroying the pool frees its command buffer.
        unsafe { device.destroy_command_pool(pool, None) };
        result
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

/// Where a [`VulkanContext`] presents: a window surface, or nothing (headless).
#[derive(Clone, Copy)]
pub enum Presentation {
    /// Render into offscreen images of [`BackendConfig::WIDTH`] x [`BackendConfig::HEIGHT`].
    Headless,
    /// Render into the swapchain of a window, whose framebuffer has the given initial size.
    Window {
        display: RawDisplayHandle,
        window: RawWindowHandle,
        framebuffer_extent: vk::Extent2D,
    },
}

/// Owns the device, the swapchain (or the headless images that stand in for it) and the frame synchronization.
///
/// A frame acquires an image with [`try_begin_render`](Self::try_begin_render),
/// submits work that waits for the image's semaphore in
/// [`image_available_semaphores_per_image`](Self::image_available_semaphores_per_image)
/// and signals the returned fence, and ends with [`end_render`](Self::end_render).
pub struct VulkanContext {
    device: Rc<Device>,
    /// The window surface, or null when headless.
    surface: vk::SurfaceKHR,
    swapchain_loader: Option<ash::khr::swapchain::Device>,
    swap_chain: vk::SwapchainKHR,
    swap_chain_images: Vec<vk::Image>,
    /// Memory of the headless images; empty with a swapchain.
    headless_image_memories: Vec<vk::DeviceMemory>,
    swap_chain_image_views: Vec<vk::ImageView>,
    swap_chain_image_format: vk::Format,
    swap_chain_extent: vk::Extent2D,
    /// The window's framebuffer size, for surfaces that leave the swapchain size to the application.
    framebuffer_extent: vk::Extent2D,
    swap_chain_out_of_date: bool,
    command_pool: vk::CommandPool,
    /// Signaled by the last submission of a frame slot; waited for before the slot is reused.
    in_flight_fences: Vec<vk::Fence>,
    /// The fence of the frame that last rendered each image (null before the first).
    image_fences: Vec<vk::Fence>,
    /// Signaled by image acquisition, one per frame slot.
    image_available_semaphores_per_frame: Vec<vk::Semaphore>,
    /// Signaled once the image is available, one per image; the graphs wait for these.
    image_available_semaphores_per_image: Vec<vk::Semaphore>,
    current_frame: usize,
}

impl VulkanContext {
    /// Creates the instance, device and offscreen images of a headless context.
    pub fn new_headless() -> Result<Self> {
        Self::new(Presentation::Headless)
    }

    /// Creates the context for `presentation`.
    pub fn new(presentation: Presentation) -> Result<Self> {
        let display = match presentation {
            Presentation::Headless => None,
            Presentation::Window { display, .. } => Some(display),
        };
        let instance = VulkanInstance::new(display)?;
        let (surface, framebuffer_extent) = match presentation {
            Presentation::Headless => (vk::SurfaceKHR::null(), vk::Extent2D::default()),
            Presentation::Window {
                display,
                window,
                framebuffer_extent,
            } => {
                let surface =
                    unsafe { ash_window::create_surface(&instance.entry, &instance.instance, display, window, None) }
                        .context("failed to create window surface!")?;
                (surface, framebuffer_extent)
            }
        };
        let device = match Device::new(instance.clone(), surface) {
            Ok(device) => device,
            Err(e) => {
                if surface != vk::SurfaceKHR::null() {
                    unsafe { instance.surface_loader.destroy_surface(surface, None) };
                }
                return Err(e);
            }
        };

        let swapchain_loader = (surface != vk::SurfaceKHR::null())
            .then(|| ash::khr::swapchain::Device::new(&device.instance.instance, &device.device));
        let mut context = Self {
            device,
            surface,
            swapchain_loader,
            swap_chain: vk::SwapchainKHR::null(),
            swap_chain_images: Vec::new(),
            headless_image_memories: Vec::new(),
            swap_chain_image_views: Vec::new(),
            swap_chain_image_format: vk::Format::UNDEFINED,
            swap_chain_extent: vk::Extent2D::default(),
            framebuffer_extent,
            swap_chain_out_of_date: false,
            command_pool: vk::CommandPool::null(),
            in_flight_fences: Vec::new(),
            image_fences: Vec::new(),
            image_available_semaphores_per_frame: Vec::new(),
            image_available_semaphores_per_image: Vec::new(),
            current_frame: 0,
        };
        if context.has_surface() {
            context.create_swap_chain(vk::SwapchainKHR::null())?;
        } else {
            context.create_swap_images_headless()?;
        }
        context.create_image_views()?;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(context.device.graphics_family);
        context.command_pool = unsafe { context.device.device.create_command_pool(&pool_info, None) }
            .context("failed to create command pool!")?;
        context.create_sync_objects()?;
        Ok(context)
    }

    /// The shared device, which buffers and pipelines keep alive.
    pub fn device(&self) -> &Rc<Device> {
        &self.device
    }

    /// The `ash::Device` function table and handle.
    pub fn vk(&self) -> &ash::Device {
        &self.device.device
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.device.graphics_queue
    }

    pub fn has_surface(&self) -> bool {
        self.surface != vk::SurfaceKHR::null()
    }

    pub fn number_of_swap_chain_images(&self) -> u32 {
        self.swap_chain_images.len() as u32
    }

    pub fn swap_chain_image(&self, index: u32) -> vk::Image {
        self.swap_chain_images[index as usize]
    }

    pub fn image_view(&self, index: u32) -> vk::ImageView {
        self.swap_chain_image_views[index as usize]
    }

    pub fn swap_chain_extent(&self) -> vk::Extent2D {
        self.swap_chain_extent
    }

    pub fn swap_chain_image_format(&self) -> vk::Format {
        self.swap_chain_image_format
    }

    /// Per image, the semaphore the work rendering into it must wait for.
    pub fn image_available_semaphores_per_image(&self) -> &[vk::Semaphore] {
        &self.image_available_semaphores_per_image
    }

    /// The command pool of the graphics queue family.
    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    /// The frame slot (0 .. MAX_FRAMES_IN_FLIGHT) the next frame uses.
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    pub fn is_swap_chain_out_of_date(&self) -> bool {
        self.swap_chain_out_of_date
    }

    /// Records commands, submits them to the graphics queue and waits for them.
    pub fn submit_immediate(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        self.device.submit_immediate(record, vk::Queue::null())
    }

    /// Copies a 4-byte-per-pixel image in `GENERAL` layout (e.g. a headless image)
    /// into host memory, row by row; the work writing it must have finished.
    pub fn read_image_to_host(&self, image: vk::Image, extent: vk::Extent2D) -> Result<Vec<u8>> {
        let bytes = extent.width as usize * extent.height as usize * 4;
        let staging = crate::vulkan_buffer::VulkanBuffer::<u8>::with_usage(
            &self.device,
            bytes as u32,
            vk::BufferUsageFlags::TRANSFER_DST,
            crate::vulkan_buffer::HOST_COHERENT,
        )?;
        self.submit_immediate(|cmd| {
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            unsafe {
                self.device.device.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::GENERAL,
                    staging.buffer(),
                    &[region],
                )
            };
        })?;
        staging.read()
    }

    fn create_swap_chain(&mut self, old: vk::SwapchainKHR) -> Result<()> {
        let surface_loader = &self.device.instance.surface_loader;
        let pd = self.device.physical_device;
        let (capabilities, formats, present_modes) = unsafe {
            (
                surface_loader.get_physical_device_surface_capabilities(pd, self.surface)?,
                surface_loader.get_physical_device_surface_formats(pd, self.surface)?,
                surface_loader.get_physical_device_surface_present_modes(pd, self.surface)?,
            )
        };
        let format = formats
            .iter()
            .copied()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .unwrap_or(formats[0]);
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        let extent = self.choose_swap_extent(&capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        // max_image_count == 0 means no upper limit.
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let families = [self.device.graphics_family, self.device.present_family];
        let mut info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            // STORAGE / COLOR_ATTACHMENT for the compute and raster paths that write the image,
            // TRANSFER_DST for clears and blits.
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);
        info = if families[0] != families[1] {
            info.image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families)
        } else {
            info.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        let loader = self.swapchain_loader.as_ref().unwrap();
        self.swap_chain = unsafe { loader.create_swapchain(&info, None) }.context("failed to create swap chain!")?;
        self.swap_chain_images = unsafe { loader.get_swapchain_images(self.swap_chain) }?;
        self.swap_chain_image_format = format.format;
        self.swap_chain_extent = extent;
        Ok(())
    }

    fn choose_swap_extent(&self, capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }
        // The surface leaves the size to the application (e.g. Wayland).
        let mut extent = self.framebuffer_extent;
        if extent.width == 0 || extent.height == 0 {
            extent = vk::Extent2D {
                width: BackendConfig::WIDTH,
                height: BackendConfig::HEIGHT,
            };
        }
        vk::Extent2D {
            width: extent
                .width
                .clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
            height: extent.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }

    fn create_swap_images_headless(&mut self) -> Result<()> {
        self.swap_chain_image_format = vk::Format::B8G8R8A8_UNORM;
        self.swap_chain_extent = vk::Extent2D {
            width: BackendConfig::WIDTH,
            height: BackendConfig::HEIGHT,
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(self.swap_chain_image_format)
            .extent(vk::Extent3D {
                width: BackendConfig::WIDTH,
                height: BackendConfig::HEIGHT,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            );
        let device = &self.device.device;
        for _ in 0..2 {
            let image = unsafe { device.create_image(&info, None) }.context("failed to create swap chain image!")?;
            self.swap_chain_images.push(image);
            let requirements = unsafe { device.get_image_memory_requirements(image) };
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(
                    self.device
                        .find_memory_type(requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?,
                );
            let memory = unsafe { device.allocate_memory(&alloc, None) }.context("failed to allocate image memory!")?;
            self.headless_image_memories.push(memory);
            unsafe { device.bind_image_memory(image, memory, 0) }?;
        }
        Ok(())
    }

    fn create_image_views(&mut self) -> Result<()> {
        for &image in &self.swap_chain_images {
            let info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.swap_chain_image_format)
                .subresource_range(color_subresource_range());
            let view = unsafe { self.device.device.create_image_view(&info, None) }
                .context("failed to create image views!")?;
            self.swap_chain_image_views.push(view);
        }
        Ok(())
    }

    fn create_sync_objects(&mut self) -> Result<()> {
        let device = &self.device.device;
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // New images have no prior submission to wait for, also after a resize.
        self.image_fences = vec![vk::Fence::null(); self.swap_chain_images.len()];
        for _ in 0..self.swap_chain_images.len() {
            self.image_available_semaphores_per_image
                .push(unsafe { device.create_semaphore(&semaphore_info, None) }?);
        }
        for _ in 0..BackendConfig::MAX_FRAMES_IN_FLIGHT {
            self.image_available_semaphores_per_frame
                .push(unsafe { device.create_semaphore(&semaphore_info, None) }?);
            self.in_flight_fences
                .push(unsafe { device.create_fence(&fence_info, None) }?);
        }
        Ok(())
    }

    fn destroy_sync_objects(&mut self) {
        let device = &self.device.device;
        unsafe {
            for s in self.image_available_semaphores_per_frame.drain(..) {
                device.destroy_semaphore(s, None);
            }
            for f in self.in_flight_fences.drain(..) {
                device.destroy_fence(f, None);
            }
            for s in self.image_available_semaphores_per_image.drain(..) {
                device.destroy_semaphore(s, None);
            }
        }
        // Borrowed handles of the frame-slot fences destroyed above.
        self.image_fences.clear();
    }

    /// Records the window's framebuffer size; a change after the first report marks the swapchain outdated.
    pub fn set_framebuffer_extent(&mut self, extent: vk::Extent2D) {
        if extent != self.framebuffer_extent {
            if self.framebuffer_extent.width != 0 || self.framebuffer_extent.height != 0 {
                self.swap_chain_out_of_date = true;
            }
            self.framebuffer_extent = extent;
        }
    }

    /// Recreates the swapchain, its image views and the sync objects for the surface's current size.
    ///
    /// Returns false, leaving the swapchain outdated, while the window is minimized
    /// or when there is no surface.
    pub fn recreate_swap_chain(&mut self) -> Result<bool> {
        if !self.has_surface() {
            return Ok(false);
        }
        let capabilities = unsafe {
            self.device
                .instance
                .surface_loader
                .get_physical_device_surface_capabilities(self.device.physical_device, self.surface)
        }?;
        let extent = self.choose_swap_extent(&capabilities);
        if extent.width == 0 || extent.height == 0 {
            self.swap_chain_out_of_date = true;
            return Ok(false);
        }
        self.device.queue_wait_idle(self.device.graphics_queue)?;
        self.device.queue_wait_idle(self.device.present_queue)?;

        // Recreated rather than reused: a skipped frame can leave a semaphore
        // signaled or a fence unsignaled with nothing pending on it.
        self.destroy_sync_objects();
        for view in self.swap_chain_image_views.drain(..) {
            unsafe { self.device.device.destroy_image_view(view, None) };
        }
        let old = self.swap_chain;
        self.create_swap_chain(old)?;
        unsafe { self.swapchain_loader.as_ref().unwrap().destroy_swapchain(old, None) };
        self.create_image_views()?;
        self.create_sync_objects()?;
        self.current_frame = 0;
        self.swap_chain_out_of_date = false;
        Ok(true)
    }

    /// Like [`try_begin_render`](Self::try_begin_render), but an outdated swapchain is an error.
    pub fn begin_render(&mut self) -> Result<(u32, vk::Fence)> {
        self.try_begin_render()?
            .ok_or_else(|| anyhow!("swap chain is out of date, call recreate_swap_chain()!"))
    }

    /// Waits for the current frame slot, acquires an image and returns its index and
    /// the fence the frame's last submission must signal.
    ///
    /// Returns None when the swapchain is outdated; nothing is pending then and the
    /// frame can be skipped. Once an image is returned, the per-image
    /// image-available semaphore is signaled when it can be rendered into.
    pub fn try_begin_render(&mut self) -> Result<Option<(u32, vk::Fence)>> {
        let device = &self.device.device;
        let fence = self.in_flight_fences[self.current_frame];
        loop {
            match unsafe { device.wait_for_fences(&[fence], true, 1_000_000_000) } {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                Err(e) => bail!("inFlightFence returned error: {e}"),
            }
        }

        let frame_semaphore = [self.image_available_semaphores_per_frame[self.current_frame]];
        let image_index = if let Some(loader) = &self.swapchain_loader {
            match unsafe {
                loader.acquire_next_image(self.swap_chain, 1_000_000_000, frame_semaphore[0], vk::Fence::null())
            } {
                Ok((index, suboptimal)) => {
                    // A suboptimal image is usable; render this frame and rebuild after.
                    if suboptimal {
                        self.swap_chain_out_of_date = true;
                    }
                    index
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swap_chain_out_of_date = true;
                    return Ok(None);
                }
                Err(e) => bail!("failed to acquire swap chain image: {e}"),
            }
        } else {
            // Headless: cycle through the images.
            (self.current_frame % self.swap_chain_images.len()) as u32
        };

        // Per-image resources are writable only after that image's last submission
        // has finished, whichever frame slot acquired it.
        let image_fence = self.image_fences[image_index as usize];
        if image_fence != vk::Fence::null() && image_fence != fence {
            unsafe { device.wait_for_fences(&[image_fence], true, u64::MAX) }
                .context("failed to wait for image resources")?;
        }
        self.image_fences[image_index as usize] = fence;
        // Reset only once this frame is certain to submit work that signals the fence.
        unsafe { device.reset_fences(&[fence]) }.context("failed to reset inFlightFence")?;

        // Hand the acquisition over to the per-image semaphore the graphs wait for.
        let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let signal = [self.image_available_semaphores_per_image[image_index as usize]];
        let mut submit = vk::SubmitInfo::default().signal_semaphores(&signal);
        if self.has_surface() {
            submit = submit
                .wait_semaphores(&frame_semaphore)
                .wait_dst_stage_mask(&wait_stages);
        }
        self.device
            .queue_submit(self.device.graphics_queue, &[submit], vk::Fence::null())
            .context("image available delegate semaphore failed to submit!")?;
        Ok(Some((image_index, fence)))
    }

    /// Presents the image once `render_finished` is signaled (headless: consumes the
    /// semaphore) and advances to the next frame slot.
    pub fn end_render(&mut self, image_index: u32, render_finished: vk::Semaphore) -> Result<()> {
        let wait = [render_finished];
        if let Some(loader) = &self.swapchain_loader {
            let swapchains = [self.swap_chain];
            let indices = [image_index];
            let info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match self.device.queue_present(loader, self.device.present_queue, &info) {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.swap_chain_out_of_date = true,
                Err(e) => bail!("failed to present swap chain image: {e}"),
            }
        } else {
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait)
                .wait_dst_stage_mask(&stages);
            self.device
                .queue_submit(self.device.graphics_queue, &[submit], vk::Fence::null())
                .context("headless render finished submit failed!")?;
        }
        self.current_frame = (self.current_frame + 1) % BackendConfig::MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    /// Ends a frame begun with `try_begin_render` without rendering: consumes the
    /// image-available semaphore, signals the fence and presents the image unchanged.
    pub fn cancel_render(&mut self, image_index: u32) -> Result<()> {
        let device = self.device.device.clone();
        let finished = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
        let mut command = vk::CommandBuffer::null();
        let result = (|| -> Result<()> {
            if self.has_surface() {
                let alloc = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                command = unsafe { device.allocate_command_buffers(&alloc) }?[0];
                let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                unsafe { device.begin_command_buffer(command, &begin) }?;
                // An acquired image needs a presentation layout even when nothing rendered it.
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(self.swap_chain_images[image_index as usize])
                    .subresource_range(color_subresource_range());
                unsafe {
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                    device.end_command_buffer(command)?;
                }
            }
            let wait = [self.image_available_semaphores_per_image[image_index as usize]];
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let signal = [finished];
            let commands = [command];
            let mut submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait)
                .wait_dst_stage_mask(&stages)
                .signal_semaphores(&signal);
            if command != vk::CommandBuffer::null() {
                submit = submit.command_buffers(&commands);
            }
            self.device.queue_submit(
                self.device.graphics_queue,
                &[submit],
                self.in_flight_fences[self.current_frame],
            )?;
            self.end_render(image_index, finished)
        })();
        // Presentation must consume the temporary semaphore before it is destroyed.
        unsafe {
            let _ = device.device_wait_idle();
            if command != vk::CommandBuffer::null() {
                device.free_command_buffers(self.command_pool, &[command]);
            }
            device.destroy_semaphore(finished, None);
        }
        result
    }

    /// Waits for all frames in flight and for the device to be idle.
    pub fn stop_render(&self) {
        unsafe {
            for &fence in &self.in_flight_fences {
                let _ = self.device.device.wait_for_fences(&[fence], true, u64::MAX);
            }
            let _ = self.device.device.device_wait_idle();
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        self.stop_render();
        let device = self.device.device.clone();
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
        }
        self.destroy_sync_objects();
        unsafe {
            for view in self.swap_chain_image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            if let Some(loader) = &self.swapchain_loader {
                loader.destroy_swapchain(self.swap_chain, None);
            } else {
                for (image, memory) in self
                    .swap_chain_images
                    .drain(..)
                    .zip(self.headless_image_memories.drain(..))
                {
                    device.destroy_image(image, None);
                    device.free_memory(memory, None);
                }
            }
            // After the swapchain, before the instance (which the device keeps alive).
            if self.has_surface() {
                self.device.instance.surface_loader.destroy_surface(self.surface, None);
            }
        }
    }
}

/// The whole colour subresource of a single-mip, single-layer image.
pub fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}
