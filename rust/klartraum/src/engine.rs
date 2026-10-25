// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`KlartraumEngine`], which owns the context and the compiled graphs and drives the frames.

use std::collections::VecDeque;
use std::rc::Rc;

use anyhow::Result;
use ash::vk;

use crate::computegraph::{CameraUbo, ComputeGraph, ElementPtr};
use crate::events::Event;
use crate::interface_camera::InterfaceCamera;
use crate::vulkan_context::VulkanContext;

/// Builds the graphs of an engine; see [`KlartraumEngine::set_graph_builder`].
pub type GraphBuilder = Box<dyn FnMut(&mut KlartraumEngine) -> Result<()>>;

/// Uploads application data for the acquired path before the graphs are submitted.
pub type FrameUpdate = Box<dyn FnMut(u32) -> Result<()>>;

/// Owns the Vulkan context and the compiled graphs, and renders frames.
///
/// Each [`step`](Self::step) acquires an image, runs the frame update, passes the
/// queued events to the camera and writes the camera uniform buffer, submits every
/// graph's path for the image (the last one signals the frame's fence) and presents.
pub struct KlartraumEngine {
    /// The compiled graphs, submitted in order every frame.
    compute_graphs: Vec<ComputeGraph>,
    interface_camera: Option<Box<dyn InterfaceCamera>>,
    /// The uniform buffer the camera writes every frame.
    camera_ubo: Option<Rc<CameraUbo>>,
    event_queue: VecDeque<Event>,
    frame_update: Option<FrameUpdate>,
    /// Rebuilds the graphs for a new swapchain; with one, the engine follows window resizes.
    graph_builder: Option<GraphBuilder>,
    profiling_enabled: bool,
    /// Declared last, so it is dropped after everything that renders with it.
    vulkan_context: VulkanContext,
}

impl KlartraumEngine {
    pub fn new(vulkan_context: VulkanContext) -> Self {
        Self {
            compute_graphs: Vec::new(),
            interface_camera: None,
            camera_ubo: None,
            event_queue: VecDeque::new(),
            frame_update: None,
            graph_builder: None,
            profiling_enabled: false,
            vulkan_context,
        }
    }

    pub fn vulkan_context(&self) -> &VulkanContext {
        &self.vulkan_context
    }

    pub fn vulkan_context_mut(&mut self) -> &mut VulkanContext {
        &mut self.vulkan_context
    }

    /// The input events `step` passes to the camera.
    pub fn event_queue(&mut self) -> &mut VecDeque<Event> {
        &mut self.event_queue
    }

    pub fn set_interface_camera(&mut self, camera: Box<dyn InterfaceCamera>) {
        self.interface_camera = Some(camera);
    }

    pub fn interface_camera_mut(&mut self) -> Option<&mut (dyn InterfaceCamera + 'static)> {
        self.interface_camera.as_deref_mut()
    }

    pub fn clear_interface_camera(&mut self) {
        self.interface_camera = None;
    }

    pub fn set_camera_ubo(&mut self, camera_ubo: Option<Rc<CameraUbo>>) {
        self.camera_ubo = camera_ubo;
    }

    pub fn set_frame_update(&mut self, update: Option<FrameUpdate>) {
        self.frame_update = update;
    }

    /// Records GPU timestamps in graphs added from now on; each step then waits for its frame.
    pub fn enable_profiling(&mut self) {
        self.profiling_enabled = true;
    }

    pub fn disable_profiling(&mut self) {
        self.profiling_enabled = false;
    }

    /// Compiles a graph from `element` with one path per swapchain image.
    pub fn add(&mut self, element: ElementPtr) -> Result<()> {
        let mut graph = ComputeGraph::new(&self.vulkan_context, self.vulkan_context.number_of_swap_chain_images());
        if self.profiling_enabled {
            graph.enable_profiling();
        }
        graph.compile_from(&element, &self.vulkan_context)?;
        self.compute_graphs.push(graph);
        Ok(())
    }

    /// The compiled graphs.
    pub fn compute_graphs(&self) -> &[ComputeGraph] {
        &self.compute_graphs
    }

    pub fn clear_compute_graphs(&mut self) {
        self.compute_graphs.clear();
    }

    /// Sets the function that builds the graphs and runs it now. The engine runs it
    /// again, after dropping the graphs, whenever the swapchain is recreated, so
    /// everything tied to the swapchain (image sources, render passes) belongs in it.
    /// `None` removes it.
    pub fn set_graph_builder(&mut self, builder: Option<GraphBuilder>) -> Result<()> {
        self.graph_builder = builder;
        self.compute_graphs.clear();
        self.run_graph_builder()
    }

    fn run_graph_builder(&mut self) -> Result<()> {
        if let Some(mut builder) = self.graph_builder.take() {
            let result = builder(self);
            self.graph_builder = Some(builder);
            result?;
        }
        Ok(())
    }

    /// Whether the engine can follow a new swapchain (it has a graph builder).
    pub fn is_resizable(&self) -> bool {
        self.graph_builder.is_some()
    }

    fn rebuild_for_swap_chain(&mut self) -> Result<bool> {
        // Waits for the queues before anything is destroyed.
        if !self.vulkan_context.recreate_swap_chain()? {
            return Ok(false);
        }
        // The graphs recorded the old swapchain's image views and semaphores.
        self.compute_graphs.clear();
        self.run_graph_builder()?;
        Ok(true)
    }

    /// Renders one frame.
    pub fn step(&mut self) -> Result<()> {
        let (image_index, fence) = if self.is_resizable() {
            if self.vulkan_context.is_swap_chain_out_of_date() && !self.rebuild_for_swap_chain()? {
                return Ok(()); // minimized: nothing to render into
            }
            match self.vulkan_context.try_begin_render()? {
                Some(frame) => frame,
                None => return Ok(()), // outdated since the last frame; rebuilt on the next step
            }
        } else {
            // Without a builder the graphs cannot follow a new swapchain, so an outdated one is an error.
            self.vulkan_context.begin_render()?
        };

        // The acquired path is free for CPU updates, which the pre-recorded graphs read.
        if let Some(update) = self.frame_update.as_mut() {
            if let Err(e) = update(image_index) {
                self.vulkan_context.cancel_render(image_index)?;
                return Err(e);
            }
        }

        if let (Some(camera), Some(ubo)) = (self.interface_camera.as_mut(), self.camera_ubo.as_ref()) {
            while let Some(event) = self.event_queue.pop_front() {
                camera.on_event(&event);
            }
            camera.update(&mut ubo.ubo_mut(), self.vulkan_context.swap_chain_extent());
            ubo.update(image_index);
        }

        if self.compute_graphs.is_empty() {
            // Nothing to render: present the image as it is and signal the fence.
            return self.vulkan_context.cancel_render(image_index);
        }

        let queue = self.vulkan_context.graphics_queue();
        let last = self.compute_graphs.len() - 1;
        let mut render_finished = vk::Semaphore::null();
        for (i, graph) in self.compute_graphs.iter_mut().enumerate() {
            // Only the frame's last submission signals the fence.
            render_finished = graph.submit_to(queue, image_index, if i == last { fence } else { vk::Fence::null() })?;
        }
        self.vulkan_context.end_render(image_index, render_finished)?;

        // Profiling waits for the frame so its timestamps can be read back.
        if self.profiling_enabled {
            self.vulkan_context.device().queue_wait_idle(queue)?;
            for graph in &mut self.compute_graphs {
                graph.read_and_accumulate_timestamps();
            }
        }
        Ok(())
    }

    /// `(label, mean milliseconds)` of every element of every graph.
    pub fn profiling_results(&mut self) -> Result<Vec<(String, f32)>> {
        self.vulkan_context
            .device()
            .queue_wait_idle(self.vulkan_context.graphics_queue())?;
        let mut results = Vec::new();
        for graph in &mut self.compute_graphs {
            graph.read_and_accumulate_timestamps();
            results.extend(graph.profiling_results());
        }
        Ok(results)
    }
}

impl Drop for KlartraumEngine {
    fn drop(&mut self) {
        self.vulkan_context.stop_render();
        // The builder and the frame update may hold GPU resources (e.g. a captured model).
        self.frame_update = None;
        self.graph_builder = None;
        self.compute_graphs.clear();
        self.camera_ubo = None;
        self.interface_camera = None;
    }
}

/// A frontend without a window, for tests and compute-only work: an engine on a headless context.
pub struct HeadlessFrontend {
    engine: KlartraumEngine,
}

impl HeadlessFrontend {
    pub fn new() -> Result<Self> {
        Ok(Self {
            engine: KlartraumEngine::new(VulkanContext::new_headless()?),
        })
    }

    pub fn engine(&mut self) -> &mut KlartraumEngine {
        &mut self.engine
    }
}
