// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`ComputeGraph`], which compiles a DAG of elements into one pre-recorded command buffer per path.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use anyhow::{Context, Result, bail, ensure};
use ash::vk;

use crate::computegraph::element::{ElementPtr, element_key};
use crate::vulkan_context::{Device, VulkanContext};

/// Timestamp query slots per pool; MoltenVK backs a pool with one counter sample
/// buffer of at most 4096 samples (2 per element).
const PROFILING_ELEMENTS_PER_POOL: usize = 2048;

/// A compiled graph: the elements reachable from one element, recorded in
/// topological order into one command buffer per path.
///
/// A full memory barrier separates consecutive elements, so each element sees all
/// writes of the elements before it; that is what orders producers before
/// consumers. A path's submission waits for the external semaphores its elements
/// were given ([`set_wait_for`](crate::computegraph::ComputeGraphElement::set_wait_for))
/// and signals the path's graph-finished semaphore.
pub struct ComputeGraph {
    device: Rc<Device>,
    number_paths: u32,
    command_pool: vk::CommandPool,
    queue_family: u32,
    command_buffers: Vec<vk::CommandBuffer>,
    /// The elements in recording order, including those that record nothing (buffers, groups).
    ordered_elements: Vec<ElementPtr>,
    /// The elements with a host-side update, collected once.
    updatable_elements: Vec<ElementPtr>,
    /// Per path, the external semaphores the submission waits for.
    wait_semaphores: Vec<Vec<vk::Semaphore>>,
    graph_finished_semaphores: Vec<vk::Semaphore>,

    profiling_enabled: bool,
    profiling_query_pools: Vec<vk::QueryPool>,
    /// Per element: accumulated nanoseconds and sample count.
    profiling_accum: Vec<(f64, u64)>,
    /// From the first element's start to the last element's end, per submission: nanoseconds and count.
    profiling_span_accum: (f64, u64),
    /// The span of the last submission read, in nanoseconds.
    last_span_ns: f64,
    timestamps_unread: bool,
}

impl ComputeGraph {
    pub fn new(context: &VulkanContext, number_paths: u32) -> Self {
        Self {
            device: context.device().clone(),
            number_paths,
            command_pool: vk::CommandPool::null(),
            queue_family: 0,
            command_buffers: Vec::new(),
            ordered_elements: Vec::new(),
            updatable_elements: Vec::new(),
            wait_semaphores: Vec::new(),
            graph_finished_semaphores: Vec::new(),
            profiling_enabled: false,
            profiling_query_pools: Vec::new(),
            profiling_accum: Vec::new(),
            profiling_span_accum: (0.0, 0),
            last_span_ns: 0.0,
            timestamps_unread: false,
        }
    }

    /// Records GPU timestamps around every element; call before [`compile_from`](Self::compile_from).
    pub fn enable_profiling(&mut self) {
        self.profiling_enabled = true;
    }

    /// The compiled elements in recording order.
    pub fn elements(&self) -> &[ElementPtr] {
        &self.ordered_elements
    }

    /// The recorded command buffer of a path.
    pub fn command_buffer(&self, path_id: u32) -> vk::CommandBuffer {
        self.command_buffers[path_id as usize]
    }

    /// The elements that execute before `element`: its inputs and its execution-only dependencies, each once.
    pub fn predecessors(element: &ElementPtr) -> Vec<ElementPtr> {
        let mut predecessors: Vec<ElementPtr> = Vec::new();
        for candidate in element.inputs().into_values().chain(element.dependencies()) {
            if !predecessors.iter().any(|p| element_key(p) == element_key(&candidate)) {
                predecessors.push(candidate);
            }
        }
        predecessors
    }

    /// Compiles the graph of the elements `element` reaches for the graphics queue:
    /// orders them, sets them up and records every path's command buffer.
    pub fn compile_from(&mut self, element: &ElementPtr, context: &VulkanContext) -> Result<()> {
        ensure!(
            self.command_pool == vk::CommandPool::null(),
            "compile_from() may only be called once per graph"
        );
        let device = &self.device.device;
        self.queue_family = self.device.graphics_family;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.queue_family);
        self.command_pool =
            unsafe { device.create_command_pool(&pool_info, None) }.context("failed to create command pool!")?;

        self.ordered_elements = Self::topological_order(element)?;

        for _ in 0..self.number_paths {
            let semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .context("failed to create graph finished semaphore!")?;
            self.graph_finished_semaphores.push(semaphore);
        }

        for element in &self.ordered_elements {
            element
                .setup(context, self.number_paths)
                .with_context(|| format!("setting up {} '{}'", element.type_name(), element.name()))?;
        }
        self.updatable_elements = self
            .ordered_elements
            .iter()
            .filter(|e| e.is_updatable())
            .cloned()
            .collect();

        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(self.number_paths);
        self.command_buffers =
            unsafe { device.allocate_command_buffers(&alloc) }.context("failed to allocate command buffers!")?;

        if self.profiling_enabled {
            self.profiling_accum = vec![(0.0, 0); self.ordered_elements.len()];
            for chunk in self.ordered_elements.chunks(PROFILING_ELEMENTS_PER_POOL) {
                let info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2 * chunk.len() as u32);
                self.profiling_query_pools.push(
                    unsafe { device.create_query_pool(&info, None) }
                        .context("failed to create timestamp query pool")?,
                );
            }
        }

        for path_id in 0..self.number_paths {
            self.record_path(path_id)?;
            let waits = self.collect_wait_semaphores(path_id);
            self.wait_semaphores.push(waits);
        }
        Ok(())
    }

    fn record_path(&self, path_id: u32) -> Result<()> {
        let device = &self.device.device;
        let cmd = self.command_buffers[path_id as usize];
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .context("failed to begin recording command buffer!")?;
        }
        for (i, element) in self.ordered_elements.iter().enumerate() {
            if i > 0 {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    )
                };
            }
            let timestamps = self
                .profiling_query_pools
                .get(i / PROFILING_ELEMENTS_PER_POOL)
                .map(|&pool| (pool, 2 * (i % PROFILING_ELEMENTS_PER_POOL) as u32));
            if let Some((pool, query)) = timestamps {
                unsafe {
                    device.cmd_reset_query_pool(cmd, pool, query, 2);
                    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pool, query);
                }
            }
            element
                .record(cmd, path_id)
                .with_context(|| format!("recording {} '{}'", element.type_name(), element.name()))?;
            if let Some((pool, query)) = timestamps {
                unsafe { device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, query + 1) };
            }
        }
        unsafe { device.end_command_buffer(cmd) }.context("failed to record command buffer!")?;
        Ok(())
    }

    fn collect_wait_semaphores(&self, path_id: u32) -> Vec<vk::Semaphore> {
        let mut waits = Vec::new();
        for element in &self.ordered_elements {
            if let Some(semaphore) = element.base().render_wait_semaphore(path_id) {
                if !waits.contains(&semaphore) {
                    waits.push(semaphore);
                }
            }
        }
        waits
    }

    fn check_queue(&self, queue: vk::Queue) -> Result<()> {
        ensure!(
            self.command_pool != vk::CommandPool::null(),
            "the graph must be compiled with compile_from() before it is submitted"
        );
        ensure!(
            self.device.queue_family(queue)? == self.queue_family,
            "the queue does not belong to the queue family the graph was compiled for"
        );
        Ok(())
    }

    /// Runs the host-side updates of a path and submits it; `fence` (may be null) is
    /// signaled when it finishes. Returns the path's graph-finished semaphore.
    pub fn submit_to(&mut self, queue: vk::Queue, path_id: u32, fence: vk::Fence) -> Result<vk::Semaphore> {
        self.check_queue(queue)?;
        for element in &self.updatable_elements {
            element.update(path_id)?;
        }
        let waits = &self.wait_semaphores[path_id as usize];
        let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let cmds = [self.command_buffers[path_id as usize]];
        let signal = [self.graph_finished_semaphores[path_id as usize]];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(waits)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&cmds)
            .signal_semaphores(&signal);
        self.device
            .queue_submit(queue, &[submit], fence)
            .context("failed to submit the graph elements!")?;
        self.timestamps_unread = self.profiling_enabled;
        Ok(signal[0])
    }

    /// Submits a path, waits for it and consumes its graph-finished semaphore.
    pub fn submit_and_wait(&mut self, queue: vk::Queue, path_id: u32) -> Result<()> {
        let device = self.device.device.clone();
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let waited = self
            .submit_to(queue, path_id, fence)
            .and_then(|_| unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.map_err(Into::into));
        unsafe { device.destroy_fence(fence, None) };
        waited.context("failed waiting for compute graph fence")?;

        // Drain the graph-finished semaphore so the path can be submitted again.
        let wait = [self.graph_finished_semaphores[path_id as usize]];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let drain = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages);
        self.device.queue_submit(queue, &[drain], vk::Fence::null())?;
        self.device.queue_wait_idle(queue)?;
        self.read_and_accumulate_timestamps();
        Ok(())
    }

    /// Reads the timestamps of the last submission (which must have finished) into the means.
    pub fn read_and_accumulate_timestamps(&mut self) {
        if !self.timestamps_unread || self.profiling_query_pools.is_empty() {
            return;
        }
        let n = self.ordered_elements.len();
        let mut ts = vec![0u64; 2 * n];
        for (p, &pool) in self.profiling_query_pools.iter().enumerate() {
            let first = p * PROFILING_ELEMENTS_PER_POOL;
            let count = (n - first).min(PROFILING_ELEMENTS_PER_POOL);
            let result = unsafe {
                self.device.device.get_query_pool_results(
                    pool,
                    0,
                    &mut ts[2 * first..2 * (first + count)],
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            if result.is_err() {
                return;
            }
        }
        self.timestamps_unread = false;
        let period = self.device.timestamp_period() as f64;
        let (mut first, mut last) = (u64::MAX, 0u64);
        for i in 0..n {
            let (start, end) = (ts[2 * i], ts[2 * i + 1]);
            if end >= start {
                self.profiling_accum[i].0 += (end - start) as f64 * period;
                self.profiling_accum[i].1 += 1;
                if start > 0 {
                    first = first.min(start);
                    last = last.max(end);
                }
            }
        }
        if last > first {
            self.last_span_ns = (last - first) as f64 * period;
            self.profiling_span_accum.0 += (last - first) as f64 * period;
            self.profiling_span_accum.1 += 1;
        }
    }

    /// The mean GPU time from the first element's start to the last element's end, in milliseconds.
    pub fn gpu_span_ms(&self) -> f64 {
        let (total, count) = self.profiling_span_accum;
        if count > 0 { total / count as f64 * 1e-6 } else { 0.0 }
    }

    /// The GPU time of the last submission read back, from the first element's start to the last element's end, in milliseconds.
    pub fn last_gpu_span_ms(&self) -> f64 {
        self.last_span_ns * 1e-6
    }

    /// `(label, mean milliseconds)` per element in recording order; the label is the
    /// element's name, or its type if it has none.
    pub fn profiling_results(&self) -> Vec<(String, f32)> {
        self.ordered_elements
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let name = e.name();
                let label = if name.is_empty() {
                    e.type_name().to_owned()
                } else {
                    name
                };
                let (total, count) = self.profiling_accum.get(i).copied().unwrap_or((0.0, 0));
                (
                    label,
                    if count > 0 {
                        (total / count as f64 * 1e-6) as f32
                    } else {
                        0.0
                    },
                )
            })
            .collect()
    }

    /// The elements `element` reaches, in recording order: every element after its
    /// predecessors. Kahn's algorithm over the predecessor edges, from `element`
    /// backwards, reversed; an error if the elements form a cycle.
    pub fn topological_order(element: &ElementPtr) -> Result<Vec<ElementPtr>> {
        type Edges = HashMap<usize, Vec<ElementPtr>>;
        fn fill_edges(edges: &mut Edges, incoming: &mut Edges, element: &ElementPtr) {
            for input in ComputeGraph::predecessors(element) {
                let list = edges.entry(element_key(element)).or_default();
                if !list.iter().any(|e| element_key(e) == element_key(&input)) {
                    list.push(input.clone());
                    incoming.entry(element_key(&input)).or_default().push(element.clone());
                    fill_edges(edges, incoming, &input);
                }
            }
        }

        let mut edges = Edges::new();
        let mut incoming = Edges::new();
        fill_edges(&mut edges, &mut incoming, element);
        // The walk starts at the root, so a cycle through the root shows up as an edge into it.
        if incoming.get(&element_key(element)).is_some_and(|e| !e.is_empty()) {
            bail!("graph has cycles!");
        }

        let mut ordered = Vec::new();
        let mut queue = VecDeque::from([element.clone()]);
        while let Some(n) = queue.pop_front() {
            ordered.push(n.clone());
            for m in Self::predecessors(&n) {
                let n_edges = edges.entry(element_key(&n)).or_default();
                if let Some(pos) = n_edges.iter().position(|e| element_key(e) == element_key(&m)) {
                    n_edges.remove(pos);
                    let m_incoming = incoming.entry(element_key(&m)).or_default();
                    if let Some(pos) = m_incoming.iter().position(|e| element_key(e) == element_key(&n)) {
                        m_incoming.remove(pos);
                    }
                    if m_incoming.is_empty() {
                        queue.push_back(m);
                    }
                }
            }
        }
        let remaining: usize =
            edges.values().map(Vec::len).sum::<usize>() + incoming.values().map(Vec::len).sum::<usize>();
        if remaining > 0 {
            bail!("graph has cycles!");
        }
        ordered.reverse();
        Ok(ordered)
    }
}

impl Drop for ComputeGraph {
    fn drop(&mut self) {
        let device = &self.device.device;
        unsafe {
            for &semaphore in &self.graph_finished_semaphores {
                device.destroy_semaphore(semaphore, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                // Destroying the pool frees its command buffers.
                device.destroy_command_pool(self.command_pool, None);
            }
            for &pool in &self.profiling_query_pools {
                device.destroy_query_pool(pool, None);
            }
        }
    }
}
