// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! The server: builds the described graphs with the Rust engine and runs them.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use ash::vk;
use flatbuffers::FlatBufferBuilder;
use klartraum::computegraph::{
    BufferElementSinglePath, ComputeGraph, ElementPtr, GeneralComputation, HostMemory, HostMemoryBufferElement, NoOp,
};
use klartraum::vulkan_buffer::HOST_COHERENT;
use klartraum::vulkan_context::VulkanContext;

use crate::framing::{read_message, write_message};
use crate::schema;
use crate::shared_memory::SharedMemoryRegion;

/// A buffer the server copies between its region and its own Vulkan buffer around
/// each execution, when the device cannot import host memory.
struct CopiedBuffer {
    region: Rc<SharedMemoryRegion>,
    buffer: Rc<BufferElementSinglePath<u8>>,
}

/// A compiled graph and what its executions need.
struct CompiledGraph {
    graph: ComputeGraph,
    /// Empty when the buffers are the shared memory itself.
    copied: Vec<CopiedBuffer>,
}

/// Compiles and runs the graphs a client describes, on a headless Vulkan context.
pub struct Server {
    graphs: HashMap<u32, CompiledGraph>,
    /// Declared last, so the graphs are dropped before the context.
    context: VulkanContext,
}

/// What handling a request produced.
pub struct Reply {
    /// The response message to send.
    pub message: Vec<u8>,
    /// Whether the client asked the server to end.
    pub shutdown: bool,
}

fn response(request_id: u64, result: Result<(bool, f32)>) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::new();
    let (ok, error, zero_copy, gpu_ms) = match result {
        Ok((zero_copy, gpu_ms)) => (true, None, zero_copy, gpu_ms),
        Err(e) => (false, Some(fbb.create_string(&format!("{e:#}"))), false, 0.0),
    };
    let body = schema::Response::create(
        &mut fbb,
        &schema::ResponseArgs {
            ok,
            error,
            zero_copy,
            gpu_ms,
        },
    );
    let message = schema::Message::create(
        &mut fbb,
        &schema::MessageArgs {
            request_id,
            body_type: schema::Body::Response,
            body: Some(body.as_union_value()),
        },
    );
    fbb.finish(message, None);
    fbb.finished_data().to_vec()
}

impl Server {
    pub fn new() -> Result<Self> {
        Ok(Self {
            graphs: HashMap::new(),
            context: VulkanContext::new_headless()?,
        })
    }

    /// Whether buffers are imported shared memory (else they are copied around executions).
    pub fn zero_copy(&self) -> bool {
        HostMemoryBufferElement::is_supported(&self.context)
    }

    /// Handles one request message and returns the response.
    pub fn handle(&mut self, message: &[u8]) -> Reply {
        let (request_id, result, shutdown) = match schema::root_as_message(message) {
            Err(e) => (0, Err(anyhow!("invalid message: {e}")), false),
            Ok(message) => {
                let id = message.request_id();
                match message.body_as_request_body() {
                    None => (id, Err(anyhow!("the message is not a request")), false),
                    Some(body) => {
                        let shutdown = body.request_type() == schema::Request::Shutdown;
                        (id, self.dispatch(body), shutdown)
                    }
                }
            }
        };
        Reply {
            message: response(request_id, result),
            shutdown,
        }
    }

    fn dispatch(&mut self, body: schema::RequestBody) -> Result<(bool, f32)> {
        if let Some(compile) = body.request_as_compile_graph() {
            let zero_copy = self.compile(compile)?;
            Ok((zero_copy, 0.0))
        } else if let Some(execute) = body.request_as_execute() {
            self.execute(execute.graph_id())
        } else if let Some(release) = body.request_as_release() {
            self.graphs
                .remove(&release.graph_id())
                .ok_or_else(|| anyhow!("unknown graph {}", release.graph_id()))?;
            Ok((self.zero_copy(), 0.0))
        } else if body.request_as_shutdown().is_some() {
            Ok((self.zero_copy(), 0.0))
        } else {
            bail!("unknown request")
        }
    }

    fn compile(&mut self, compile: schema::CompileGraph) -> Result<bool> {
        let graph_id = compile.graph_id();
        ensure!(!self.graphs.contains_key(&graph_id), "graph {graph_id} already exists");
        let zero_copy = self.zero_copy();

        let regions = compile
            .regions()
            .iter()
            .map(|r| SharedMemoryRegion::open(r.name(), r.size() as usize).map(Rc::new))
            .collect::<Result<Vec<_>>>()?;

        let descriptions: Vec<schema::Element> = compile.elements().iter().collect();
        let mut elements: HashMap<u32, ElementPtr> = HashMap::new();
        let mut computations: HashMap<u32, Rc<GeneralComputation<()>>> = HashMap::new();
        let mut copied = Vec::new();
        for description in &descriptions {
            let id = description.id();
            ensure!(!elements.contains_key(&id), "element {id} is described twice");
            let element: ElementPtr = match description.kind() {
                schema::ElementKind::NoOp => NoOp::new(),
                schema::ElementKind::Buffer => {
                    let region = regions
                        .get(description.region() as usize)
                        .ok_or_else(|| anyhow!("element {id}: unknown region {}", description.region()))?
                        .clone();
                    let size = description.byte_size() as usize;
                    ensure!(
                        size > 0 && size <= region.len(),
                        "element {id}: {size} bytes outside its region"
                    );
                    if zero_copy {
                        HostMemoryBufferElement::import(&self.context, region, size)
                            .with_context(|| format!("element {id}"))?
                    } else {
                        let buffer = BufferElementSinglePath::<u8>::new(
                            &self.context,
                            size as u32,
                            vk::BufferUsageFlags::STORAGE_BUFFER,
                            HOST_COHERENT,
                        )?;
                        copied.push(CopiedBuffer {
                            region,
                            buffer: buffer.clone(),
                        });
                        buffer
                    }
                }
                schema::ElementKind::Computation => {
                    let shaders: Vec<Vec<u32>> = description
                        .shaders()
                        .ok_or_else(|| anyhow!("element {id}: a computation without shaders"))?
                        .iter()
                        .map(|s| s.code().iter().collect())
                        .collect();
                    ensure!(!shaders.is_empty(), "element {id}: a computation without shaders");
                    let computation = GeneralComputation::<()>::from_spirv(shaders);
                    computation.set_group_count(
                        description.group_count_x(),
                        description.group_count_y(),
                        description.group_count_z(),
                    );
                    if let Some(rounds) = description.push_constants() {
                        computation
                            .set_raw_push_constants(rounds.iter().map(|p| p.data().bytes().to_vec()).collect())?;
                    }
                    computations.insert(id, computation.clone());
                    computation
                }
                _ => bail!("element {id}: unsupported kind"),
            };
            if let Some(name) = description.name() {
                element.set_name(name);
            }
            elements.insert(id, element);
        }

        let find = |id: u32| {
            elements
                .get(&id)
                .cloned()
                .ok_or_else(|| anyhow!("unknown element {id}"))
        };

        // An input given as a slot of another element resolves through that element's
        // inputs, so elements are wired after the ones they take slots from.
        let mut wired = vec![false; descriptions.len()];
        let mut index_of: HashMap<u32, usize> = HashMap::new();
        for (i, d) in descriptions.iter().enumerate() {
            index_of.insert(d.id(), i);
        }
        let mut remaining = descriptions.len();
        while remaining > 0 {
            let before = remaining;
            for (i, description) in descriptions.iter().enumerate() {
                if wired[i] {
                    continue;
                }
                let inputs: Vec<schema::Input> = description.inputs().map(|v| v.iter().collect()).unwrap_or_default();
                let ready = inputs.iter().all(|input| {
                    input.slot() < 0 || index_of.get(&input.element()).is_some_and(|&k| wired[k] || k == i)
                });
                if !ready {
                    continue;
                }
                let element = find(description.id())?;
                for input in &inputs {
                    element
                        .set_input_slot(find(input.element())?, input.index(), input.slot())
                        .with_context(|| format!("element {}: input {}", description.id(), input.index()))?;
                }
                if let Some(computation) = computations.get(&description.id()) {
                    for scratch in description.scratch().iter().flatten() {
                        computation.add_scratch_buffer_element(find(scratch.element())?, scratch.zero());
                    }
                }
                for dependency in description.dependencies().iter().flatten() {
                    element.add_dependency(find(dependency)?)?;
                }
                wired[i] = true;
                remaining -= 1;
            }
            ensure!(remaining < before, "the slot inputs of the elements form a cycle");
        }

        let root = find(compile.root())?;
        let mut graph = ComputeGraph::new(&self.context, 1);
        graph.enable_profiling();
        graph.compile_from(&root, &self.context)?;
        self.graphs.insert(graph_id, CompiledGraph { graph, copied });
        Ok(zero_copy)
    }

    fn execute(&mut self, graph_id: u32) -> Result<(bool, f32)> {
        let queue = self.context.graphics_queue();
        let compiled = self
            .graphs
            .get_mut(&graph_id)
            .ok_or_else(|| anyhow!("unknown graph {graph_id}"))?;
        for copy in &compiled.copied {
            let size = copy.buffer.buffer().buffer_mem_size() as usize;
            copy.buffer
                .buffer()
                .memcopy_from_bytes(&copy.region.as_slice()[..size])?;
        }
        compiled.graph.submit_and_wait(queue, 0)?;
        for copy in &compiled.copied {
            let size = copy.buffer.buffer().buffer_mem_size() as usize;
            let data = copy.buffer.buffer().read()?;
            // The client does not touch the region while the execution runs.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), copy.region.ptr(), size) };
        }
        Ok((compiled.copied.is_empty(), compiled.graph.last_gpu_span_ms() as f32))
    }

    /// Answers the requests of one connection until it closes or asks for a shutdown;
    /// returns whether a shutdown was requested.
    pub fn serve(&mut self, stream: &mut (impl Read + Write)) -> Result<bool> {
        while let Some(message) = read_message(stream)? {
            let reply = self.handle(&message);
            write_message(stream, &reply.message)?;
            if reply.shutdown {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Listens on the Unix socket `path` and serves one connection after another until
/// a client requests a shutdown. A stale socket file at `path` is replaced.
pub fn listen(path: &Path) -> Result<()> {
    let mut server = Server::new()?;
    if path.exists() {
        std::fs::remove_file(path).with_context(|| format!("cannot remove {}", path.display()))?;
    }
    let listener = UnixListener::bind(path).with_context(|| format!("cannot listen on {}", path.display()))?;
    println!(
        "klartraum_ipc_server: listening on {} (zero copy: {})",
        path.display(),
        server.zero_copy()
    );
    let result = (|| -> Result<()> {
        for stream in listener.incoming() {
            let mut stream = stream?;
            match server.serve(&mut stream) {
                Ok(true) => return Ok(()),
                Ok(false) => {}
                // A broken connection ends that client only.
                Err(e) => eprintln!("klartraum_ipc_server: connection ended: {e:#}"),
            }
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(path);
    result
}
