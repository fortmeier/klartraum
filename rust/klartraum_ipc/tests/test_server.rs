// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - add_graph_runs_on_shared_memory: a compiled operator_add graph writes A + B into the result
 *   region, again with new inputs on a second Execute without recompiling, and the response
 *   reports zero copy when the device can import host memory
 * - slot_inputs_chain_computations: doubling feeding an addition through a slot input computes
 *   2A + B, so the server orders the producer first
 * - errors_are_reported_not_fatal: executing an unknown graph, compiling with an unknown region
 *   and executing a released graph answer ok = false with a message, and the server keeps serving
 * - shutdown_ends_the_connection: after a Shutdown request serve() returns true
 **/
use std::os::unix::net::UnixStream;
use std::thread::JoinHandle;

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use klartraum::vulkan_helpers::read_file;
use klartraum_ipc::framing::{read_message, write_message};
use klartraum_ipc::schema::*;
use klartraum_ipc::server::Server;
use klartraum_ipc::shared_memory::SharedMemoryRegion;

/// A server on its own thread (it is not Send) serving one end of a socket pair.
fn start_server() -> (UnixStream, JoinHandle<bool>) {
    let (client, mut server_end) = UnixStream::pair().unwrap();
    let handle = std::thread::spawn(move || {
        let mut server = Server::new().unwrap();
        server.serve(&mut server_end).unwrap()
    });
    (client, handle)
}

struct Reply {
    ok: bool,
    error: String,
    zero_copy: bool,
}

fn request(
    stream: &mut UnixStream,
    id: u64,
    build: impl FnOnce(&mut FlatBufferBuilder) -> (Request, WIPOffset<flatbuffers::UnionWIPOffset>),
) -> Reply {
    let mut fbb = FlatBufferBuilder::new();
    let (kind, request) = build(&mut fbb);
    let body = RequestBody::create(
        &mut fbb,
        &RequestBodyArgs {
            request_type: kind,
            request: Some(request),
        },
    );
    let message = Message::create(
        &mut fbb,
        &MessageArgs {
            request_id: id,
            body_type: Body::RequestBody,
            body: Some(body.as_union_value()),
        },
    );
    fbb.finish(message, None);
    write_message(stream, fbb.finished_data()).unwrap();
    let bytes = read_message(stream).unwrap().unwrap();
    let message = root_as_message(&bytes).unwrap();
    assert_eq!(message.request_id(), id);
    let response = message.body_as_response().unwrap();
    Reply {
        ok: response.ok(),
        error: response.error().unwrap_or("").to_owned(),
        zero_copy: response.zero_copy(),
    }
}

fn execute(stream: &mut UnixStream, id: u64, graph_id: u32) -> Reply {
    request(stream, id, |fbb| {
        (
            Request::Execute,
            Execute::create(fbb, &ExecuteArgs { graph_id }).as_union_value(),
        )
    })
}

/// A graph description: regions by name, buffers (region index, bytes), computations.
enum Desc {
    Buffer {
        region: u32,
        bytes: u64,
    },
    Computation {
        shader: &'static str,
        groups: u32,
        inputs: Vec<(i32, u32, i32)>,
    },
}

fn compile(
    stream: &mut UnixStream,
    id: u64,
    graph_id: u32,
    regions: &[&SharedMemoryRegion],
    elements: &[Desc],
    root: u32,
) -> Reply {
    request(stream, id, |fbb| {
        let regions: Vec<_> = regions
            .iter()
            .map(|r| {
                let name = fbb.create_string(r.name());
                SharedRegion::create(
                    fbb,
                    &SharedRegionArgs {
                        name: Some(name),
                        size: r.as_slice().len() as u64,
                    },
                )
            })
            .collect();
        let regions = fbb.create_vector(&regions);
        let elements: Vec<_> = elements
            .iter()
            .enumerate()
            .map(|(i, d)| match d {
                Desc::Buffer { region, bytes } => Element::create(
                    fbb,
                    &ElementArgs {
                        id: i as u32,
                        kind: ElementKind::Buffer,
                        region: *region,
                        byte_size: *bytes,
                        ..Default::default()
                    },
                ),
                Desc::Computation { shader, groups, inputs } => {
                    let words: Vec<u32> = bytemuck::pod_collect_to_vec(&read_file(shader).unwrap());
                    let code = fbb.create_vector(&words);
                    let spirv = Spirv::create(fbb, &SpirvArgs { code: Some(code) });
                    let shaders = fbb.create_vector(&[spirv]);
                    let inputs: Vec<_> = inputs
                        .iter()
                        .map(|&(index, element, slot)| Input::create(fbb, &InputArgs { index, element, slot }))
                        .collect();
                    let inputs = fbb.create_vector(&inputs);
                    Element::create(
                        fbb,
                        &ElementArgs {
                            id: i as u32,
                            kind: ElementKind::Computation,
                            shaders: Some(shaders),
                            group_count_x: *groups,
                            inputs: Some(inputs),
                            ..Default::default()
                        },
                    )
                }
            })
            .collect();
        let elements = fbb.create_vector(&elements);
        let compile = CompileGraph::create(
            fbb,
            &CompileGraphArgs {
                graph_id,
                regions: Some(regions),
                elements: Some(elements),
                root,
            },
        );
        (Request::CompileGraph, compile.as_union_value())
    })
}

fn floats(region: &SharedMemoryRegion, n: usize) -> Vec<f32> {
    bytemuck::pod_collect_to_vec(&region.as_slice()[..4 * n])
}

fn write_floats(region: &mut SharedMemoryRegion, values: &[f32]) {
    region.as_mut_slice()[..4 * values.len()].copy_from_slice(bytemuck::cast_slice(values));
}

fn shutdown(stream: &mut UnixStream, handle: JoinHandle<bool>) {
    let reply = request(stream, 999, |fbb| {
        (
            Request::Shutdown,
            Shutdown::create(fbb, &ShutdownArgs {}).as_union_value(),
        )
    });
    assert!(reply.ok);
    assert!(handle.join().unwrap());
}

#[test]
fn add_graph_runs_on_shared_memory() {
    let n = 4096usize;
    let (mut a, mut b, r) = (
        SharedMemoryRegion::create(4 * n).unwrap(),
        SharedMemoryRegion::create(4 * n).unwrap(),
        SharedMemoryRegion::create(4 * n).unwrap(),
    );
    write_floats(&mut a, &(0..n).map(|i| i as f32).collect::<Vec<_>>());
    write_floats(&mut b, &vec![0.25; n]);
    let (mut stream, handle) = start_server();
    let bytes = 4 * n as u64;
    let elements = [
        Desc::Buffer { region: 0, bytes },
        Desc::Buffer { region: 1, bytes },
        Desc::Buffer { region: 2, bytes },
        Desc::Computation {
            shader: "shaders/operator_add.comp.spv",
            groups: n as u32,
            inputs: vec![(0, 0, -1), (1, 1, -1), (2, 2, -1)],
        },
    ];
    let reply = compile(&mut stream, 1, 7, &[&a, &b, &r], &elements, 3);
    assert!(reply.ok, "{}", reply.error);
    println!("zero copy: {}", reply.zero_copy);

    let reply = execute(&mut stream, 2, 7);
    assert!(reply.ok, "{}", reply.error);
    assert_eq!(floats(&r, n), (0..n).map(|i| i as f32 + 0.25).collect::<Vec<_>>());

    write_floats(&mut b, &vec![-1.0; n]);
    let reply = execute(&mut stream, 3, 7);
    assert!(reply.ok, "{}", reply.error);
    assert_eq!(floats(&r, n), (0..n).map(|i| i as f32 - 1.0).collect::<Vec<_>>());
    shutdown(&mut stream, handle);
}

#[test]
fn slot_inputs_chain_computations() {
    let n = 512usize;
    let regions: Vec<SharedMemoryRegion> = (0..5).map(|_| SharedMemoryRegion::create(4 * n).unwrap()).collect();
    let mut regions = regions;
    write_floats(&mut regions[0], &(0..n).map(|i| i as f32).collect::<Vec<_>>());
    write_floats(&mut regions[3], &vec![10.0; n]);
    let (mut stream, handle) = start_server();
    let bytes = 4 * n as u64;
    // 0: A, 1: unused, 2: doubled, 3: B, 4: result; 5: double(A) -> doubled; 6: add(doubled via slot 2 of 5, B) -> result.
    let elements = [
        Desc::Buffer { region: 0, bytes },
        Desc::Buffer { region: 1, bytes },
        Desc::Buffer { region: 2, bytes },
        Desc::Buffer { region: 3, bytes },
        Desc::Buffer { region: 4, bytes },
        Desc::Computation {
            shader: "shaders/operator_double.comp.spv",
            groups: n as u32,
            inputs: vec![(0, 0, -1), (1, 1, -1), (2, 2, -1)],
        },
        Desc::Computation {
            shader: "shaders/operator_add.comp.spv",
            groups: n as u32,
            inputs: vec![(0, 5, 2), (1, 3, -1), (2, 4, -1)],
        },
    ];
    let refs: Vec<&SharedMemoryRegion> = regions.iter().collect();
    let reply = compile(&mut stream, 1, 1, &refs, &elements, 6);
    assert!(reply.ok, "{}", reply.error);
    let reply = execute(&mut stream, 2, 1);
    assert!(reply.ok, "{}", reply.error);
    assert_eq!(
        floats(&regions[4], n),
        (0..n).map(|i| 2.0 * i as f32 + 10.0).collect::<Vec<_>>()
    );
    shutdown(&mut stream, handle);
}

#[test]
fn errors_are_reported_not_fatal() {
    let (mut stream, handle) = start_server();
    let reply = execute(&mut stream, 1, 42);
    assert!(!reply.ok);
    assert!(reply.error.contains("unknown graph 42"), "{}", reply.error);

    let region = SharedMemoryRegion::create(1024).unwrap();
    let reply = compile(
        &mut stream,
        2,
        1,
        &[&region],
        &[Desc::Buffer { region: 3, bytes: 16 }],
        0,
    );
    assert!(!reply.ok);
    assert!(reply.error.contains("unknown region 3"), "{}", reply.error);

    let elements = [Desc::Buffer { region: 0, bytes: 16 }];
    assert!(compile(&mut stream, 3, 2, &[&region], &elements, 0).ok);
    let reply = request(&mut stream, 4, |fbb| {
        (
            Request::Release,
            Release::create(fbb, &ReleaseArgs { graph_id: 2 }).as_union_value(),
        )
    });
    assert!(reply.ok, "{}", reply.error);
    assert!(!execute(&mut stream, 5, 2).ok);
    shutdown(&mut stream, handle);
}

#[test]
fn shutdown_ends_the_connection() {
    let (mut stream, handle) = start_server();
    shutdown(&mut stream, handle);
}
