// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - compile_graph_round_trips_over_a_socket: a CompileGraph request with a region, a buffer and a
 *   computation (SPIR-V, push constants, inputs with a slot) is framed, sent over a Unix socket
 *   pair and read back with every field intact
 * - stream_end_between_messages_is_not_an_error: reading after the last message returns None
 **/
use std::os::unix::net::UnixStream;

use flatbuffers::FlatBufferBuilder;
use klartraum_ipc::framing::{read_message, write_message};
use klartraum_ipc::schema::*;

fn compile_graph_message() -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::new();
    let name = fbb.create_string("/kt1_0");
    let region = SharedRegion::create(
        &mut fbb,
        &SharedRegionArgs {
            name: Some(name),
            size: 65536,
        },
    );
    let regions = fbb.create_vector(&[region]);

    let buffer = Element::create(
        &mut fbb,
        &ElementArgs {
            id: 0,
            kind: ElementKind::Buffer,
            region: 0,
            byte_size: 1024,
            ..Default::default()
        },
    );
    let code = fbb.create_vector(&[0x0723_0203u32, 1, 2, 3]);
    let spirv = Spirv::create(&mut fbb, &SpirvArgs { code: Some(code) });
    let shaders = fbb.create_vector(&[spirv]);
    let data = fbb.create_vector(&[1u8, 2, 3, 4]);
    let pc = PushConstants::create(&mut fbb, &PushConstantsArgs { data: Some(data) });
    let push_constants = fbb.create_vector(&[pc]);
    let input = Input::create(
        &mut fbb,
        &InputArgs {
            index: 0,
            element: 0,
            slot: -1,
        },
    );
    let inputs = fbb.create_vector(&[input]);
    let computation = Element::create(
        &mut fbb,
        &ElementArgs {
            id: 1,
            kind: ElementKind::Computation,
            shaders: Some(shaders),
            group_count_x: 7,
            push_constants: Some(push_constants),
            inputs: Some(inputs),
            ..Default::default()
        },
    );
    let elements = fbb.create_vector(&[buffer, computation]);
    let compile = CompileGraph::create(
        &mut fbb,
        &CompileGraphArgs {
            graph_id: 3,
            regions: Some(regions),
            elements: Some(elements),
            root: 1,
        },
    );
    let body = RequestBody::create(
        &mut fbb,
        &RequestBodyArgs {
            request_type: Request::CompileGraph,
            request: Some(compile.as_union_value()),
        },
    );
    let message = Message::create(
        &mut fbb,
        &MessageArgs {
            request_id: 42,
            body_type: Body::RequestBody,
            body: Some(body.as_union_value()),
        },
    );
    fbb.finish(message, None);
    fbb.finished_data().to_vec()
}

#[test]
fn compile_graph_round_trips_over_a_socket() {
    let (mut a, mut b) = UnixStream::pair().unwrap();
    write_message(&mut a, &compile_graph_message()).unwrap();
    let bytes = read_message(&mut b).unwrap().unwrap();

    let message = root_as_message(&bytes).unwrap();
    assert_eq!(message.request_id(), 42);
    let request = message.body_as_request_body().unwrap();
    let compile = request.request_as_compile_graph().unwrap();
    assert_eq!(compile.graph_id(), 3);
    assert_eq!(compile.root(), 1);
    assert_eq!(compile.regions().get(0).name(), "/kt1_0");
    assert_eq!(compile.regions().get(0).size(), 65536);
    let elements = compile.elements();
    assert_eq!(elements.len(), 2);
    assert_eq!(elements.get(0).kind(), ElementKind::Buffer);
    assert_eq!(elements.get(0).byte_size(), 1024);
    let computation = elements.get(1);
    assert_eq!(computation.kind(), ElementKind::Computation);
    assert_eq!(computation.group_count_x(), 7);
    assert_eq!(computation.group_count_y(), 1);
    assert_eq!(
        computation.shaders().unwrap().get(0).code().iter().collect::<Vec<_>>(),
        vec![0x0723_0203, 1, 2, 3]
    );
    assert_eq!(
        computation.push_constants().unwrap().get(0).data().bytes(),
        &[1, 2, 3, 4]
    );
    let input = computation.inputs().unwrap().get(0);
    assert_eq!((input.index(), input.element(), input.slot()), (0, 0, -1));
}

#[test]
fn stream_end_between_messages_is_not_an_error() {
    let (mut a, mut b) = UnixStream::pair().unwrap();
    write_message(&mut a, b"x").unwrap();
    drop(a);
    assert_eq!(read_message(&mut b).unwrap(), Some(b"x".to_vec()));
    assert_eq!(read_message(&mut b).unwrap(), None);
}
