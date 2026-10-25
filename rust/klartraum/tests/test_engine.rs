// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - step_without_graphs_presents: a headless engine without graphs renders frames without errors
 * - step_clears_the_image_of_each_path: a graph made of one image source with a clear colour
 *   leaves every headless image in that colour after one frame per image
 * - frame_update_sees_each_image_index: the frame update runs once per step with the acquired image index
 * - graph_builder_runs_on_set_and_makes_the_engine_resizable: setting a builder runs it, and the
 *   graphs it adds are compiled
 * - camera_writes_the_ubo_of_the_acquired_path: with an interface camera and a camera UBO, each
 *   step writes the camera matrices into the UBO
 **/
mod common;

use std::cell::RefCell;
use std::rc::Rc;

use klartraum::camera::CameraMVP;
use klartraum::computegraph::{CameraUbo, ComputeGraphElement, ImageViewSrc, NoOp};
use klartraum::engine::HeadlessFrontend;
use klartraum::interface_camera::{InterfaceCameraOrbit, UpDirection};
use klartraum::vulkan_context::validation_error_count;

#[test]
fn step_without_graphs_presents() {
    let _gpu = common::gpu_lock();
    let errors = validation_error_count();
    let mut frontend = HeadlessFrontend::new().unwrap();
    for _ in 0..3 {
        frontend.engine().step().unwrap();
    }
    assert_eq!(validation_error_count(), errors);
}

#[test]
fn step_clears_the_image_of_each_path() {
    let _gpu = common::gpu_lock();
    let errors = validation_error_count();
    let mut frontend = HeadlessFrontend::new().unwrap();
    let engine = frontend.engine();
    let source = ImageViewSrc::for_swap_chain(engine.vulkan_context());
    source.set_clear_color([1.0, 0.5, 0.0, 1.0]);
    engine.add(source).unwrap();
    for _ in 0..2 {
        engine.step().unwrap();
    }
    let vc = engine.vulkan_context();
    vc.device().queue_wait_idle(vc.graphics_queue()).unwrap();
    for image in 0..vc.number_of_swap_chain_images() {
        let pixels = vc
            .read_image_to_host(vc.swap_chain_image(image), vc.swap_chain_extent())
            .unwrap();
        // BGRA: blue 0, green 128 (0.5 rounds to 128 or 127), red 255, alpha 255.
        for px in pixels.chunks(4) {
            assert_eq!(px[0], 0);
            assert!(px[1] == 127 || px[1] == 128);
            assert_eq!(px[2], 255);
            assert_eq!(px[3], 255);
        }
    }
    assert_eq!(validation_error_count(), errors);
}

#[test]
fn frame_update_sees_each_image_index() {
    let _gpu = common::gpu_lock();
    let mut frontend = HeadlessFrontend::new().unwrap();
    let seen = Rc::new(RefCell::new(Vec::new()));
    let seen_in_update = seen.clone();
    let engine = frontend.engine();
    engine.set_frame_update(Some(Box::new(move |index| {
        seen_in_update.borrow_mut().push(index);
        Ok(())
    })));
    let source = ImageViewSrc::for_swap_chain(engine.vulkan_context());
    engine.add(source).unwrap();
    for _ in 0..4 {
        engine.step().unwrap();
    }
    assert_eq!(*seen.borrow(), vec![0, 1, 0, 1]);
}

#[test]
fn graph_builder_runs_on_set_and_makes_the_engine_resizable() {
    let _gpu = common::gpu_lock();
    let mut frontend = HeadlessFrontend::new().unwrap();
    let engine = frontend.engine();
    assert!(!engine.is_resizable());
    engine
        .set_graph_builder(Some(Box::new(|e| {
            let source = ImageViewSrc::for_swap_chain(e.vulkan_context());
            let root = NoOp::new();
            root.set_input(source, 0)?;
            e.add(root)
        })))
        .unwrap();
    assert!(engine.is_resizable());
    assert_eq!(engine.compute_graphs().len(), 1);
    assert_eq!(engine.compute_graphs()[0].elements().len(), 2);
    engine.step().unwrap();
}

#[test]
fn camera_writes_the_ubo_of_the_acquired_path() {
    let _gpu = common::gpu_lock();
    let mut frontend = HeadlessFrontend::new().unwrap();
    let engine = frontend.engine();
    let ubo = CameraUbo::new();
    let root = NoOp::new();
    root.set_input(ImageViewSrc::for_swap_chain(engine.vulkan_context()), 0)
        .unwrap();
    root.set_input(ubo.clone(), 1).unwrap();
    engine.add(root).unwrap();
    engine.set_camera_ubo(Some(ubo.clone()));
    engine.set_interface_camera(Box::new(InterfaceCameraOrbit::new(UpDirection::Y)));
    engine.step().unwrap();
    assert_ne!(*ubo.ubo(), CameraMVP::default());
    assert_eq!(ubo.ubo().view[3][0], 0.0);
    assert_eq!(ubo.ubo().view[3][2], -2.0);
}
