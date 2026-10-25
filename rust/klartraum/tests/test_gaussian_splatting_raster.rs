// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - lantern_renders_without_validation_errors: the raster backend renders the lantern over
 *   several frames on both headless paths without validation errors, and the image is not black
 * - stages_produce_consistent_buffers: after a frame, the visible count lies in (0, N], the draw
 *   arguments hold 4 vertices and the visible count as instances, the depth keys over the visible
 *   count are sorted and the sorted indices are distinct splat ids
 * - render_is_deterministic: two renders of the same scene are byte-identical
 * - lantern_matches_cpp_render: the render equals the C++ raster backend's render of the same
 *   scene and camera (golden image dumped by RustReference.dumpRasterLanternRender); the
 *   difference statistics are printed, and the Rust render is written to
 *   build/TestingOutput/rust_gsplat_raster.ppm
 **/
mod common;

use std::collections::HashSet;
use std::io::Write;
use std::rc::Rc;

use klartraum::computegraph::{CameraUbo, ImageViewSrc};
use klartraum::engine::HeadlessFrontend;
use klartraum::gaussian_data::{Gaussian3D, GaussianDataStandard, load_gaussians_spz, median_centre};
use klartraum::gaussian_splatting_types::GsplatConfig;
use klartraum::glm::Vec3;
use klartraum::interface_camera::{InterfaceCamera, InterfaceCameraOrbit, UpDirection};
use klartraum::vulkan_context::validation_error_count;
use klartraum::vulkan_gaussian_splatting_raster::VulkanGaussianSplattingRaster;

fn lantern() -> Vec<Gaussian3D> {
    common::enter_repo_root();
    load_gaussians_spz(common::LANTERN_PATH, true).unwrap()
}

/// The camera of `test_scene::frameLantern()`.
fn frame_lantern(gaussians: &[Gaussian3D]) -> InterfaceCameraOrbit {
    let mut orbit = InterfaceCameraOrbit::new(UpDirection::Y);
    orbit.set_position(-Vec3::from(median_centre(gaussians)));
    orbit.set_azimuth(0.9f32 as f64);
    orbit.set_elevation(-0.5f32 as f64);
    orbit.set_distance(0.35f32 as f64);
    orbit
}

/// Renders `frames` frames as GaussianSplattingRaster.classWithLanternScene does and
/// returns headless image 0 (BGRA) and the backend.
fn render(gaussians: &[Gaussian3D], frames: usize) -> (Vec<u8>, HeadlessFrontend, Rc<VulkanGaussianSplattingRaster>) {
    let mut frontend = HeadlessFrontend::new().unwrap();
    let engine = frontend.engine();
    let vc = engine.vulkan_context();
    let source = ImageViewSrc::for_swap_chain(vc);
    let camera_ubo = CameraUbo::new();
    frame_lantern(gaussians).update(&mut camera_ubo.ubo_mut(), vc.swap_chain_extent());
    let model = GaussianDataStandard::new(vc, gaussians).unwrap();
    let splatting =
        VulkanGaussianSplattingRaster::new(vc, source, camera_ubo.clone(), model.buffers(), GsplatConfig::default())
            .unwrap();
    engine.add(splatting.clone()).unwrap();
    for path in 0..engine.vulkan_context().number_of_swap_chain_images() {
        camera_ubo.update(path);
    }
    for _ in 0..frames {
        engine.step().unwrap();
        let vc = engine.vulkan_context();
        vc.device().queue_wait_idle(vc.graphics_queue()).unwrap();
    }
    let vc = engine.vulkan_context();
    let pixels = vc
        .read_image_to_host(vc.swap_chain_image(0), vc.swap_chain_extent())
        .unwrap();
    (pixels, frontend, splatting)
}

fn write_ppm(name: &str, bgra: &[u8], width: u32, height: u32) {
    let dir = std::path::Path::new("build/TestingOutput");
    std::fs::create_dir_all(dir).unwrap();
    let mut f = std::fs::File::create(dir.join(name)).unwrap();
    write!(f, "P6\n{width} {height}\n255\n").unwrap();
    let rgb: Vec<u8> = bgra.chunks(4).flat_map(|p| [p[2], p[1], p[0]]).collect();
    f.write_all(&rgb).unwrap();
}

#[test]
fn lantern_renders_without_validation_errors() {
    let _gpu = common::gpu_lock();
    let gaussians = lantern();
    let errors = validation_error_count();
    let (pixels, _frontend, _splatting) = render(&gaussians, 5);
    assert_eq!(validation_error_count(), errors, "validation errors while rendering");
    let max = *pixels.iter().max().unwrap();
    assert!(max > 10, "rendered image is all black");
}

#[test]
fn stages_produce_consistent_buffers() {
    let _gpu = common::gpu_lock();
    let gaussians = lantern();
    let n = gaussians.len() as u32;
    // One frame renders path 0.
    let (_pixels, _frontend, splatting) = render(&gaussians, 1);
    let visible = splatting.visible_count().buffer(0).read().unwrap()[0];
    println!("visible splats: {visible} of {n}");
    assert!(visible > 0 && visible <= n);

    let args = splatting.draw_args().buffer(0).read().unwrap()[0];
    assert_eq!(args.vertex_count, 4);
    assert_eq!(args.instance_count, visible);

    let keys = splatting.keys().buffer(0).read().unwrap();
    let keys = &keys[..visible as usize];
    assert!(keys.windows(2).all(|w| w[0] <= w[1]), "depth keys are not sorted");

    let indices = splatting.sorted_indices().buffer(0).read().unwrap();
    let indices = &indices[..visible as usize];
    assert!(indices.iter().all(|&i| i < n));
    assert_eq!(
        indices.iter().collect::<HashSet<_>>().len(),
        visible as usize,
        "sorted indices repeat"
    );
}

#[test]
fn render_is_deterministic() {
    let _gpu = common::gpu_lock();
    let gaussians = lantern();
    let (a, _f1, _s1) = render(&gaussians, 1);
    let (b, _f2, _s2) = render(&gaussians, 1);
    assert!(a == b, "two renders differ");
}

#[test]
fn lantern_matches_cpp_render() {
    let _gpu = common::gpu_lock();
    let Some(expected) = common::read_reference("raster_lantern.bgra") else {
        return;
    };
    let gaussians = lantern();
    let (pixels, mut frontend, _splatting) = render(&gaussians, 5);
    let extent = frontend.engine().vulkan_context().swap_chain_extent();
    write_ppm("rust_gsplat_raster.ppm", &pixels, extent.width, extent.height);
    assert_eq!(pixels.len(), expected.len());

    let mut sum = 0u64;
    let mut max = 0u8;
    let mut over_2 = 0usize;
    let mut identical_pixels = 0usize;
    for (a, e) in pixels.chunks(4).zip(expected.chunks(4)) {
        let mut same = true;
        for (&x, &y) in a.iter().zip(e) {
            let d = x.abs_diff(y);
            sum += d as u64;
            max = max.max(d);
            if d != 0 {
                same = false;
            }
        }
        if a.iter().zip(e).any(|(&x, &y)| x.abs_diff(y) > 2) {
            over_2 += 1;
        }
        if same {
            identical_pixels += 1;
        }
    }
    let n_pixels = pixels.len() / 4;
    let mean = sum as f64 / pixels.len() as f64;
    println!(
        "vs C++: mean abs diff {mean:.4}/255, max {max}/255, identical pixels {:.3}%, pixels off by more than 2/255: {over_2}",
        100.0 * identical_pixels as f64 / n_pixels as f64
    );
    assert!(mean < 0.5, "mean abs difference {mean} too large");
    assert!(
        (over_2 as f64) < 0.001 * n_pixels as f64,
        "{over_2} pixels differ by more than 2/255"
    );
}
