// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - orbit_mvp_matches_cpp: the orbit camera with the frameLantern() parameters at the
 *   headless extent writes the CameraMVP the C++ InterfaceCameraOrbit writes
 *   (reference dumped by RustReference.dumpOrbitCameraMvp), within 1e-6
 **/
mod common;

use ash::vk;
use klartraum::camera::CameraMVP;
use klartraum::gaussian_data::{load_gaussians_spz, median_centre};
use klartraum::glm::Vec3;
use klartraum::interface_camera::{InterfaceCamera, InterfaceCameraOrbit, UpDirection};

#[test]
fn orbit_mvp_matches_cpp() {
    let Some(mvp_bytes) = common::read_reference("lantern_camera_mvp.bin") else {
        return;
    };
    let Some(extent_bytes) = common::read_reference("headless_extent.bin") else {
        return;
    };
    let expected: CameraMVP = bytemuck::pod_read_unaligned(&mvp_bytes);
    let extent: [u32; 2] = bytemuck::pod_read_unaligned(&extent_bytes);

    let gaussians = load_gaussians_spz(common::LANTERN_PATH, true).unwrap();
    let centre = Vec3::from(median_centre(&gaussians));
    let mut orbit = InterfaceCameraOrbit::new(UpDirection::Y);
    orbit.set_position(-centre);
    orbit.set_azimuth(0.9f32 as f64);
    orbit.set_elevation(-0.5f32 as f64);
    orbit.set_distance(0.35f32 as f64);
    let mut mvp = CameraMVP::default();
    orbit.update(
        &mut mvp,
        vk::Extent2D {
            width: extent[0],
            height: extent[1],
        },
    );

    let a: &[f32] = bytemuck::cast_slice(std::slice::from_ref(&mvp));
    let e: &[f32] = bytemuck::cast_slice(std::slice::from_ref(&expected));
    let differing = a.iter().zip(e).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let max = a.iter().zip(e).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("CameraMVP: {differing} of 52 floats differ in their bits, max abs difference {max:e}");
    assert!(max <= 1e-6, "{mvp:#?}\n!=\n{expected:#?}");
}
