// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - lantern_header: the lantern file is an NGSP (version 4) file with 145562 degree-3 Gaussians
 * - lantern_matches_cpp_loader: load_gaussians_spz with the Y flip yields, field by field, the
 *   Gaussians the C++ loadGaussiansSpz yields (reference dumped by RustReference.dumpLanternGaussians)
 * - lantern_without_flip_matches_cpp_loader: likewise without the Y flip
 * - lantern_centre_matches_cpp: the per-axis median centre equals test_scene::loadLantern().centre
 **/
mod common;

use klartraum::gaussian_data::{Gaussian3D, load_gaussians_spz};
use klartraum::spz;

/// Compares two scenes field by field and returns the largest absolute difference.
fn max_difference(actual: &[Gaussian3D], expected: &[Gaussian3D]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let a: &[f32] = bytemuck::cast_slice(actual);
    let e: &[f32] = bytemuck::cast_slice(expected);
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(e) {
        if x == y {
            continue; // also covers equal infinities
        }
        max = max.max((x - y).abs());
    }
    max
}

fn compare_with_reference(flip_y: bool, reference: &str) {
    let Some(bytes) = common::read_reference(reference) else {
        return;
    };
    let expected: Vec<Gaussian3D> = bytemuck::pod_collect_to_vec(&bytes);
    let actual = load_gaussians_spz(common::LANTERN_PATH, flip_y).unwrap();
    let identical = bytemuck::cast_slice::<_, u8>(&actual) == &bytes[..];
    let max = max_difference(&actual, &expected);
    println!("{reference}: bit-identical={identical}, max abs difference={max:e}");
    assert!(max <= 1e-6, "Gaussians differ from the C++ loader by up to {max}");
}

#[test]
fn lantern_header() {
    common::enter_repo_root();
    let packed = spz::load_spz_packed(common::LANTERN_PATH).unwrap();
    assert_eq!(packed.version, 4);
    assert_eq!(packed.num_points, 145562);
    assert_eq!(packed.sh_degree, 3);
    assert!(packed.uses_quaternion_smallest_three);
    assert!(!packed.uses_float16());
}

#[test]
fn lantern_matches_cpp_loader() {
    compare_with_reference(true, "lantern_gaussians_flip_y.bin");
}

#[test]
fn lantern_without_flip_matches_cpp_loader() {
    compare_with_reference(false, "lantern_gaussians.bin");
}

#[test]
fn lantern_centre_matches_cpp() {
    let Some(bytes) = common::read_reference("lantern_centre.bin") else {
        return;
    };
    let expected: Vec<f32> = bytemuck::pod_collect_to_vec(&bytes);
    let gaussians = load_gaussians_spz(common::LANTERN_PATH, true).unwrap();
    let centre = klartraum::gaussian_data::median_centre(&gaussians);
    assert_eq!(centre.to_vec(), expected);
}
