// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Gaussian scenes on the CPU: [`Gaussian3D`] and the `.spz` loader that produces them.

use std::path::Path;

use anyhow::Result;
use bytemuck::{Pod, Zeroable};

use crate::spz;

/// One Gaussian of a scene, with the same fields and layout as the C++ `Gaussian3D`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Gaussian3D {
    /// x, y, z
    pub position: [f32; 3],
    /// x, y, z, w
    pub rotation: [f32; 4],
    /// Linear scale (files store its log; [`load_gaussians_spz`] applies exp).
    pub scale: [f32; 3],
    /// RGB SH0 encoding.
    pub color: [f32; 3],
    /// Opacity in [0, 1] (files store its logit; [`load_gaussians_spz`] applies the sigmoid).
    pub alpha: f32,
    pub sh_r: [f32; 15],
    pub sh_g: [f32; 15],
    pub sh_b: [f32; 15],
}

fn sigmoid_standard(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Loads an `.spz` scene with activated scales and opacities, up to SH degree 3.
///
/// With `flip_y` the scene is converted from RUB to RDB, i.e. mirrored across the
/// Y axis; captures stored with Y pointing down (such as `data/lantern.spz`) need it.
pub fn load_gaussians_spz(path: impl AsRef<Path>, flip_y: bool) -> Result<Vec<Gaussian3D>> {
    let path = path.as_ref();
    let packed = spz::load_spz_packed(path)?;
    let conv = if flip_y {
        spz::coordinate_converter(spz::CoordinateSystem::Rub, spz::CoordinateSystem::Rdb)
    } else {
        spz::CoordinateConverter::default()
    };

    let gaussians: Vec<Gaussian3D> = (0..packed.num_points)
        .map(|i| {
            let ug = packed.unpack(i, &conv);
            let mut g = Gaussian3D {
                position: ug.position,
                rotation: ug.rotation,
                scale: ug.scale.map(f32::exp),
                color: ug.color,
                alpha: sigmoid_standard(ug.alpha),
                ..Default::default()
            };
            g.sh_r.copy_from_slice(&ug.sh_r[..15]);
            g.sh_g.copy_from_slice(&ug.sh_g[..15]);
            g.sh_b.copy_from_slice(&ug.sh_b[..15]);
            g
        })
        .collect();
    println!(
        "Loaded {} gaussians from {}{}",
        gaussians.len(),
        path.display(),
        if flip_y { " (Y axis flipped)" } else { "" }
    );
    Ok(gaussians)
}

/// The per-axis median of the Gaussian positions, as `test_scene::loadLantern()` computes the scene centre.
///
/// A few stray splats do not pull it off the scene, unlike the mean.
pub fn median_centre(gaussians: &[Gaussian3D]) -> [f32; 3] {
    let mut centre = [0.0; 3];
    if gaussians.is_empty() {
        return centre;
    }
    let mut values = vec![0.0f32; gaussians.len()];
    for axis in 0..3 {
        for (value, g) in values.iter_mut().zip(gaussians) {
            *value = g.position[axis];
        }
        let middle = values.len() / 2;
        let (_, median, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
        centre[axis] = *median;
    }
    centre
}
