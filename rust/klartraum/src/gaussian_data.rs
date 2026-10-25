// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Gaussian scenes: [`Gaussian3D`], the `.spz` loader and the GPU upload [`GaussianDataStandard`].

use std::path::Path;

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::computegraph::{BufferElementSinglePath, ComputeGraphElement};
use crate::gaussian_splatting_types::{BufferRef, GaussianSoABuffers};
use crate::spz;
use crate::vulkan_context::VulkanContext;

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

/// A Gaussian scene uploaded once into device-local storage shared by all paths, as [`GaussianSoABuffers`].
///
/// The CPU-side array is not kept. Consumers (e.g. the splatting backends) take the
/// buffer bundle from [`buffers`](Self::buffers), not this type, so loading and
/// ownership are decoupled from how the buffers are wired.
pub struct GaussianDataStandard {
    buffers: GaussianSoABuffers,
}

impl GaussianDataStandard {
    /// Loads an `.spz` file (see [`load_gaussians_spz`]) and uploads it.
    pub fn from_spz(context: &VulkanContext, path: impl AsRef<Path>, flip_y: bool) -> Result<Self> {
        Self::new(context, &load_gaussians_spz(path, flip_y)?)
    }

    /// Uploads `gaussians` as a structure of arrays.
    pub fn new(context: &VulkanContext, gaussians: &[Gaussian3D]) -> Result<Self> {
        let n = gaussians.len();
        anyhow::ensure!(n > 0, "GaussianDataStandard: no Gaussians");
        let mut pos = Vec::with_capacity(n);
        let mut rot = Vec::with_capacity(n);
        let mut scale = Vec::with_capacity(n);
        let mut col_alpha = Vec::with_capacity(n);
        let (mut sh_r, mut sh_g, mut sh_b) = (vec![0.0f32; 15 * n], vec![0.0f32; 15 * n], vec![0.0f32; 15 * n]);
        for (i, g) in gaussians.iter().enumerate() {
            pos.push(g.position);
            rot.push(g.rotation);
            scale.push(g.scale);
            col_alpha.push([g.color[0], g.color[1], g.color[2], g.alpha]);
            for b in 0..15 {
                sh_r[b * n + i] = g.sh_r[b];
                sh_g[b * n + i] = g.sh_g[b];
                sh_b[b * n + i] = g.sh_b[b];
            }
        }

        fn upload<T: Pod>(context: &VulkanContext, data: &[T], name: &str) -> Result<BufferRef> {
            let element = BufferElementSinglePath::<T>::new(
                context,
                data.len() as u32,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            element.set_name(name);
            element.buffer().memcopy_from(data)?;
            Ok(BufferRef::new(element))
        }
        Ok(Self {
            buffers: GaussianSoABuffers {
                count: n as u32,
                pos: upload(context, &pos, "Pos3D")?,
                rot: upload(context, &rot, "Rot3D")?,
                scale: upload(context, &scale, "Scale3D")?,
                col_alpha: upload(context, &col_alpha, "ColAlpha3D")?,
                sh_r: upload(context, &sh_r, "ShR")?,
                sh_g: upload(context, &sh_g, "ShG")?,
                sh_b: upload(context, &sh_b, "ShB")?,
            },
        })
    }

    /// The uploaded buffers.
    pub fn buffers(&self) -> &GaussianSoABuffers {
        &self.buffers
    }
}
