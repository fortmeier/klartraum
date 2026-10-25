// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Configuration, push constants and buffer bundles shared by the Gaussian splatting backends.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};

use crate::computegraph::{ComputeGraphElement, ElementPtr, GeneralComputation};

/// Runtime-tunable knobs of the Gaussian splatting pipeline; set before construction.
#[derive(Clone, Copy, Debug)]
pub struct GsplatConfig {
    /// Gaussian footprint multiplier of the compute backend's bin overlap test (2.5 sigma).
    pub spread_multiplier: f32,
    /// The compute backend's binned buffer holds `N * max_mod` entries.
    pub max_mod: u32,
    /// Hard cap on the radix sort's workgroup count.
    pub num_sort_wgs_cap: u32,
    /// Splatting tile size of the compute backend.
    pub splat_tile_x: u32,
    pub splat_tile_y: u32,
    /// Spherical-harmonics degree the raster backend evaluates per splat: 0 (DC only) to 3.
    pub sh_degree: i32,
    /// Splats with a post-activation alpha below this are culled before sorting (0 disables).
    pub alpha_cull_threshold: f32,
    /// Use the VK_EXT_mesh_shader draw path; the Rust port has only the vertex path and falls back to it.
    pub use_mesh_shader: bool,
    /// Also write a depth buffer; not ported yet, so the raster backend rejects it.
    pub output_depth: bool,
}

impl Default for GsplatConfig {
    fn default() -> Self {
        Self {
            spread_multiplier: 2.5,
            max_mod: 2,
            num_sort_wgs_cap: 320,
            splat_tile_x: 8,
            splat_tile_y: 8,
            sh_degree: 3,
            alpha_cull_threshold: 1.0 / 255.0,
            use_mesh_shader: false,
            output_depth: false,
        }
    }
}

/// Push constants of `gsplat_cull_and_compact_*.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CullAndCompactPushConstants {
    pub num_splats: u32,
    /// Dilates the cull frustum so near-edge splat footprints survive.
    pub frustum_dilation: f32,
    /// Drops splats with a post-activation alpha below this (0 = off).
    pub alpha_threshold: f32,
}

/// Push constants of `gsplat_raster_project.comp`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RasterProjectPushConstants {
    pub num_splats: u32,
    pub screen_width: f32,
    pub screen_height: f32,
    pub splat_scale: f32,
    pub sh_degree: i32,
}

/// Push constants of the radix sort shaders (`gsplat_radix_sort_*.comp`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SortPushConstants {
    pub pass: u32,
    pub num_elements: u32,
    pub num_bins: u32,
    /// 1: read the active element count from the count buffer (binding 6) instead of `num_elements`.
    pub use_count_buffer: u32,
}

pub type GaussianCullAndCompact = GeneralComputation<CullAndCompactPushConstants>;
pub type GaussianRasterProject = GeneralComputation<RasterProjectPushConstants>;
pub type RadixSort = GeneralComputation<SortPushConstants>;

/// A buffer as a consumer connects to it: `element` itself (slot -1) or the buffer
/// at input `slot` of `element`, e.g. one that a compute element writes; the consumer
/// then depends on that element, so it runs first.
#[derive(Clone)]
pub struct BufferRef {
    pub element: ElementPtr,
    pub slot: i32,
}

impl BufferRef {
    pub fn new(element: ElementPtr) -> Self {
        Self { element, slot: -1 }
    }

    /// The buffer element itself.
    pub fn buffer(&self) -> Result<ElementPtr> {
        if self.slot < 0 {
            Ok(self.element.clone())
        } else {
            self.element.input_element(self.slot)
        }
    }

    /// Makes the buffer input `index` of `consumer`.
    pub fn connect_to(&self, consumer: &dyn ComputeGraphElement, index: i32) -> Result<()> {
        consumer.set_input_slot(self.element.clone(), index, self.slot)
    }
}

/// A 3D Gaussian model on the GPU as a structure of arrays, as both splatting backends read it.
///
/// Seven buffers — position vec3, rotation vec4 (x, y, z, w), linear scale vec3, colour
/// and alpha vec4, and three spherical-harmonics streams of 15 floats per Gaussian,
/// coefficient-major (`sh[b * count + i]`) — plus the number of Gaussians.
#[derive(Clone)]
pub struct GaussianSoABuffers {
    pub count: u32,
    pub pos: BufferRef,
    pub rot: BufferRef,
    pub scale: BufferRef,
    pub col_alpha: BufferRef,
    pub sh_r: BufferRef,
    pub sh_g: BufferRef,
    pub sh_b: BufferRef,
}
