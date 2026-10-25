// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`VulkanGaussianSplattingRaster`], the Gaussian splatting backend that sorts once and rasterizes.

use std::collections::BTreeMap;
use std::rc::Rc;

use anyhow::{Result, bail};
use ash::vk;

use crate::computegraph::{
    BufferElement, BufferToGraphicsBarrier, CameraUbo, ComputeGraphElement, DrawIndirectCommand, ElementBase,
    ElementPtr, ImageViewSrcInterface, RenderPass, downcast,
};
use crate::gaussian_splat_rasterizer::{GaussianSplatRasterPushConstants, GaussianSplatRasterizer};
use crate::gaussian_splatting_types::{
    CullAndCompactPushConstants, GaussianCullAndCompact, GaussianRasterProject, GaussianSoABuffers, GsplatConfig,
    RadixSort, RasterProjectPushConstants, SortPushConstants,
};
use crate::vulkan_context::VulkanContext;

/// Gaussian splatting that sorts the Gaussians once and draws them with hardware rasterization.
///
/// It is a group of graph elements:
/// - `GaussianCullAndCompact`: culling, depth keys and compaction in splat-index order
///   (`gsplat_cull_and_compact_{count,scan,scatter}.comp`);
/// - `RasterProject`: per-splat 2D attributes (covariance, colour from SH), once per splat;
/// - a radix sort of the depth keys over the visible count;
/// - a `BufferToGraphicsBarrier` ordering the indirect draw after the compute work;
/// - a `RenderPass` with a [`GaussianSplatRasterizer`] drawing instanced quads with
///   premultiplied "over" blending onto the target's contents.
///
/// The render pass is the group's output element, through which the graph reaches all others.
pub struct VulkanGaussianSplattingRaster {
    base: ElementBase,
    /// The group's output: the render pass.
    output_elements: BTreeMap<i32, ElementPtr>,
    /// Indirect draw arguments; the cull writes the instance count, setup seeds the vertex count.
    draw_args: Rc<BufferElement<DrawIndirectCommand>>,
    cull_and_compact: Rc<GaussianCullAndCompact>,
    project: Rc<GaussianRasterProject>,
    sort: Rc<RadixSort>,
    keys_a: Rc<BufferElement<u32>>,
    indices_a: Rc<BufferElement<u32>>,
    total_count: Rc<BufferElement<u32>>,
    splat_2d: Rc<BufferElement<f32>>,
    render_pass: Rc<RenderPass>,
    rasterizer: Rc<GaussianSplatRasterizer>,
}

impl VulkanGaussianSplattingRaster {
    pub fn new(
        context: &VulkanContext,
        image_view_src: ElementPtr,
        camera_ubo: Rc<CameraUbo>,
        buffers: &GaussianSoABuffers,
        config: GsplatConfig,
    ) -> Result<Rc<Self>> {
        if config.output_depth {
            bail!("VulkanGaussianSplattingRaster: the depth output is not ported to Rust yet");
        }
        let Some(target) = image_view_src.as_image_view_src() else {
            bail!("VulkanGaussianSplattingRaster: input 0 must be an ImageViewSrc");
        };
        let extent = target.image_extent(0);
        let n = buffers.count;
        let storage_dst = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let u32_buffer = |count: u32, name: &str| {
            let b = BufferElement::<u32>::with_usage(context, count, storage_dst, local);
            b.set_name(name);
            b
        };

        // --- Stage A: cull + depth key + compaction ---
        // Culling generates depth keys and splat indices; the radix sort alternates between A and B.
        let keys_a = u32_buffer(n, "RasterKeysA");
        let indices_a = u32_buffer(n, "RasterIndicesA");
        let keys_b = u32_buffer(n, "RasterKeysB");
        let indices_b = u32_buffer(n, "RasterIndicesB");

        // The cull writes the visible count straight into the indirect draw arguments.
        let draw_args = BufferElement::<DrawIndirectCommand>::with_usage(
            context,
            1,
            vk::BufferUsageFlags::INDIRECT_BUFFER | storage_dst,
            local,
        );
        draw_args.set_name("RasterDrawArgs");

        // Three sub-shaders with one dispatch size: count visible splats per workgroup,
        // scan the counts into offsets (and write the total into the draw arguments and
        // the sort's count buffer), scatter keys and indices. Compacting in splat-index
        // order makes the stable sort break key ties by index, so every run draws the same image.
        let num_cull_wgs = n.div_ceil(256);
        let workgroup_offsets = u32_buffer(num_cull_wgs, "RasterCullAndCompactWorkgroupOffsets");
        let cull_and_compact = GaussianCullAndCompact::with_shaders(&[
            "shaders/gsplat/gsplat_cull_and_compact_count.comp.spv",
            "shaders/gsplat/gsplat_cull_and_compact_scan.comp.spv",
            "shaders/gsplat/gsplat_cull_and_compact_scatter.comp.spv",
        ]);
        cull_and_compact.set_name("GaussianCullAndCompact");
        buffers.pos.connect_to(&*cull_and_compact, 0)?;
        cull_and_compact.set_input(camera_ubo.clone(), 1)?;
        cull_and_compact.set_input(keys_a.clone(), 2)?;
        cull_and_compact.set_input(indices_a.clone(), 3)?;
        cull_and_compact.set_input(draw_args.clone(), 4)?;
        buffers.col_alpha.connect_to(&*cull_and_compact, 5)?; // alpha for the opacity cull
        cull_and_compact.set_input(workgroup_offsets.clone(), 7)?;
        cull_and_compact.set_group_count_x(num_cull_wgs);
        cull_and_compact.set_push_constants(vec![CullAndCompactPushConstants {
            num_splats: n,
            frustum_dilation: 0.1,
            alpha_threshold: config.alpha_cull_threshold,
        }]);

        // --- Stage A2: per-splat 2D attributes, 16 floats per splat, keyed by splat id ---
        let splat_2d = BufferElement::<f32>::with_usage(context, 16 * n, vk::BufferUsageFlags::STORAGE_BUFFER, local);
        splat_2d.set_name("RasterSplat2D");
        let project = GaussianRasterProject::new("shaders/gsplat/gsplat_raster_project.comp.spv");
        project.set_name("RasterProject");
        buffers.pos.connect_to(&*project, 0)?;
        buffers.rot.connect_to(&*project, 1)?;
        buffers.scale.connect_to(&*project, 2)?;
        buffers.col_alpha.connect_to(&*project, 3)?;
        buffers.sh_r.connect_to(&*project, 4)?;
        buffers.sh_g.connect_to(&*project, 5)?;
        buffers.sh_b.connect_to(&*project, 6)?;
        project.set_input(camera_ubo.clone(), 7)?;
        project.set_input(splat_2d.clone(), 8)?;
        project.set_group_count_x(n.div_ceil(256));
        project.set_push_constants(vec![RasterProjectPushConstants {
            num_splats: n,
            screen_width: extent.width as f32,
            screen_height: extent.height as f32,
            splat_scale: 3.0,
            sh_degree: config.sh_degree,
        }]);

        // --- Stage B: radix sort over the visible count. 8 passes of 4 bits sort the
        // whole 32-bit key; the even pass count leaves the result in the A buffers. The
        // group count stays fixed (consistent histogram stride); the count buffer
        // gives the number of items. ---
        let num_bins = 16;
        let threads_per_group = 256;
        let num_sort_wgs = config.num_sort_wgs_cap.min(n / threads_per_group + 1).max(1);
        let scratch_hist = u32_buffer(num_bins * num_sort_wgs, "RasterSortScratchHist");
        let scratch_counts = u32_buffer(num_bins, "RasterSortScratchCounts");
        let scratch_offsets = u32_buffer(num_sort_wgs + 1, "RasterSortScratchOffsets");
        // Holds the visible count, fully written each frame by the cull's scan shader.
        let total_count = u32_buffer(1, "RasterSortTotalCount");
        cull_and_compact.set_input(total_count.clone(), 6)?;

        let sort = RadixSort::with_shaders(&[
            "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
            "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
            "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv",
        ]);
        sort.set_name("RasterRadixSort");
        sort.set_input_slot(cull_and_compact.clone(), 0, 2)?; // keys A
        sort.set_input_slot(cull_and_compact.clone(), 1, 3)?; // indices A
        sort.set_input(keys_b.clone(), 2)?;
        sort.set_input(indices_b.clone(), 3)?;
        sort.add_scratch_buffer_element(scratch_counts, true);
        sort.add_scratch_buffer_element(scratch_offsets, true);
        sort.add_scratch_buffer_element(total_count.clone(), false);
        sort.add_scratch_buffer_element(scratch_hist, true);
        sort.set_group_count_x(num_sort_wgs);
        sort.set_push_constants(
            (0..32 / 4)
                .map(|pass| SortPushConstants {
                    pass,
                    num_elements: n,
                    num_bins,
                    use_count_buffer: 1,
                })
                .collect(),
        );

        // --- Order the compute writes before the render pass reads them indirectly / in the vertex shader ---
        let barrier = BufferToGraphicsBarrier::new();
        barrier.set_name("RasterBarrier");
        barrier.add_buffer(indices_a.clone());
        barrier.add_buffer(draw_args.clone());
        barrier.add_buffer(splat_2d.clone());
        barrier.set_input_slot(sort.clone(), 0, 1)?; // indices A, sorted
        barrier.set_input_slot(cull_and_compact.clone(), 1, 4)?; // draw arguments
        barrier.set_input_slot(project.clone(), 2, 8)?; // Splat2D

        // --- Stage C: instanced indirect draw, hardware rasterized ---
        let render_pass = RenderPass::new(context.swap_chain_image_format(), extent);
        render_pass.set_name("RasterRenderPass");
        // Splats are drawn over the target's contents (black unless its source was cleared otherwise).
        render_pass.set_load_existing(true);
        render_pass.set_input(image_view_src.clone(), 0)?;
        render_pass.set_input(camera_ubo.clone(), 1)?;

        // The vertex shader reads only Splat2D (binding 0) through the sorted indices (binding 1).
        let rasterizer = GaussianSplatRasterizer::new(vec![splat_2d.clone(), indices_a.clone()], draw_args.clone());
        rasterizer.set_push_constants(GaussianSplatRasterPushConstants {
            resolution: [extent.width as f32, extent.height as f32],
            focal: [1000.0, 1000.0],
            // 3 sigma covers ~99.7% of each Gaussian; the fragment shader evaluates the exact falloff.
            splat_scale: 3.0,
            sh_degree: 0,
            num_splats: n,
        });
        render_pass.add_draw_component(rasterizer.clone());
        render_pass.add_compute_dependency(barrier);

        let element = Self {
            base: ElementBase::new(),
            output_elements: BTreeMap::from([(0, render_pass.clone() as ElementPtr)]),
            draw_args,
            cull_and_compact,
            project,
            sort,
            keys_a,
            indices_a,
            total_count,
            splat_2d,
            render_pass,
            rasterizer,
        };
        element.set_input(image_view_src, 0)?;
        element.set_input(camera_ubo, 1)?;
        Ok(Rc::new(element))
    }

    /// The camera uniform buffer at input 1.
    pub fn camera_ubo(&self) -> Result<Rc<CameraUbo>> {
        downcast::<CameraUbo>(&self.input_element(1)?).ok_or_else(|| anyhow::anyhow!("input 1 is not a CameraUbo"))
    }

    pub fn set_push_constants(&self, push_constants: GaussianSplatRasterPushConstants) {
        self.rasterizer.set_push_constants(push_constants);
    }

    /// Internal elements, for tests and inspection of the stages.
    pub fn cull_and_compact(&self) -> &Rc<GaussianCullAndCompact> {
        &self.cull_and_compact
    }

    pub fn project(&self) -> &Rc<GaussianRasterProject> {
        &self.project
    }

    pub fn sort(&self) -> &Rc<RadixSort> {
        &self.sort
    }

    pub fn render_pass(&self) -> &Rc<RenderPass> {
        &self.render_pass
    }

    /// Sorted depth keys (valid over the visible count after a frame).
    pub fn keys(&self) -> &Rc<BufferElement<u32>> {
        &self.keys_a
    }

    /// Splat indices in drawing order (valid over the visible count after a frame).
    pub fn sorted_indices(&self) -> &Rc<BufferElement<u32>> {
        &self.indices_a
    }

    /// The visible count the cull wrote.
    pub fn visible_count(&self) -> &Rc<BufferElement<u32>> {
        &self.total_count
    }

    pub fn draw_args(&self) -> &Rc<BufferElement<DrawIndirectCommand>> {
        &self.draw_args
    }

    pub fn splat_2d(&self) -> &Rc<BufferElement<f32>> {
        &self.splat_2d
    }
}

impl ComputeGraphElement for VulkanGaussianSplattingRaster {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn type_name(&self) -> &'static str {
        "GaussianSplattingRaster"
    }

    fn check_input(&self, input: &ElementPtr, index: i32) -> Result<()> {
        match index {
            0 if input.as_image_view_src().is_none() => {
                bail!("VulkanGaussianSplattingRaster: input 0 must be ImageViewSrc")
            }
            1 if downcast::<CameraUbo>(input).is_none() => {
                bail!("VulkanGaussianSplattingRaster: input 1 must be CameraUbo")
            }
            0 | 1 => Ok(()),
            _ => bail!("VulkanGaussianSplattingRaster: input index out of range"),
        }
    }

    // A group: the graph reaches the other elements through the output.
    fn inputs(&self) -> BTreeMap<i32, ElementPtr> {
        self.output_elements.clone()
    }

    fn setup(&self, context: &VulkanContext, number_paths: u32) -> Result<()> {
        // The draw arguments exist only now. The vertex count (4, a triangle-strip quad)
        // is constant, so it is written once; the cull writes the instance count every frame.
        context.submit_immediate(|cmd| {
            for path_id in 0..number_paths {
                unsafe {
                    context.vk().cmd_fill_buffer(
                        cmd,
                        self.draw_args.buffer(path_id).buffer(),
                        std::mem::offset_of!(DrawIndirectCommand, vertex_count) as vk::DeviceSize,
                        4,
                        4,
                    )
                };
            }
        })?;
        self.base.set_initialized(true);
        Ok(())
    }
}

impl ImageViewSrcInterface for VulkanGaussianSplattingRaster {
    fn image_view(&self, path_id: u32) -> vk::ImageView {
        self.render_pass.image_view(path_id)
    }

    fn image(&self, path_id: u32) -> vk::Image {
        self.render_pass.image(path_id)
    }

    fn image_extent(&self, path_id: u32) -> vk::Extent2D {
        self.render_pass.image_extent(path_id)
    }
}
