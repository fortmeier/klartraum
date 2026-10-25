// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Renders a Gaussian splatting scene in a window, the Rust counterpart of
//! `examples/gaussian_splatting_example.cpp` (raster backend).
//!
//! Options:
//!   --frames N                 close after N frames and print the GPU timing
//!   --backend raster           the rendering backend (only `raster` is ported)
//!   --file PATH                load a specific .spz scene instead of the lantern
//!   --camera-position X Y Z    world-space eye position, looking at the origin
//!   --flip-y                   mirror the scene given with --file across the Y axis
//!
//! Run from the repository root (the default scene is `./data/lantern.spz`).

use std::rc::Rc;

use anyhow::{Result, bail};
use klartraum::computegraph::{CameraUbo, ComputeGraphElement, ImageViewSrc};
use klartraum::gaussian_data::GaussianDataStandard;
use klartraum::gaussian_splatting_types::GsplatConfig;
use klartraum::glm::Vec3;
use klartraum::interface_camera::{InterfaceCameraOrbit, UpDirection};
use klartraum::vulkan_gaussian_splatting_raster::VulkanGaussianSplattingRaster;
use klartraum::window_frontend::WindowFrontend;

struct Options {
    max_frames: i64,
    spz_file: String,
    camera_position: [f32; 3],
    flip_y: bool,
}

fn parse_args() -> Result<Options> {
    let mut options = Options {
        max_frames: -1,
        spz_file: "./data/lantern.spz".into(),
        camera_position: [0.55, 0.48, 0.69],
        flip_y: false,
    };
    let mut file_given = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" if i + 1 < args.len() => {
                i += 1;
                options.max_frames = args[i].parse().unwrap_or(-1);
            }
            "--backend" if i + 1 < args.len() => {
                i += 1;
                match args[i].as_str() {
                    "raster" => {}
                    "compute" => bail!("the compute backend is not ported to Rust yet; use --backend raster"),
                    other => bail!("Unknown --backend value '{other}' (expected 'compute' or 'raster')"),
                }
            }
            "--file" if i + 1 < args.len() => {
                i += 1;
                options.spz_file = args[i].clone();
                file_given = true;
            }
            "--camera-position" => {
                if i + 3 >= args.len() {
                    bail!("--camera-position requires X Y Z");
                }
                for k in 0..3 {
                    options.camera_position[k] = args[i + 1 + k]
                        .parse()
                        .map_err(|_| anyhow::anyhow!("Invalid --camera-position (expected three numbers)"))?;
                }
                i += 3;
            }
            "--flip-y" => options.flip_y = true,
            _ => {}
        }
        i += 1;
    }
    // The lantern capture is stored with Y pointing down.
    if !file_given {
        options.flip_y = true;
    }
    Ok(options)
}

fn main() -> Result<()> {
    let options = parse_args()?;
    let [x, y, z] = options.camera_position;
    let distance = ((x * x + y * y + z * z) as f64).sqrt();
    if !distance.is_finite() || distance <= 0.0 {
        bail!("--camera-position must be finite and different from the origin");
    }

    print!("Gaussian Splatting example (backend: raster)");
    if options.max_frames > 0 {
        print!(" (closing after {} frames)", options.max_frames);
    }
    println!();
    println!("Loading scene: {}", options.spz_file);
    if options.flip_y {
        println!("Flipping scene Y axis");
    }
    println!("Camera position: {x} {y} {z} (looking at 0 0 0)");

    let mut frontend = WindowFrontend::new()?;
    let engine = frontend.engine();
    if options.max_frames > 0 {
        engine.enable_profiling();
    }

    // Loaded once; the graph builder below reuses it on every rebuild.
    let model = Rc::new(GaussianDataStandard::from_spz(
        engine.vulkan_context(),
        &options.spz_file,
        options.flip_y,
    )?);

    // Everything tied to the swapchain is created in the graph builder, which the
    // engine runs now and again after each window resize.
    engine.set_graph_builder(Some(Box::new(move |e| {
        let vc = e.vulkan_context();
        let image_view_src = ImageViewSrc::for_swap_chain(vc);
        let camera_ubo = CameraUbo::new();
        camera_ubo.set_name("CameraUBO");
        let splatting = VulkanGaussianSplattingRaster::new(
            vc,
            image_view_src,
            camera_ubo.clone(),
            model.buffers(),
            GsplatConfig::default(),
        )?;
        e.add(splatting)?;
        e.set_camera_ubo(Some(camera_ubo));
        Ok(())
    })))?;

    let mut orbit = InterfaceCameraOrbit::new(UpDirection::Y);
    orbit.set_position(Vec3::ZERO);
    orbit.set_azimuth(z.atan2(x) as f64);
    orbit.set_elevation(-(y as f64 / distance).asin());
    orbit.set_distance(distance);
    engine.set_interface_camera(Box::new(orbit));

    frontend.run(options.max_frames)?;

    if options.max_frames > 0 {
        println!("\n--- GPU timing (mean over {} frames) ---", options.max_frames);
        for (name, ms) in frontend.engine().profiling_results()? {
            println!("  {name}: {ms} ms");
        }
    }
    Ok(())
}
