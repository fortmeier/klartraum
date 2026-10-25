// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Compiles the GLSL shaders the Rust engine uses from the repository's `shaders/`
//! directory with `glslc` (as the CMake target `Shaders` does) and generates a table
//! that embeds the SPIR-V in the library, keyed by the path the C++ engine reads it
//! from, e.g. `shaders/gsplat/gsplat_raster.vert.spv`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The shader directories (relative to `shaders/`) compiled for the Rust engine.
const SHADER_DIRS: &[&str] = &["", "gsplat"];
const STAGES: &[&str] = &["comp", "vert", "frag"];

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_root = manifest_dir
        .join("../../shaders")
        .canonicalize()
        .expect("shaders/ not found");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| {
        std::env::var("VULKAN_SDK")
            .map(|sdk| Path::new(&sdk).join("bin/glslc").to_string_lossy().into_owned())
            .unwrap_or_else(|_| "glslc".into())
    });
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let mut entries = Vec::new();
    for dir in SHADER_DIRS {
        let source_dir = shader_root.join(dir);
        println!("cargo:rerun-if-changed={}", source_dir.display());
        let mut files: Vec<PathBuf> = std::fs::read_dir(&source_dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        for source in files {
            let extension = source.extension().and_then(|e| e.to_str()).unwrap_or("");
            // Included files: a change recompiles the directory through rerun-if-changed.
            if extension == "glsl" {
                println!("cargo:rerun-if-changed={}", source.display());
                continue;
            }
            if !STAGES.contains(&extension) {
                continue;
            }
            println!("cargo:rerun-if-changed={}", source.display());
            let name = source.file_name().unwrap().to_str().unwrap();
            let key = if dir.is_empty() {
                format!("shaders/{name}.spv")
            } else {
                format!("shaders/{dir}/{name}.spv")
            };
            let spirv = out_dir.join(&key);
            std::fs::create_dir_all(spirv.parent().unwrap()).unwrap();
            let status = Command::new(&glslc)
                .arg("-o")
                .arg(&spirv)
                .arg(&source)
                .status()
                .unwrap_or_else(|e| panic!("cannot run glslc ({glslc}): {e}; set GLSLC or VULKAN_SDK"));
            assert!(status.success(), "glslc failed for {}", source.display());
            entries.push((key, spirv));
        }
    }

    let mut table = String::from("/// The compiled shaders, keyed by the path the C++ engine loads them from.\n");
    table.push_str("pub static EMBEDDED_SHADERS: &[(&str, &[u8])] = &[\n");
    for (key, spirv) in &entries {
        table.push_str(&format!(
            "    ({key:?}, include_bytes!({:?})),\n",
            spirv.to_str().unwrap()
        ));
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("embedded_shaders.rs"), table).unwrap();
}
