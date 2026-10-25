// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Reading files and shaders, and creating shader modules.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use ash::vk;

include!(concat!(env!("OUT_DIR"), "/embedded_shaders.rs"));

static ASSET_ROOT: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Sets the directory relative paths are resolved against when they are not found
/// in the working directory (see [`read_file`]).
pub fn set_asset_root(directory: impl Into<PathBuf>) {
    *ASSET_ROOT.lock().unwrap() = Some(directory.into());
}

/// The asset root: the one set with [`set_asset_root`], else `KLARTRAUM_ASSET_DIR`, if any.
pub fn asset_root() -> Option<PathBuf> {
    if let Some(root) = ASSET_ROOT.lock().unwrap().clone() {
        return Some(root);
    }
    std::env::var_os("KLARTRAUM_ASSET_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Reads a file. A relative path is looked up in the working directory, then in the
/// asset root, and last among the shaders compiled into the library, so a file next
/// to the caller overrides the one shipped with the build, as in the C++ engine.
pub fn read_file(filename: impl AsRef<Path>) -> Result<Vec<u8>> {
    let path = filename.as_ref();
    if let Ok(bytes) = std::fs::read(path) {
        return Ok(bytes);
    }
    if path.is_relative() {
        if let Some(root) = asset_root() {
            if let Ok(bytes) = std::fs::read(root.join(path)) {
                return Ok(bytes);
            }
        }
        let key = path.to_string_lossy().replace('\\', "/");
        if let Some((_, bytes)) = EMBEDDED_SHADERS.iter().find(|(name, _)| *name == key) {
            return Ok(bytes.to_vec());
        }
    }
    Err(anyhow!("failed to open file: {}", path.display()))
}

/// Creates a shader module from SPIR-V bytes.
pub fn create_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(code)).context("invalid SPIR-V")?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    Ok(unsafe { device.create_shader_module(&info, None) }.context("failed to create shader module")?)
}

/// Reads a shader with [`read_file`] and creates its module.
pub fn load_shader_module(device: &ash::Device, path: &str) -> Result<vk::ShaderModule> {
    let code = read_file(path)?;
    create_shader_module(device, &code).with_context(|| format!("shader {path}"))
}
