// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Contents of the camera uniform buffer.

use bytemuck::{Pod, Zeroable};

use crate::glm::{Mat4, Vec4};

/// Contents of the camera uniform buffer (`CameraUbo`): model, view and projection
/// matrices and the camera position, in std140 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct CameraMVP {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
    /// World-space camera position (xyz); w unused. A vec4, so the std140 layouts of both sides agree.
    pub camera_world_pos: Vec4,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_std140() {
        assert_eq!(std::mem::size_of::<CameraMVP>(), 208);
        assert_eq!(std::mem::offset_of!(CameraMVP, proj), 128);
        assert_eq!(std::mem::offset_of!(CameraMVP, camera_world_pos), 192);
    }
}
