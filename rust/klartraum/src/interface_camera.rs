// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Cameras that the engine updates every frame, and the orbit camera.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::Instant;

use ash::vk;

use crate::camera::CameraMVP;
use crate::events::{Action, Event, Key, MouseButton};
use crate::glm::{self, Mat4, Vec3, vec3};

/// A camera the engine updates every frame.
///
/// Every frame, `KlartraumEngine::step()` passes the queued input events to
/// [`on_event`](Self::on_event) and then calls [`update`](Self::update), which writes
/// the camera matrices for a target of `extent` into the camera uniform buffer data.
pub trait InterfaceCamera {
    fn update(&mut self, mvp: &mut CameraMVP, extent: vk::Extent2D);
    fn on_event(&mut self, _event: &Event) {}
}

/// Which world axis points up for an [`InterfaceCameraOrbit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpDirection {
    Z,
    Y,
}

/// Seconds since the first call; the orbit camera scales mouse drags by it, as the C++ camera does.
fn seconds_since_start() -> f32 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f32()
}

/// A camera that orbits a pivot: dragging with the left button turns it, scrolling
/// changes the distance, W/A/S/D move the pivot and Space resets the view.
#[derive(Clone, Debug)]
pub struct InterfaceCameraOrbit {
    azimuth: f64,
    elevation: f64,
    distance: f64,
    /// The world is translated by this before the rotation about the origin, so `-pivot` orbits `pivot`.
    position: Vec3,
    near_plane: f32,
    far_plane: f32,
    /// Overrides the target's aspect ratio for the projection.
    projection_aspect_ratio: Option<f32>,
    left_button_down: bool,
    up: UpDirection,
    pressed_keys: HashSet<Key>,
}

impl Default for InterfaceCameraOrbit {
    fn default() -> Self {
        Self::new(UpDirection::Z)
    }
}

impl InterfaceCameraOrbit {
    pub fn new(up: UpDirection) -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 2.0,
            position: Vec3::ZERO,
            near_plane: 0.1,
            far_plane: 1000.0,
            projection_aspect_ratio: None,
            left_button_down: false,
            up,
            pressed_keys: HashSet::new(),
        }
    }

    pub fn set_up_direction(&mut self, up: UpDirection) {
        self.up = up;
    }

    pub fn set_distance(&mut self, distance: f64) {
        self.distance = distance;
    }

    pub fn set_azimuth(&mut self, azimuth: f64) {
        self.azimuth = azimuth;
    }

    pub fn set_elevation(&mut self, elevation: f64) {
        self.elevation = elevation;
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn set_near_plane(&mut self, near_plane: f32) {
        self.near_plane = near_plane;
    }

    pub fn set_far_plane(&mut self, far_plane: f32) {
        self.far_plane = far_plane;
    }

    pub fn set_projection_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.projection_aspect_ratio = Some(aspect_ratio);
    }

    pub fn use_swap_chain_aspect_ratio(&mut self) {
        self.projection_aspect_ratio = None;
    }

    pub fn distance(&self) -> f64 {
        self.distance
    }

    pub fn azimuth(&self) -> f64 {
        self.azimuth
    }

    pub fn elevation(&self) -> f64 {
        self.elevation
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn near_plane(&self) -> f32 {
        self.near_plane
    }

    pub fn far_plane(&self) -> f32 {
        self.far_plane
    }

    /// Moves the pivot along the view for the held W/A/S/D keys.
    pub fn update_position(&mut self) {
        let move_speed = (0.01 * self.distance) as f32;
        let (az, el) = (self.azimuth, self.elevation);
        let (forward, up_vec) = match self.up {
            UpDirection::Y => (
                vec3(
                    (-az.cos() * el.cos()) as f32,
                    (-az.sin() * el.cos()) as f32,
                    el.sin() as f32,
                ),
                vec3(0.0, 1.0, 0.0),
            ),
            UpDirection::Z => (
                vec3(
                    (-az.cos() * el.cos()) as f32,
                    el.sin() as f32,
                    (-az.sin() * el.cos()) as f32,
                ),
                vec3(0.0, 0.0, 1.0),
            ),
        };
        let right = glm::normalize(glm::cross(forward, up_vec));

        if self.pressed_keys.contains(&Key::W) {
            self.position = self.position + forward * move_speed;
        }
        if self.pressed_keys.contains(&Key::S) {
            self.position = self.position - forward * move_speed;
        }
        if self.pressed_keys.contains(&Key::A) {
            self.position = self.position + right * move_speed;
        }
        if self.pressed_keys.contains(&Key::D) {
            self.position = self.position - right * move_speed;
        }
    }
}

impl InterfaceCamera for InterfaceCameraOrbit {
    fn update(&mut self, mvp: &mut CameraMVP, extent: vk::Extent2D) {
        self.update_position();

        mvp.model = Mat4::IDENTITY;

        let identity = Mat4::IDENTITY;
        let eye = vec3(self.distance as f32, 0.0, 0.0);
        mvp.view = match self.up {
            UpDirection::Y => {
                glm::look_at(eye, Vec3::ZERO, vec3(0.0, 1.0, 0.0))
                    * glm::rotate(identity, self.elevation as f32, vec3(0.0, 0.0, 1.0))
                    * glm::rotate(identity, self.azimuth as f32, vec3(0.0, 1.0, 0.0))
                    * glm::translate(identity, self.position)
            }
            UpDirection::Z => {
                glm::look_at(eye, Vec3::ZERO, vec3(0.0, 0.0, 1.0))
                    * glm::rotate(identity, self.elevation as f32, vec3(0.0, 1.0, 0.0))
                    * glm::rotate(identity, self.azimuth as f32, vec3(0.0, 0.0, 1.0))
            }
        };

        let aspect_ratio = self
            .projection_aspect_ratio
            .unwrap_or(extent.width as f32 / extent.height as f32);
        mvp.proj = glm::perspective(glm::radians(45.0), aspect_ratio, self.near_plane, self.far_plane);

        // Vulkan's clip space has Y pointing down, OpenGL's up.
        mvp.proj[1][1] *= -1.0;

        // The world-space camera position, which the SH evaluation needs for the view direction.
        mvp.camera_world_pos = glm::inverse(mvp.model * mvp.view)[3];
    }

    fn on_event(&mut self, event: &Event) {
        let time = seconds_since_start();
        match *event {
            Event::MouseMove { dx, dy, .. } => {
                if self.left_button_down {
                    self.azimuth += (dx as f32 / 1000.0 * time) as f64;
                    self.elevation += (dy as f32 / 1000.0 * time) as f64;

                    let max_elevation = glm::radians(89.0) as f64;
                    self.elevation = self.elevation.clamp(-max_elevation, max_elevation);

                    let full_turn = glm::radians(360.0) as f64;
                    if self.azimuth > full_turn {
                        self.azimuth -= full_turn;
                    }
                    if self.azimuth < 0.0 {
                        self.azimuth += full_turn;
                    }
                }
            }
            Event::MouseButton {
                button: MouseButton::Left,
                action,
            } => {
                self.left_button_down = action == Action::Press;
            }
            Event::MouseButton { .. } => {}
            Event::MouseScroll { y, .. } => {
                self.distance = (self.distance + y / 10.0).clamp(0.1, 10000.0);
            }
            Event::Key { key: Key::Escape, .. } => {
                // As the C++ camera: Escape ends the program.
                std::process::exit(0);
            }
            Event::Key { key: Key::Space, .. } => {
                self.azimuth = 0.0;
                self.elevation = 0.0;
                self.distance = 2.0;
                self.position = Vec3::ZERO;
            }
            Event::Key { key, action } => {
                if action == Action::Press {
                    self.pressed_keys.insert(key);
                } else {
                    self.pressed_keys.remove(&key);
                }
            }
        }
    }
}
