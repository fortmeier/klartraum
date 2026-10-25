// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Rust port of the Klartraum engine: a Vulkan compute/rendering framework that
//! pre-records all GPU work into command buffers at setup time and submits them
//! unchanged every frame.
//!
//! The layers follow the C++ engine: [`vulkan_context`] owns the device and the
//! swapchain, [`computegraph`] compiles a DAG of elements into one command buffer
//! per path, [`engine`] drives the frame loop and the frontends open a window or
//! run headless.

pub mod gaussian_data;
pub mod spz;
