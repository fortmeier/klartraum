// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Remote compute graphs: a client (the C++ engine) describes a compute graph in a
//! FlatBuffers message (schema `ipc/klartraum_ipc.fbs`), this crate's server builds
//! it with the Rust engine, compiles it once and runs it on request.
//!
//! Tensor data never travels in the messages: every buffer is a POSIX shared-memory
//! object both processes map, and the server imports the mapping as Vulkan memory
//! (`VK_EXT_external_memory_host`), so its shaders read the inputs and write the
//! results in the very pages the client reads and writes.

#[rustfmt::skip]
#[allow(clippy::all, unused_imports, dead_code, mismatched_lifetime_syntaxes, unsafe_op_in_unsafe_fn)]
mod klartraum_ipc_generated;

/// The generated FlatBuffers types of `ipc/klartraum_ipc.fbs`.
pub use klartraum_ipc_generated::klartraum::ipc as schema;

pub mod framing;
pub mod shared_memory;
