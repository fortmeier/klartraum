// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Serves remote compute graphs on a Unix socket:
//!
//!     klartraum_ipc_server SOCKET_PATH
//!
//! It runs until a client sends a Shutdown request.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: klartraum_ipc_server SOCKET_PATH"))?;
    klartraum_ipc::server::listen(&path)
}
