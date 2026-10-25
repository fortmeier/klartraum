// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Message framing on a stream: a 4-byte little-endian length, then the FlatBuffer.

use std::io::{Read, Write};

use anyhow::{Context, Result, ensure};

/// Messages larger than this are rejected; graph descriptions with inline SPIR-V stay far below.
pub const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

/// Writes one framed message.
pub fn write_message(stream: &mut impl Write, message: &[u8]) -> Result<()> {
    ensure!(
        message.len() <= MAX_MESSAGE_SIZE,
        "message of {} bytes is too large",
        message.len()
    );
    stream.write_all(&(message.len() as u32).to_le_bytes())?;
    stream.write_all(message)?;
    stream.flush()?;
    Ok(())
}

/// Reads one framed message, or None when the stream ended between messages.
pub fn read_message(stream: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut length = [0u8; 4];
    match stream.read_exact(&mut length) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let length = u32::from_le_bytes(length) as usize;
    ensure!(length <= MAX_MESSAGE_SIZE, "message of {length} bytes is too large");
    let mut message = vec![0u8; length];
    stream.read_exact(&mut message).context("message ended early")?;
    Ok(Some(message))
}
