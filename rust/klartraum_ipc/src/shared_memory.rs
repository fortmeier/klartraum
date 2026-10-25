// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! POSIX shared-memory objects mapped into this process (the counterpart of the C++
//! `klartraum::ipc::SharedMemoryRegion`).

use std::ffi::CString;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, bail};
use klartraum::computegraph::HostMemory;

/// Alignment of region sizes, as the C++ `SharedMemoryRegion::kRegionAlignment`.
pub const REGION_ALIGNMENT: usize = 64 * 1024;

/// A mapped POSIX shared-memory object.
///
/// A region made with [`create`](Self::create) owns the name and unlinks it when
/// dropped; one made with [`open`](Self::open) only unmaps.
pub struct SharedMemoryRegion {
    name: String,
    /// The start of the mapping.
    ptr: *mut u8,
    /// Bytes mapped.
    len: usize,
    /// Whether this region created the object and unlinks its name.
    owner: bool,
}

fn os_error(what: &str) -> anyhow::Error {
    anyhow::anyhow!("{what}: {}", std::io::Error::last_os_error())
}

impl SharedMemoryRegion {
    /// Maps the existing object `name` of `len` bytes.
    pub fn open(name: &str, len: usize) -> Result<Self> {
        let c_name = CString::new(name)?;
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(os_error(&format!("shm_open({name})")));
        }
        Self::map(name.to_owned(), fd, len, false)
    }

    /// Creates a new zero-filled object of at least `len` bytes (rounded up to
    /// [`REGION_ALIGNMENT`]) with a unique name, e.g. for tests.
    pub fn create(len: usize) -> Result<Self> {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let aligned = len.max(1).div_ceil(REGION_ALIGNMENT) * REGION_ALIGNMENT;
        // "/ktr" + pid + "_" + counter stays within the 31 characters macOS allows.
        let name = format!("/ktr{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::SeqCst));
        let c_name = CString::new(name.clone())?;
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_EXCL | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(os_error(&format!("shm_open({name})")));
        }
        if unsafe { libc::ftruncate(fd, aligned as libc::off_t) } != 0 {
            let error = os_error(&format!("ftruncate({name})"));
            unsafe {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr());
            }
            return Err(error);
        }
        let region = Self::map(name, fd, aligned, true);
        if region.is_err() {
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
        }
        region
    }

    fn map(name: String, fd: libc::c_int, len: usize, owner: bool) -> Result<Self> {
        if len == 0 {
            unsafe { libc::close(fd) };
            bail!("cannot map {name} with size 0");
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        unsafe { libc::close(fd) };
        if ptr == libc::MAP_FAILED {
            return Err(os_error(&format!("mmap({name})")));
        }
        Ok(Self {
            name,
            ptr: ptr as *mut u8,
            len,
            owner,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The mapped bytes. Another process may write them concurrently; read them only
    /// when the protocol says the other side is done.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl HostMemory for SharedMemoryRegion {
    fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for SharedMemoryRegion {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
            if self.owner {
                if let Ok(c_name) = CString::new(self.name.clone()) {
                    libc::shm_unlink(c_name.as_ptr());
                }
            }
        }
    }
}
