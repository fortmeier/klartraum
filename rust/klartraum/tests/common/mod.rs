// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Helpers shared by the integration tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, Once};

/// The repository root: the tests read `data/` and write `build/TestingOutput/` relative to it.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Makes the repository root the working directory, as for the C++ tests.
///
/// Cargo runs integration tests in the package directory; every test that reads
/// `data/` or writes `build/TestingOutput/` calls this first.
pub fn enter_repo_root() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_current_dir(repo_root()).unwrap());
}

/// The lantern scene the tests render, relative to the repository root.
pub const LANTERN_PATH: &str = "data/lantern.spz";

/// Where the C++ test `RustReference.*` (tests/test_rust_reference.cpp) writes its reference data.
pub const REFERENCE_DIR: &str = "build/TestingOutput/rust_reference";

/// Reads a reference file, or returns None (and says why) when the C++ tests have not written it yet.
pub fn read_reference(name: &str) -> Option<Vec<u8>> {
    enter_repo_root();
    let path = Path::new(REFERENCE_DIR).join(name);
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!(
                "skipped: {} not found; run `./build/klartraum_tests --gtest_filter='RustReference.*'` first",
                path.display()
            );
            None
        }
    }
}

/// Serializes the tests that use the GPU, so they do not compete for the device.
pub fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
