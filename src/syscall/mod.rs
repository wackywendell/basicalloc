//! Syscall interface for memory management.
//!
//! This module provides a platform-independent interface for requesting memory
//! from the OS via `mmap`. It abstracts over two implementation strategies:
//!
//! - **`asm`**: Direct syscalls using inline assembly. Available on supported
//!   architecture + OS combinations (x86_64 and aarch64, Linux and macOS),
//!   regardless of whether `use_libc` is enabled.
//!
//! - **`libc`**: Wrapper around `libc::mmap`. Available when the `use_libc`
//!   feature is enabled. Works on any platform that libc supports.
//!
//! The top-level [`mmap`] function is cfg-selected to use one of these:
//! `libc` when the feature is on, `asm` otherwise.
//!
//! ## Usage
//!
//! Most code should use [`mmap`] and [`page_size`] directly:
//!
//! ```ignore
//! use crate::syscall;
//!
//! let ptr = unsafe {
//!     syscall::mmap(
//!         core::ptr::null_mut(),
//!         syscall::page_size(),
//!         syscall::consts::PROT_READ | syscall::consts::PROT_WRITE,
//!         syscall::consts::MAP_ANON | syscall::consts::MAP_PRIVATE,
//!         0, 0,
//!     )
//! };
//! ```

pub mod consts;

// The asm module is available on supported platforms regardless of use_libc,
// so both implementations can be accessed directly if needed.
#[cfg(any(
    all(target_arch = "x86_64", target_os = "linux"),
    all(target_arch = "x86_64", target_os = "macos"),
    all(target_arch = "aarch64", target_os = "linux"),
    all(target_arch = "aarch64", target_os = "macos"),
))]
pub mod asm;

// The libc implementation of mmap
#[cfg(feature = "use_libc")]
pub mod libc_mmap;

// The default `mmap`: use libc when the feature is on, asm otherwise.
#[cfg(not(feature = "use_libc"))]
pub use self::asm::mmap;
#[cfg(feature = "use_libc")]
pub use self::libc_mmap::mmap;

/// Error returned by a failed mmap syscall.
///
/// Contains the error code from the kernel — typically an errno value
/// (e.g. ENOMEM, EINVAL). The meaning is the same regardless of whether
/// the asm or libc path was used.
#[derive(Debug)]
pub struct MmapError {
    code: i64,
}

impl MmapError {
    /// Create an MmapError from a raw error code.
    pub(crate) fn from_code(code: i64) -> Self {
        MmapError { code }
    }

    /// The raw error code (typically an errno value).
    pub fn code(&self) -> i64 {
        self.code
    }
}

/// Returns the OS page size for the current target.
///
/// All mmap allocations are rounded up to a multiple of this value,
/// since the kernel allocates memory in whole pages.
///
/// When `use_libc` is enabled, this queries the OS at runtime via sysconf,
/// which is the most robust approach. Without `use_libc`, we use well-known
/// compile-time constants per architecture + OS (4K for most targets, 16K
/// for Apple Silicon).
pub fn page_size() -> usize {
    #[cfg(feature = "use_libc")]
    {
        sysconf::page::pagesize()
    }
    #[cfg(not(feature = "use_libc"))]
    {
        // Apple Silicon uses 16K pages; everything else we support uses 4K.
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            16384
        }
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        {
            4096
        }
    }
}
