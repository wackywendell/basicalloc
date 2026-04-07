//! Raw mmap syscall implementations using inline assembly.
//!
//! Each submodule provides an `mmap` function for a specific architecture + OS
//! combination. This module re-exports the correct one for the current target.
//!
//! All implementations share the same function signature:
//! ```ignore
//! pub unsafe fn mmap(
//!     addr: *mut u8, len: usize, prot: u64, flags: u64, fd: u64, offset: i64,
//! ) -> Result<*mut u8, MmapError>
//! ```

mod x86_64_linux;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub use x86_64_linux::mmap;

mod x86_64_macos;
#[cfg(all(target_arch = "x86_64", target_os = "macos"))]
pub use x86_64_macos::mmap;

mod aarch64_linux;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub use aarch64_linux::mmap;

mod aarch64_macos;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use aarch64_macos::mmap;

#[cfg(test)]
mod tests {
    use super::mmap;
    use crate::syscall::consts;

    use test_log::test;

    #[test]
    fn test_mmap() {
        let ptr = unsafe {
            mmap(
                core::ptr::null_mut(),
                8,
                consts::PROT_WRITE | consts::PROT_READ,
                consts::MAP_ANON | consts::MAP_PRIVATE,
                0,
                0,
            )
        };

        assert!(ptr.is_ok(), "Error: {:?}", ptr.unwrap_err());
    }

    #[test]
    fn test_mmap_err() {
        let ptr = unsafe {
            mmap(
                core::ptr::null_mut(),
                8,
                consts::PROT_WRITE | consts::PROT_READ,
                // Without MAP_ANON, fd=0 should produce an error
                0,
                0,
                0,
            )
        };

        assert!(ptr.is_err());
    }
}
