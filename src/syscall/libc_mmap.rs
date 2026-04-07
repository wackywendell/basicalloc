//! mmap wrapper using the libc crate.
//!
//! This provides the same `mmap` interface as the inline assembly
//! implementations in [`super::asm`], but delegates to `libc::mmap`
//! instead of making syscalls directly. This works on any platform
//! that libc supports, not just the specific arch+OS combinations
//! that have hand-written assembly.
//!
//! This module is only compiled when the `use_libc` feature is enabled.

use super::MmapError;

/// Call mmap via the libc crate.
///
/// This is a thin wrapper that translates libc's return conventions
/// (MAP_FAILED + errno) into our `Result<*mut u8, MmapError>`.
///
/// # Safety
///
/// The caller must ensure the arguments form a valid mmap request.
/// See `mmap(2)` for details on each parameter.
pub unsafe fn mmap(
    addr: *mut u8,
    len: usize,
    prot: u64,
    flags: u64,
    fd: u64,
    offset: i64,
) -> Result<*mut u8, MmapError> {
    let ptr = libc::mmap(
        addr as *mut libc::c_void,
        len,
        prot as i32,
        flags as i32,
        fd as i32,
        offset as libc::off_t,
    );

    if ptr == libc::MAP_FAILED {
        return Err(MmapError::from_code(errno::errno().0 as i64));
    }

    Ok(ptr as *mut u8)
}
