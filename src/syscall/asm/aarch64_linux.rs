#![cfg(all(target_arch = "aarch64", target_os = "linux"))]
//! mmap syscall for aarch64 Linux.
//!
//! On aarch64 Linux, the syscall convention is:
//! - Syscall number in `x8`
//! - Arguments in `x0` through `x5`
//! - Invoke with `svc #0` (supervisor call)
//! - Return value in `x0`; negative values indicate an error (negated errno)
//!
//! This is the same error convention as x86_64 Linux (negative return = error),
//! but uses different registers and a different instruction to invoke the syscall.

use super::super::MmapError;
use core::arch::asm;

const SYS_MMAP: u64 = 222;

/// Call the mmap syscall directly via inline assembly.
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
    let out_addr: i64;

    asm!(
        "svc #0",
        in("x8") SYS_MMAP,
        inout("x0") addr as i64 => out_addr,
        in("x1") len,
        in("x2") prot,
        in("x3") flags,
        in("x4") fd,
        in("x5") offset,
    );

    if out_addr < 0 {
        return Err(MmapError::from_code(-out_addr));
    }

    Ok(out_addr as *mut u8)
}
