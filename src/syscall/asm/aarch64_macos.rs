#![cfg(all(target_arch = "aarch64", target_os = "macos"))]
//! mmap syscall for aarch64 macOS (Apple Silicon).
//!
//! On aarch64 macOS (XNU), the syscall convention is:
//! - Syscall number in `x16`
//! - Arguments in `x0` through `x5`
//! - Invoke with `svc #0x80` (supervisor call; the 0x80 immediate is Apple's
//!   convention — the kernel reportedly ignores the immediate value)
//! - On success: carry flag (C bit in NZCV) is clear, return value in `x0`
//! - On error: carry flag is set, errno value in `x0`
//!
//! Unlike x86_64 macOS, aarch64 macOS does NOT use the `0x2000000` class prefix
//! for Unix syscall numbers. The kernel extracts only the low 16 bits of x16
//! via `(unsigned short)x16` and uses that directly as the syscall table index.
//! Positive values are BSD/Unix syscalls; negative values are Mach traps.

use super::super::MmapError;
use core::arch::asm;

// No class prefix on aarch64 — just the raw syscall number.
// (Compare x86_64 macOS which uses 0x2000000 + 197.)
const SYS_MMAP: u64 = 197;

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
    let out_addr: u64;
    let err: u64;

    asm!(
        "svc #0x80",
        // Use cset to capture the carry flag into a register.
        // "cs" = carry set, which macOS uses to indicate syscall error.
        // Sets err to 1 on error, 0 on success.
        "cset {err:w}, cs",
        err = out(reg) err,
        in("x16") SYS_MMAP,
        inout("x0") addr as u64 => out_addr,
        in("x1") len,
        in("x2") prot,
        in("x3") flags,
        in("x4") fd,
        in("x5") offset,
    );

    if err != 0 {
        return Err(MmapError::from_code(out_addr as i64));
    }

    Ok(out_addr as *mut u8)
}
