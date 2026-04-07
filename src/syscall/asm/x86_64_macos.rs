#![cfg(all(target_arch = "x86_64", target_os = "macos"))]
//! mmap syscall for x86_64 macOS.
//!
//! On x86_64 macOS (XNU), the syscall convention is:
//! - Syscall number in `rax`, with a class prefix: Unix syscalls use `0x2000000 + number`
//! - Arguments in `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9`
//! - Invoke with the `syscall` instruction
//! - On success: carry flag is clear, return value in `rax`
//! - On error: carry flag is set, errno value in `rax`
//!
//! This differs from Linux, where errors are indicated by a negative return value.

use super::super::MmapError;
use core::arch::asm;

// macOS x86_64 syscall numbers include a class prefix.
// Class 2 (0x2000000) = BSD/Unix syscalls.
const SYS_MMAP: i64 = 0x2000000 + 197;

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
    let addr = addr as i64;
    let out_addr: i64;
    let err: i64;

    asm!(
        // Make the syscall
        "syscall",
        // macOS sets the carry flag on error. Branch to label 1 if so.
        "jc 1f",
        // No error: set edx to 0
        "mov edx, 0",
        "jmp 2f",
        "1:",
        // Error: set edx to 1
        "mov edx, 1",
        "2:",
        inout("eax") SYS_MMAP => out_addr,
        in("edi") addr,
        in("esi") len,
        inout("edx") prot => err,
        in("r10d") flags,
        in("r8d") fd,
        in("r9d") offset,
    );

    if err != 0 {
        return Err(MmapError::from_code(out_addr));
    }

    Ok(out_addr as *mut u8)
}
