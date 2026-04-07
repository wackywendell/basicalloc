#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
//! mmap syscall for x86_64 Linux.
//!
//! On x86_64 Linux, the syscall convention is:
//! - Syscall number in `rax` (aliased as `eax` for 32-bit values)
//! - Arguments in `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9`
//! - Invoke with the `syscall` instruction
//! - Return value in `rax`; negative values indicate an error (negated errno)

use super::super::MmapError;
use core::arch::asm;

const SYS_MMAP: i64 = 0x09;

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

    asm!(
        "syscall",
        inout("eax") SYS_MMAP => out_addr,
        in("edi") addr,
        in("esi") len,
        in("edx") prot,
        in("r10d") flags,
        in("r8d") fd,
        in("r9d") offset,
    );

    if out_addr < 0 {
        return Err(MmapError::from_code(-out_addr));
    }

    Ok(out_addr as *mut u8)
}
