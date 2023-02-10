#![cfg(not(feature = "use_libc"))]
/// Some unix constants and an mmap function using assembly.
///
/// The constants were taken from sys/mman.h, and have been tested on osx and
/// linux.
use core::arch::asm;
//============================================================
// System call code
#[cfg(target_os = "macos")]
const SYS_MMAP: i64 = 0x2000000 + 197;
#[cfg(target_os = "linux")]
const SYS_MMAP: i64 = 0x09;

//============================================================
// Flags for protection

// These seem to be the same across linux and OS X
pub const PROT_READ: u64 = 0x01; // [MC2] pages can be read
pub const PROT_WRITE: u64 = 0x02; // [MC2] pages can be written
const _PROT_EXEC: u64 = 0x04;

//============================================================
// Flags contain sharing type and options.

//Sharing types; choose one.
const _MAP_SHARED: u64 = 0x0001; // [MF|SHM] share changes
pub const MAP_PRIVATE: u64 = 0x0002; // [MF|SHM] changes are private

// Other flags
const _MAP_FIXED: u64 = 0x0010; // [MF|SHM] interpret addr exactly
const _MAP_RENAME: u64 = 0x0020; // Sun: rename private pages to file
const _MAP_NORESERVE: u64 = 0x0040; // Sun: don't reserve needed swap area
const _MAP_RESERVED: u64 = 0x0080; // previously unimplemented MAP_INHERIT
const _MAP_NOEXTEND: u64 = 0x0100; // for MAP_FILE, don't change file size
const _MAP_HASSEMAPHORE: u64 = 0x0200; // region may contain semaphores
const _MAP_NOCACHE: u64 = 0x0400; // don't cache pages for this mapping
const _MAP_JIT: u64 = 0x0800; // Allocate a region that will be used for JIT purposes

// Mapping type
const _MAP_FILE: u64 = 0x0000; // map from file (default)
#[cfg(target_os = "macos")]
pub const MAP_ANON: u64 = 0x1000; // allocated from memory, swap space
#[cfg(target_os = "linux")]
pub const MAP_ANON: u64 = 0x20;

const _MAP_ANONYMOUS: u64 = MAP_ANON;

#[derive(Debug)]
pub struct MmapError {
    code: i64,
}

#[cfg(all(not(feature = "use_libc"), target_os = "linux"))]
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
        r"
        syscall
    ",
    inout("eax") SYS_MMAP => out_addr,
    in("edi") addr,
    in("esi") len,
    in("edx") prot,
    in("r10d") flags,
    in("r8d") fd,
    in("r9d") offset,
    );

    if out_addr < 0 {
        return Err(MmapError { code: (-out_addr) });
    }

    Ok(out_addr as *mut u8)
}

#[cfg(all(not(feature = "use_libc"), target_os = "macos"))]
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
        r"
        // Make a syscall, using the parameters in the registers
        syscall
        // osx sets the carry bit if there's an error. If that happens, we jump
        // to label 1
        jc 1f
        // Set edx to 0 to indicate no error
        mov edx, 0
        // Jump to label 2 to finish this
        jmp 2f
1:
        // There was an error. Set edx to 1 to indicate that.
        mov edx, 1
2:
    ",
    inout("eax") SYS_MMAP => out_addr,
    in("edi") addr,
    in("esi") len,
    inout("edx") prot => err,
    in("r10d") flags,
    in("r8d") fd,
    in("r9d") offset,
    );

    if err != 0 {
        return Err(MmapError { code: out_addr });
    }

    Ok(out_addr as *mut u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    use test_env_log::test;

    #[test]
    fn test_mmap() {
        let ptr = unsafe {
            mmap(
                // Address
                core::ptr::null_mut(),
                // Amount of memory to allocate
                8,
                // We want read/write access to this memory
                PROT_WRITE | PROT_READ,
                // Mapping flags; MAP_ANON says fd should not be 0
                MAP_ANON | MAP_PRIVATE,
                // The file descriptor we want memory mapped. We don't want a memory
                // mapped file, so 0 it is.
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
                // Address
                core::ptr::null_mut(),
                // Amount of memory to allocate
                8,
                // We want read/write access to this memory
                PROT_WRITE | PROT_READ,
                // Mapping flags; we use 0 for now
                0,
                // The file descriptor we want memory mapped. Without MAP_ANON, this should be set.
                // This should be an error.
                0,
                0,
            )
        };

        assert!(ptr.is_err());
    }
}
