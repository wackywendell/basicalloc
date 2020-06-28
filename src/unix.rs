//============================================================
// System call code
#[cfg(target_os = "macos")]
const SYS_MMAP: i64 = 0x2000000 + 197;
#[cfg(target_os = "linux")]
const SYS_MMAP: i64 = 0x09;

//============================================================
// Flags for protection

pub const PROT_READ: u64 = 0x01; // [MC2] pages can be read
pub const PROT_WRITE: u64 = 0x02; // [MC2] pages can be written
pub const _PROT_EXEC: u64 = 0x04;

//============================================================
// Flags contain sharing type and options.

//Sharing types; choose one.
pub const _MAP_SHARED: u64 = 0x0001; // [MF|SHM] share changes
pub const MAP_PRIVATE: u64 = 0x0002; // [MF|SHM] changes are private

// Other flags
pub const _MAP_FIXED: u64 = 0x0010; // [MF|SHM] interpret addr exactly
pub const _MAP_RENAME: u64 = 0x0020; // Sun: rename private pages to file
pub const _MAP_NORESERVE: u64 = 0x0040; // Sun: don't reserve needed swap area
pub const _MAP_RESERVED: u64 = 0x0080; // previously unimplemented MAP_INHERIT
pub const _MAP_NOEXTEND: u64 = 0x0100; // for MAP_FILE, don't change file size
pub const _MAP_HASSEMAPHORE: u64 = 0x0200; // region may contain semaphores
pub const _MAP_NOCACHE: u64 = 0x0400; // don't cache pages for this mapping
pub const _MAP_JIT: u64 = 0x0800; // Allocate a region that will be used for JIT purposes

// Mapping type
pub const _MAP_FILE: u64 = 0x0000; // map from file (default)
pub const MAP_ANON: u64 = 0x1000; // allocated from memory, swap space
pub const _MAP_ANONYMOUS: u64 = MAP_ANON;

pub unsafe fn mmap(
    addr: *mut u8,
    len: usize,
    prot: u64,
    flags: u64,
    fd: u64,
    offset: i64,
) -> *mut u8 {
    let addr = addr as i64;
    let mut in_out: i64 = SYS_MMAP;

    asm!(
        r"
        syscall
    ",
    inout("eax") in_out,
    in("edi") addr,
    in("esi") len,
    in("edx") prot,
    in("r10d") flags,
    in("r8d") fd,
    in("r9d") offset,
    );

    in_out as *mut u8
}
