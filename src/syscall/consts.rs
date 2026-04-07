//! Constants for mmap syscalls, taken from `sys/mman.h`.
//!
//! These constants are used as arguments to `mmap` to control memory protection
//! and mapping behavior. They are OS-specific but architecture-independent.

//============================================================
// Flags for memory protection (PROT_*)
//
// These control what operations are allowed on the mapped memory.

/// Pages can be read
pub const PROT_READ: u64 = 0x01;
/// Pages can be written
pub const PROT_WRITE: u64 = 0x02;
/// Pages can be executed
#[allow(dead_code)]
const _PROT_EXEC: u64 = 0x04;

//============================================================
// Flags for mapping type and options (MAP_*)
//
// These control how the mapping is created and shared.

// Sharing types; choose one.

/// Share changes with other processes mapping the same region
#[allow(dead_code)]
const _MAP_SHARED: u64 = 0x0001;
/// Changes are private to this process (copy-on-write)
pub const MAP_PRIVATE: u64 = 0x0002;

// Other flags

/// Interpret addr exactly (don't let the kernel choose)
#[allow(dead_code)]
const _MAP_FIXED: u64 = 0x0010;
/// Sun: rename private pages to file
#[allow(dead_code)]
const _MAP_RENAME: u64 = 0x0020;
/// Sun: don't reserve needed swap area
#[allow(dead_code)]
const _MAP_NORESERVE: u64 = 0x0040;
/// Previously unimplemented MAP_INHERIT
#[allow(dead_code)]
const _MAP_RESERVED: u64 = 0x0080;
/// For MAP_FILE, don't change file size
#[allow(dead_code)]
const _MAP_NOEXTEND: u64 = 0x0100;
/// Region may contain semaphores
#[allow(dead_code)]
const _MAP_HASSEMAPHORE: u64 = 0x0200;
/// Don't cache pages for this mapping
#[allow(dead_code)]
const _MAP_NOCACHE: u64 = 0x0400;
/// Allocate a region that will be used for JIT purposes
#[allow(dead_code)]
const _MAP_JIT: u64 = 0x0800;

// Mapping type

/// Map from file (default)
#[allow(dead_code)]
const _MAP_FILE: u64 = 0x0000;

/// Allocated from memory, not backed by a file. The value differs between
/// macOS and Linux.
#[cfg(target_os = "macos")]
pub const MAP_ANON: u64 = 0x1000;
/// Allocated from memory, not backed by a file. The value differs between
/// macOS and Linux.
#[cfg(target_os = "linux")]
pub const MAP_ANON: u64 = 0x20;

#[allow(dead_code)]
const _MAP_ANONYMOUS: u64 = MAP_ANON;
