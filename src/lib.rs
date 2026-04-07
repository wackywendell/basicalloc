#![no_std]

//! A simple memory allocator, written for educational purposes.
//!
//! This module was written primarily for the code to be read. The allocator
//! enclosed can be used as a memory allocator in a rust program.
//!
//! ## Usage
//!
//! ```rust
//! use basic_allocator::UnixAllocator;
//!
//! #[global_allocator]
//! static ALLOCATOR: UnixAllocator = UnixAllocator::new();
//! fn main() {
//!     println!("It works!")
//! }
//! ```
//!
//! ## Logging
//!
//! The allocator emits `log::trace!` messages during allocation and
//! deallocation (block splitting, merging, etc.). These are zero-cost when
//! no logger is installed. To see them in tests:
//!
//! ```console
//! $ RUST_LOG=trace cargo test
//! ```
//!
//! Note: setting up a logger in a program that uses this as its global
//! allocator is tricky — the logger itself may allocate. The trace output
//! is most useful during testing with `ToyHeap` or when debugging with
//! an external logger that doesn't allocate.
//!
//! See also
//! [`core::alloc::GlobalAlloc`](https://doc.rust-lang.org/nightly/core/alloc/trait.GlobalAlloc.html).
//!
//! ## Major Components
//!
//! This module has several parts.
//!
//! ### [`BlockList`](blocklist/struct.BlockList.html)
//!
//! A `BlockList` is a linked list of _freed_ memory not returned to the OS,
//! which can be reused by the allocator.
//!
//! The free block starts with a header, and then has unused memory after that.
//! The header is 16 bytes, and consists of a pointer to the next block and the
//! size of the block as a whole.
//!
//! ### [`RawAlloc`](allocators/struct.RawAlloc.html)
//!
//! A `RawAlloc` is a single-threaded, non-thread-safe heap and freed memory
//! manager, implementing
//! [`core::alloc::GlobalAlloc`](https://doc.rust-lang.org/nightly/core/alloc/trait.GlobalAlloc.html).
//! However, because it is not thread-safe, it cannot be used as a global
//! allocator.
//!
//! ### [`UnixAllocator`](allocators/struct.UnixAllocator.html)
//!
//! A `UnixAllocator` wraps `RawAlloc` with a spin lock to make it thread-safe,
//! allowing it to be used as the global allocator. It combines `RawAlloc`
//! with [`MmapHeapGrower`](allocators/struct.MmapHeapGrower.html), which uses
//! the [`syscall`] module to request virtual memory pages from the OS.
//!
//! ### [`HeapGrower`](allocators/trait.HeapGrower.html)
//!
//! `HeapGrower` is a simple trait interface meant to abstract over the calls to
//! the OS to expand the heap.
//!
//! ### [`syscall`](syscall/index.html)
//!
//! The `syscall` module provides a platform-independent interface for calling
//! `mmap`. It supports both inline assembly (on x86_64 and aarch64, Linux and
//! macOS) and a libc wrapper (via the `use_libc` feature). Both paths expose
//! the same function signature, so code above this layer needs no cfg gates.
//!
//! ## Implementation
//!
//! Free memory is maintained in a linked list. The allocator has a pointer to
//! the first block, and each block starts with a header with a pointer to the
//! next block and the size of the current block. Blocks are in-order, so that
//! merges are easily implemented.
//!
//! ### Allocation
//!
//! When [`RawAlloc`](allocators/struct.RawAlloc.html) is
//! [called](allocators/struct.RawAlloc.html#method.alloc) to allocate `size` bytes:
//!
//! 1. The [`BlockList`](blocklist/struct.BlockList.html) is iterated through, and if any
//!    free block is found there large enough for the request, it is used. If
//!    the found block is just the right size, "popped" out of the linked list,
//!    and returned as a block of free memory; otherwise, the last `size` bytes
//!    of the block is returned as free memory, and the block's header is
//!    adjusted as needed.
//! 2. If no suitable block is found in the list, the
//!    [`HeapGrower`](allocators/trait.HeapGrower.html) instance is
//!    [called](allocators/trait.HeapGrower.html#tymethod.grow_heap) to "grow the heap".
//!    For [`MmapHeapGrower`](allocators/struct.MmapHeapGrower.html), this means that
//!    one or more pages of virtual memory are requested from the OS via
//!    [`syscall::mmap`](syscall/fn.mmap.html). The first `size` bytes are
//!    returned, and the remainder of the page is added to the
//!    [`BlockList`](blocklist/struct.BlockList.html).
//!
//! ### Deallocation
//!
//! When [`RawAlloc`](allocators/struct.RawAlloc.html) is
//! [called](allocators/struct.RawAlloc.html#method.dealloc) to deallocate `size` bytes at
//! a pointer `ptr`:
//!
//! 1. The [`BlockList`](blocklist/struct.BlockList.html) is iterated through to find
//!    where in the list `ptr` should be to remain sorted.
//! 2. `ptr` is inserted, and an attempt is made to merge with both the
//!    preceding and following blocks. Each attempt is successful only if the
//!    two blocks involved are adjacent.
//!
//! ## Possible Extensions
//!
//! This is a very simple allocator, by design. There are a number of ways it
//! could be better, in terms of features and performance:
//!
//! 1. It could return memory to the OS when it was done with a page
//! 2. It could not require 16-byte alignment
//! 3. It could have a thread-safe linked-list implementation, removing the need
//!    for a spin lock
//! 4. It could implement
//!    [`realloc`](https://doc.rust-lang.org/core/alloc/trait.GlobalAlloc.html#method.realloc),
//!    so that containers could be resized in place when possible
//!
//! ... and probably more. Beyond those basic features, there are lots of
//! optimizations in other allocators that make them more performant.

pub mod allocators;
pub mod blocklist;
pub mod syscall;

pub use allocators::{RawAlloc, UnixAllocator};
pub use blocklist::BlockList;
