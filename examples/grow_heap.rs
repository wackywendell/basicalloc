//! A minimal example showing direct use of the HeapGrower interface.
//!
//! This demonstrates how MmapHeapGrower requests memory from the OS.
//! The same code works whether using inline assembly or libc under
//! the hood — the `use_libc` feature only affects the syscall layer.

use basic_allocator::allocators::{HeapGrower, MmapHeapGrower};

fn main() {
    let mut grower = MmapHeapGrower::default();
    let (p, sz) = unsafe { grower.grow_heap(8).unwrap() };
    println!("Returned: ({:p}={}, {})", p, p as i64, sz);
}
