//! This is a very minimal example to show using the HeapGrower functions.

#[cfg(feature = "use_libc")]
use basic_allocator::allocators::LibcHeapGrower;
use basic_allocator::allocators::{HeapGrower, SyscallHeapGrower};

fn main() {
    #[cfg(feature = "use_libc")]
    {
        // LibcHeapGrower uses libc to call mmap
        let mut lhg = LibcHeapGrower::default();
        let (p, sz) = unsafe { lhg.grow_heap(8).unwrap() };
        println!("Returned: ({:p}={}, {})", p, p as i64, sz);
    }

    // SyscallHeapGrower uses inline assembly to make a direct mmap syscall.
    // This requires nightly
    let mut shg = SyscallHeapGrower::default();
    let (p, sz) = unsafe { shg.grow_heap(8).unwrap() };
    println!("Returned: ({:p}={}, {})", p, p as i64, sz);
}
