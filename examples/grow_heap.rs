//! This is a very minimal example to show using the HeapGrower functions.

use basic_allocator::allocators::HeapGrower;
#[cfg(feature = "use_libc")]
use basic_allocator::allocators::LibcHeapGrower;
#[cfg(not(feature = "use_libc"))]
use basic_allocator::allocators::SyscallHeapGrower;

fn main() {
    #[cfg(feature = "use_libc")]
    {
        // LibcHeapGrower uses libc to call mmap
        println!("Using libc");
        let mut lhg = LibcHeapGrower::default();
        let (p, sz) = unsafe { lhg.grow_heap(8).unwrap() };
        println!("Returned: ({:p}={}, {})", p, p as i64, sz);
    }

    #[cfg(not(feature = "use_libc"))]
    {
        // SyscallHeapGrower uses inline assembly to make a direct mmap syscall.
        // This requires nightly
        println!("Using assembly syscalls");
        let mut shg = SyscallHeapGrower::default();
        let (p, sz) = unsafe { shg.grow_heap(8).unwrap() };
        println!("Returned: ({:p}={}, {})", p, p as i64, sz);
    }
}
