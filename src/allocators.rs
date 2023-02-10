//! Basic allocator types, both generic and Unix-specific.
//!
//! ## Basic Types
//!
//! ### [`RawAlloc`](struct.RawAlloc.html)
//!
//! A `RawAlloc` is a single-threaded, non-thread-safe heap and freed memory
//! manager, implementing
//! [`core::alloc::GlobalAlloc`](https://doc.rust-lang.org/nightly/core/alloc/trait.GlobalAlloc.html).
//! However, because it is not thread-safe, it canot be used as a global
//! allocator.BlockList
//!
//! ### [`UnixAllocator`](struct.UnixAllocator.html)
//!
//! A `UnixAllocator` wraps `RawAlloc` with a spin lock to make it thread-safe,
//! allowing it to be used as the global allocator. It also combines `RawAlloc`
//! with a unix-specific `UnixHeapGrower` to use virtual memory pages as its
//! underlying basis for making those calls.
//!
//! ### [`HeapGrower`](struct.HeapGrower.html)
//!
//! `HeapGrower` is a simple trait interface meant to abstract over the calls to
//! the OS to expand the heap.
//!
//! ### [`ToyHeap`](struct.ToyHeap.html)
//!
//! `ToyHeap` is a static array on the stack that can pretend to be a heap, and
//! implements `HeapGrower` for such a purpose. It is mainly useful for testing.

use core::alloc::{GlobalAlloc, Layout};
use core::mem::MaybeUninit;
use core::ptr::{null_mut, NonNull};
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "use_libc")]
use errno::Errno;
use spin::{Mutex, MutexGuard};

use crate::blocklist::{BlockList, Stats, Validity};
#[cfg(not(feature = "use_libc"))]
use crate::unix::{self, mmap, MmapError};

// Round up value to the nearest multiple of increment
fn round_up(value: usize, increment: usize) -> usize {
    if value == 0 {
        return 0;
    }
    increment * ((value - 1) / increment + 1)
}

pub trait HeapGrower {
    type Err;
    /// Grow the heap by at least size. Returns a pointer and the size of the
    /// memory available at that pointer.
    ///
    /// # Safety
    ///
    /// This is pretty much entirely unsafe.
    ///
    /// For this to function properly with the other types in this module:
    ///
    /// - The return value may be (null, 0), indicating allocation failure.
    /// - The return value may be (ptr, new_size), where new_size >= size, and
    ///   where the memory pointed to by ptr must be available and untracked by
    ///   any other rust code, including the allocator itself.
    unsafe fn grow_heap(&mut self, size: usize) -> Result<(*mut u8, usize), Self::Err>;
}

/// UnixHeapGrower uses virtual memory to grow the heap upon request.
#[cfg(feature = "use_libc")]
#[derive(Default)]
pub struct LibcHeapGrower {
    // Just for tracking, not really needed
    pages: usize,
    growths: usize,
}

#[cfg(feature = "use_libc")]
impl HeapGrower for LibcHeapGrower {
    type Err = Errno;
    unsafe fn grow_heap(&mut self, size: usize) -> Result<(*mut u8, usize), Self::Err> {
        if size == 0 {
            // TODO this should maybe return a valid pointer?
            return Ok((null_mut(), 0));
        }
        let pagesize = sysconf::page::pagesize();
        let to_allocate = round_up(size, pagesize);

        let ptr = libc::mmap(
            // Address we want the memory at. We don't care, so null it is.
            null_mut(),
            // Amount of memory to allocate
            to_allocate,
            // We want read/write access to this memory
            libc::PROT_WRITE | libc::PROT_READ,
            // MAP_ANON: We don't want a file descriptor, we're just going to
            //   use the memory.
            //
            // MAP_PRIVATE: We're not sharing this with any other process.
            //
            // Well, I'm pretty unsure about these choices, but they seem to work...
            libc::MAP_ANON | libc::MAP_PRIVATE,
            // The file descriptor we want memory mapped. We don't want a memory
            // mapped file, so 0 it is.
            0,
            0,
        );

        if (ptr as i64) == -1 {
            // Error!
            return Err(errno::errno());
        }

        if ptr.is_null() {
            // No memory was allocated. Guess we're out of memory...
            // XXX: Is this possible?
            return Ok((ptr as *mut u8, 0));
        }

        self.pages += to_allocate / pagesize;
        self.growths += 1;

        Ok((ptr as *mut u8, to_allocate))
    }
}

/// SyscallHeapGrower uses virtual memory to grow the heap upon request.
#[cfg(not(feature = "use_libc"))]
#[derive(Default)]
pub struct SyscallHeapGrower {
    // Just for tracking, not really needed
    pages: usize,
    growths: usize,
}

#[cfg(not(feature = "use_libc"))]
impl HeapGrower for SyscallHeapGrower {
    type Err = MmapError;

    unsafe fn grow_heap(&mut self, size: usize) -> Result<(*mut u8, usize), MmapError> {
        if size == 0 {
            // TODO: I think this should _not_ be a null pointer?
            return Ok((null_mut(), 0));
        }

        // Page size is 4 kb "on most architectures"
        let pagesize = 4096;
        let to_allocate = round_up(size, pagesize);

        let ptr = mmap(
            // Address we want the memory at. We don't care, so null it is.
            null_mut(),
            // Amount of memory to allocate
            to_allocate,
            // We want read/write access to this memory
            unix::PROT_WRITE | unix::PROT_READ,
            // MAP_ANON: We don't want a file descriptor, we're just going to
            //   use the memory.
            //
            // MAP_PRIVATE: We're not sharing this with any other process.
            //
            // Well, I'm pretty unsure about these choices, but they seem to work...
            unix::MAP_ANON | unix::MAP_PRIVATE,
            // The file descriptor we want memory mapped. We don't want a memory
            // mapped file, so 0 it is.
            0,
            0,
        )?;

        if ptr.is_null() {
            // I'm not sure when this would happen...
            return Ok((ptr as *mut u8, 0));
        }

        self.pages += to_allocate / pagesize;
        self.growths += 1;

        Ok((ptr as *mut u8, to_allocate))
    }
}

/// A raw allocator, capable of growing the heap, returning pointers to new
/// allocations, and tracking and reusing freed memory.
///
/// Note: It never returns memory to the OS; that is not implemented.
///
/// This roughly corresponds to the
/// [`AllocRef`](https://doc.rust-lang.org/nightly/core/alloc/trait.AllocRef.html)
/// trait in Rust nightly, but does not directly implement that trait (although
/// it probably could... TODO!)
pub struct RawAlloc<G> {
    pub grower: G,
    pub blocks: BlockList,
}

impl<G> Drop for RawAlloc<G> {
    fn drop(&mut self) {
        let blocks = core::mem::take(&mut self.blocks);
        // When we drop an allocator, we lose all access to the memory it has
        // freed.
        core::mem::forget(blocks);
    }
}

impl<G: HeapGrower + Default> Default for RawAlloc<G> {
    fn default() -> Self {
        RawAlloc {
            grower: G::default(),
            blocks: BlockList::default(),
        }
    }
}

impl<G: HeapGrower> RawAlloc<G> {
    /// Create a new `RawAlloc`
    #[allow(dead_code)]
    pub fn new(grower: G) -> Self {
        RawAlloc {
            grower,
            blocks: BlockList::default(),
        }
    }

    /// Get statistics on this allocator, and verify validity of the BlockList
    pub fn stats(&self) -> (Validity, Stats) {
        self.blocks.stats()
    }

    /// Calculate the minimum size of a block to be allocated for the given layout.
    pub fn block_size(layout: Layout) -> usize {
        // We align everything to 16 bytes, and all blocks are at least 16 bytes.
        // Its pretty wasteful, but easy!
        let aligned_layout = layout
            .align_to(16)
            .expect("Whoa, serious memory issues")
            .pad_to_align();

        aligned_layout.size()
    }

    ////////////////////////////////////////////////////////////
    // Functions for implementing GlobalAlloc

    /// Allocate space for something fitting in layout
    ///
    /// # Safety
    ///
    /// This is very unsafe. See GlobalAlloc for details.
    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let needed_size = RawAlloc::<G>::block_size(layout);

        if let Some(range) = self.blocks.pop_size(needed_size) {
            return range.start.as_ptr();
        }

        let growth = self.grower.grow_heap(needed_size);

        let (ptr, size) = match growth {
            Err(_) => return null_mut(),
            Ok(res) => res,
        };

        if size == needed_size {
            return ptr;
        }

        let free_ptr = NonNull::new_unchecked(ptr.add(needed_size));
        if size >= needed_size + BlockList::header_size() {
            self.blocks.add_block(free_ptr, size - needed_size);
        } else {
            // Uh-oh. We have a bit of extra free memory, but not enough to add
            // a header and call it a new free block. This could happen if our
            // page size was not a multiple of 16. Weird.
            //
            // We have two choices here: we could return null, indicating memory
            // allocation failure, or we could leak it, and log it if possible.
            //
            // Leaking it is relatively safe, and should be uncommon; at most
            // once per page, and the only memory leaked would be smaller than
            // `header_size()`, so let's do that. Preferably, we would log it
            // too, but that would require a logging implementation that does
            // not rely on `std` or on allocation, which is not easily
            // available.
            //
            // This is not generally expected, so we add a debug_assert here.
            debug_assert!(
                false,
                "Unexpected memory left over. Is page_size a multiple of header size?"
            );
        }

        ptr
    }

    /// Deallocate (or "free") a memory block.
    ///
    /// # Safety
    ///
    /// This is very unsafe. See GlobalAlloc for details.
    pub unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        let size = RawAlloc::<G>::block_size(layout);
        self.blocks.add_block(NonNull::new_unchecked(ptr), size);
    }
}

/// A thread-safe allocator, using a spin lock around a RawAlloc.
///
/// Thread-safety is required for an allocator to be used as a global allocator,
/// so that was easy to add with a spin lock.
pub struct GenericAllocator<G> {
    // Values:
    // - 0: Untouched
    // - 1: Initialization in progress
    // - 2: Initialized
    init: AtomicU8,
    raw: MaybeUninit<Mutex<RawAlloc<G>>>,
}

impl<G: HeapGrower + Default> Default for GenericAllocator<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G> GenericAllocator<G> {
    pub const fn new() -> Self {
        GenericAllocator {
            init: AtomicU8::new(0),
            raw: MaybeUninit::uninit(),
        }
    }
}

impl<G: HeapGrower + Default> GenericAllocator<G> {
    /// Get a reference to the underlying RawAlloc.
    ///
    /// # Safety
    ///
    /// This is unsafe because it blocks allocation while the mutex guard is in
    /// place.
    pub unsafe fn get_raw(&self) -> MutexGuard<RawAlloc<G>> {
        // The plan:
        // - Check if initialization hasn't started (0)
        // - If initializing hasn't yet started (0):
        //   - Mark it as initializing (1), then initialize, then mark it as fully initialized (2)
        // - If it has started but not completed (1):
        //   - Enter a spin loop until it is fully initiialized (2)
        // - If it finished initializing (2):
        //   - Continue
        //
        // The ordering here is SeqCst because that's the safest, if not the
        // most efficient. This could probably be downgraded, but would require
        // some analysis and understanding to do so.
        let state = self
            .init
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst);

        match state {
            Err(2) => {
                // This is fully initialized, no need to do anything
            }
            Ok(0) => {
                // We haven't initialized, so we do that now.

                // We cast the raw pointer to be
                let raw_loc: *const Mutex<RawAlloc<G>> = self.raw.as_ptr();
                let raw_mut: *mut Mutex<RawAlloc<G>> = raw_loc as *mut Mutex<RawAlloc<G>>;
                raw_mut.write(Mutex::new(RawAlloc::default()));
                let mx: &mut Mutex<RawAlloc<G>> = raw_mut.as_mut().unwrap();

                // Let other threads know that the mutex and raw allocator are now initialized,
                // and they are free to use the mutex to access the raw allocator
                self.init.store(2, Ordering::SeqCst);
                return mx.lock();
            }
            Err(1) => {
                // Some other thread is currently initializing. We wait for it.

                // Spin while we wait for the state to become 2
                loop {
                    // Hint to the processor that we're in a spin loop
                    core::hint::spin_loop();

                    // Check if the
                    match self.init.load(Ordering::SeqCst) {
                        1 => continue,
                        2 => break,
                        state => panic!("Unexpected state {}", state),
                    }
                }
            }
            Ok(v) => panic!("Unexpected OK state loaded: {}", v),
            Err(v) => panic!("Unexpected Err state loaded: {}", v),
        }

        let ptr = self.raw.as_ptr().as_ref().unwrap();

        ptr.lock()
    }

    pub fn stats(&self) -> (Validity, Stats) {
        unsafe { self.get_raw().stats() }
    }
}

#[derive(Default)]
pub struct UnixAllocator {
    #[cfg(not(feature = "use_libc"))]
    alloc: GenericAllocator<SyscallHeapGrower>,

    #[cfg(feature = "use_libc")]
    alloc: GenericAllocator<LibcHeapGrower>,
}

impl UnixAllocator {
    pub const fn new() -> Self {
        UnixAllocator {
            alloc: GenericAllocator::new(),
        }
    }

    pub fn stats(&self) -> (Validity, Stats) {
        self.alloc.stats()
    }
}

unsafe impl GlobalAlloc for UnixAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc.get_raw().alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.alloc.get_raw().dealloc(ptr, layout)
    }
}

pub struct ToyHeap {
    pub page_size: usize,
    pub size: usize,
    pub heap: [u8; 256 * 1024],
}

impl Default for ToyHeap {
    fn default() -> Self {
        ToyHeap {
            page_size: 64,
            size: 0,
            heap: [0; 256 * 1024],
        }
    }
}

pub struct ToyHeapOverflowError();

impl HeapGrower for ToyHeap {
    type Err = ToyHeapOverflowError;

    unsafe fn grow_heap(&mut self, size: usize) -> Result<(*mut u8, usize), Self::Err> {
        if self.size + size > self.heap.len() {
            return Err(ToyHeapOverflowError());
        }

        let allocating = round_up(size, self.page_size);
        let ptr = self.heap.as_mut_ptr().add(self.size);
        self.size += allocating;
        Ok((ptr, allocating))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use test_log::test;

    #[test]
    fn test_basic() {
        let toy_heap = ToyHeap::default();
        let mut allocator = RawAlloc::new(toy_heap);

        const BLOCKS: usize = 3;

        let layouts: [Layout; BLOCKS] = [
            Layout::from_size_align(64, 16).unwrap(),
            Layout::from_size_align(64, 16).unwrap(),
            Layout::from_size_align(224, 16).unwrap(),
        ];

        let pointers: [*mut u8; BLOCKS] = unsafe {
            let mut pointers = [null_mut(); BLOCKS];
            for (i, &l) in layouts.iter().enumerate() {
                pointers[i] = allocator.alloc(l);
                let (validity, _stats) = allocator.stats();
                assert!(validity.is_valid());
            }
            pointers
        };

        for i in 0..BLOCKS - 1 {
            let l = layouts[i];
            let expected = unsafe { pointers[i].add(l.size()) };
            let found = pointers[i + 1];
            assert_eq!(expected, found);
        }

        let toy_heap = &allocator.grower;
        let page_size = toy_heap.page_size;
        // Toy heap should be the same size as the blocks requested
        let total_allocated: usize = layouts.iter().map(|l| l.size()).sum();
        let page_space = round_up(total_allocated, page_size);

        assert_eq!(toy_heap.size, page_space);

        ////////////////////////////////////////////////////////////
        // Deallocation

        // Deallocate the second pointer
        unsafe { allocator.dealloc(pointers[1], layouts[1]) };
        let (validity, _stats) = allocator.stats();
        assert!(validity.is_valid());

        // Check that the block list is as expected
        let mut iter = allocator.blocks.iter();
        let first = iter.next();
        assert!(first.is_some());

        let first = first.expect("This should not be null");
        assert_eq!(first.size(), layouts[1].size());
        let next_exists = iter.next().is_some();
        log::info!("dealloc: {}", allocator.blocks);
        // We should still have the remainder left over from the last page
        // allocation
        assert!(next_exists);

        // The block list now has 1 64-byte block on it
        log::info!("post-alloc: {}", allocator.blocks);
        ////////////////////////////////////////////////////////////
        // Allocation with a block list
        unsafe {
            // Allocate 112 bytes, more than fits in the block on the block list
            let newp = allocator.alloc(Layout::from_size_align(112, 16).unwrap());
            let (validity, _stats) = allocator.stats();
            assert!(validity.is_valid());
            assert_eq!(
                newp,
                pointers[2].add(round_up(layouts[2].size(), page_size))
            );
            log::info!("p112: {}", allocator.blocks);

            // Allocate 32 bytes, which should fit in the block
            let p32 = allocator.alloc(Layout::from_size_align(32, 16).unwrap());
            let (validity, _stats) = allocator.stats();
            assert!(validity.is_valid());
            // The algorithm returns the second half of the block
            assert_eq!(p32, pointers[1].add(32));

            // We should now still have 32 bytes in 1 block in the block list (plus page leftovers)

            // Allocate 8 bytes and another 16 bytes, which should both fit in the block
            // and completely consume it - because the 8 bytes should expand to 16
            log::info!("p32: {}", allocator.blocks);
            let p8 = allocator.alloc(Layout::from_size_align(16, 4).unwrap());
            let (validity, _stats) = allocator.stats();
            assert!(validity.is_valid());
            log::info!("p8: {}", allocator.blocks);
            let p16 = allocator.alloc(Layout::from_size_align(8, 1).unwrap());
            let (validity, _stats) = allocator.stats();
            assert!(validity.is_valid());
            // The algorithm returns the second half of the block
            log::info!("p16: {}", allocator.blocks);
            assert_eq!(p8, pointers[1].add(16));
            assert_eq!(p16, pointers[1]);
            log::info!("done: {}", allocator.blocks);
        };
    }
}
