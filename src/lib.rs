#![no_std]

//! A basic memory allocator.

use core::alloc::{GlobalAlloc, Layout};
use core::fmt;
use core::mem::MaybeUninit;
use core::ops::Range;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicU8, Ordering};

use spin::{Mutex, MutexGuard};
use static_assertions::const_assert;

// The header for our free blocks.
//
// The header includes a pointer to the next free block, and the size of the
// current block (including the header).
//
// We use C representation and align to 16 bytes for... simplicity. This is
// perhaps a stronger constraint that we need, but it does make things simple
// and straightforward.
#[derive(Copy, Clone)]
#[repr(C, align(16))]
struct FreeHeader {
    next: *mut FreeHeader,
    size: usize,
}

// We will align to 16 bytes and our headers will be given that much space
// Similarly, all blocks will be at least 16 bytes large, even if they aren't
// aware of it.
//
// This is likely a stronger constraint than is entirely needed, but it does
// simplify things.
const HEADER_SIZE: usize = 16;
const_assert!(HEADER_SIZE <= core::mem::size_of::<FreeHeader>());

// An enum for easy comparison of blocks and their order
enum Relation {
    Before,
    AdjacentBefore,
    Overlapping,
    AdjacentAfter,
    After,
}

impl FreeHeader {
    #[allow(clippy::cast_ptr_alignment)]
    unsafe fn from_raw(ptr: *mut u8, next: *mut FreeHeader, size: usize) -> *mut FreeHeader {
        let header = FreeHeader { next, size };
        core::ptr::write(ptr as *mut FreeHeader, header);
        ptr as *mut FreeHeader
    }
}

// Invariants that should be maintained:
//
// - header should never be a null pointer
// - header.next should be null or point to a valid MemoryBlock
// - Any safe functions should preserve LinkedLists
#[derive(Copy, Clone)]
struct FreeBlock {
    header: *mut FreeHeader,
}

impl FreeBlock {
    fn from_raw(ptr: *mut u8, next: Option<FreeBlock>, size: usize) -> FreeBlock {
        if size < HEADER_SIZE {
            panic!("Can't recapture a block smaller than HEADER_SIZE");
        }
        let next_ptr = next.map(|b| b.header).unwrap_or(null_mut());
        let header = unsafe { FreeHeader::from_raw(ptr, next_ptr, size) };
        FreeBlock { header }
    }

    fn _as_slice(&self) -> &[u8] {
        unsafe {
            let size = self.header.read().size;
            core::slice::from_raw_parts(self.header as *const u8, size)
        }
    }

    fn as_range(&self) -> Range<*const u8> {
        unsafe {
            let size = self.header.read().size;
            let start = self.header as *const u8;
            start..(start.add(size))
        }
    }

    fn relation(&self, other: &Self) -> Relation {
        let self_range = self.as_range();
        let other_range = other.as_range();

        if self_range.end < other_range.start {
            Relation::Before
        } else if self_range.end == other_range.start {
            Relation::AdjacentBefore
        } else if self_range.start < other_range.end {
            Relation::Overlapping
        } else if self_range.start == other_range.end {
            Relation::AdjacentAfter
        } else {
            Relation::After
        }
    }

    fn next(&self) -> Option<Self> {
        let next = self.header_view().next;
        if next.is_null() {
            return None;
        }
        Some(FreeBlock { header: next })
    }

    fn size(&self) -> usize {
        self.header_view().size
    }

    fn header_view(&self) -> &FreeHeader {
        unsafe {
            self.header
                .as_ref()
                .expect("FreeBlock pointers should never be null")
        }
    }

    // Get a mutable view of the header.
    //
    // This method is unsafe because it allows modifying the size or pointer of
    // a free block in safe code, which could lead to corruption.
    unsafe fn header_mut(&mut self) -> &mut FreeHeader {
        self.header
            .as_mut()
            .expect("FreeBlock pointers should never be null")
    }

    // Remove the block after this one from the linked list, and return
    // a pointer to that block and its size.
    //
    // If next is null, returns a null pointer and size 0.
    fn pop_next(&mut self) -> (*mut u8, usize) {
        let header = unsafe { self.header_mut() };

        let next = header.next;
        if next.is_null() {
            return (null_mut(), 0);
        }

        let block = FreeBlock { header: next };
        // Update this block to look to next's next, cutting next out of the chain
        header.next = block.header_view().next;

        (block.header as *mut u8, block.size())
    }

    // Insert a new element, after this one
    unsafe fn insert(&mut self, ptr: *mut u8, size: usize) {
        let block = FreeBlock::from_raw(ptr, self.next(), size);
        self.header_mut().next = block.header;
    }

    unsafe fn insert_merge(&mut self, ptr: *mut u8, size: usize) -> usize {
        let end = (self.header as *const u8).add(self.size());

        let (merges, mut try_next) = if end == ptr {
            self.header_mut().size += size;
            (1, *self)
        } else {
            self.insert(ptr, size);
            (0, self.next().unwrap())
        };

        merges + if try_next.try_merge_next() { 1 } else { 0 }
    }

    // Split off part of this FreeBlock, and return a pointer to the split off
    // data.
    //
    // The returned pointer is to a region of size 'size' that is no longer
    // considered free.
    //
    // Panics if 'size' is greater than this block's size - HEADER_SIZE, as
    // there is no way to split off a chunk that large while leaving behind a
    // FreeBlock with an intact header.
    fn split(&mut self, size: usize) -> *mut u8 {
        if size + HEADER_SIZE > self.header_view().size {
            panic!(
                "Can't split a block of size {} off of a block of size {} - need {} for header",
                size,
                self.size(),
                HEADER_SIZE,
            )
        }

        unsafe {
            // let self_size = self.size();
            let header = self.header_mut();
            header.size -= size;
            (header as *mut FreeHeader as *mut u8).add(header.size)
            // log::trace!(
            //     "Splitting {} bytes off from {:?}:{} to get {:?}",
            //     size,
            //     (header as *mut FreeHeader as *mut u8),
            //     self_size,
            //     ptr,
            // );
        }
    }

    // Attempt to merge this block with the next.
    //
    // If the next block exaists, is adjacent, and exists directly after this
    // block, the two will merge and this will return True; otherwise, this will
    // return False.
    fn try_merge_next(&mut self) -> bool {
        let next = match self.next() {
            None => return false,
            Some(block) => block,
        };
        let end = unsafe { (self.header as *const u8).add(self.size()) };
        if end != (next.header as *const u8) {
            return false;
        };

        unsafe {
            let header = self.header_mut();
            header.size += next.size();
            header.next = next.header_view().next;
        }

        true
    }
}

// A BlockList is a linked list of "free" blocks.
//
// It maintains a few invariants:
//
// - Each block should link to the next, with the last one linking to null.
// - Each block should have a pointer < next.
// - No two blocks should be precisely adjacent (those should be automatically
//   merged on insertion).
pub struct BlockList {
    first: Option<FreeBlock>,
}

// A BlockList is sendable - as long as the whole "chain" is maintained across
// threads, its fine.
//
// With some tweaking and atomic pointer swapping, we could make a thread-safe
// version of BlockList, but that's more trouble than I'm willing to go to right
// now
unsafe impl Send for FreeBlock {}

impl Default for BlockList {
    fn default() -> Self {
        BlockList { first: None }
    }
}

impl fmt::Display for BlockList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockList(")?;
        let mut next = self.first;
        let mut start = true;
        while let Some(ref block) = next {
            if !start {
                write!(f, ", ")?;
            } else {
                start = false;
            }
            write!(f, "FreeBlock({:?}, {})", block.header, block.size())?;

            next = block.next();
        }

        write!(f, ")")
    }
}

#[derive(Default, Debug)]
pub struct Validity {
    // Number of blocks overlapping other blocks.
    //
    // This likely indicates corruption.
    //
    // If there are also out of order blocks, this might undercount.
    pub overlaps: usize,

    // Number of blocks that are directly adjacent to each other, and not
    // merged. This shouldn't happen, but isn't totally corrupt.
    pub adjacents: usize,
    // Number of blocks that do not have an address less than their next.
    //
    // This shouldn't occur.
    pub out_of_orders: usize,
}

impl Validity {
    pub fn is_valid(&self) -> bool {
        self.overlaps == 0 && self.adjacents == 0 && self.out_of_orders == 0
    }
}

impl From<Validity> for bool {
    fn from(v: Validity) -> bool {
        v.is_valid()
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub length: usize,
    pub size: usize,
}

impl BlockList {
    // Check current size of the list, and whether its valid.
    pub fn stats(&self) -> (Validity, Stats) {
        let mut previous = match self.first {
            None => {
                // No blocks, no stats
                return Default::default();
            }
            Some(p) => p,
        };

        let mut validity: Validity = Default::default();

        let mut stats = Stats {
            length: 1,
            size: previous.size(),
        };

        while let Some(next) = previous.next() {
            match previous.relation(&next) {
                Relation::Before => {
                    // This is valid, do nothing.
                }
                Relation::AdjacentBefore => {
                    // Right order, but these should be merged.
                    validity.adjacents += 1;
                }
                Relation::Overlapping => {
                    // This is really bad.
                    validity.overlaps += 1;
                }
                Relation::AdjacentAfter => {
                    // Wrong order, and these should be merged.
                    validity.out_of_orders += 1;
                    validity.adjacents += 1;
                }
                Relation::After => {
                    // Wrong order.
                    validity.out_of_orders += 1;
                }
            }

            stats.length += 1;
            stats.size += next.size();
            previous = next;
        }

        (validity, stats)
    }

    // Find and remove a chunk of size 'size' from the linked list
    fn pop_size(&mut self, size: usize) -> Option<*mut u8> {
        // debug!("pop_size({})", size);

        let mut first = self.first?;
        // debug!("  pop_size got first");
        if first.size() == size {
            // debug!("  First block at {:?} is big enough", first.header);
            self.first = first.next();
            return Some(first.header as *mut u8);
        } else if first.size() >= size + HEADER_SIZE {
            let split = first.split(size);
            // debug!(
            //     "  Split off from first block at {:?} to {:?}",
            //     first.header, split,
            // );
            return Some(split);
        }

        let mut parent = first;
        loop {
            let mut next = parent.next()?;
            // log::trace!("  Checking block at {:?} Size {}", next.header, next.size());

            if next.size() == size {
                let (ptr, _) = parent.pop_next();
                // log::trace!("  Found correctly sized block at {:?}", ptr);
                return Some(ptr);
            }

            if next.size() < size + HEADER_SIZE {
                // This block is too small, skip it
                parent = parent.next().unwrap();
                continue;
            }

            // This block is bigger than we need, split it
            // log::trace!("  Found big block at {:?}", next.header);
            return Some(next.split(size));
        }
    }

    /// Add a block to the linked list. Takes ownership of ptr.
    unsafe fn add_block(&mut self, ptr: *mut u8, size: usize) {
        let mut new_block = FreeBlock::from_raw(ptr, None, size);

        let mut previous = match self.first {
            None => {
                // There are no blocks in this list, so we make this the head of
                // the list and return
                self.first = Some(new_block);
                return;
            }
            Some(p) => p,
        };

        // We keep the list in sorted order, by pointer, to enable merging.
        match new_block.relation(&previous) {
            Relation::Before => {
                // This block is well before the first one in the list, so we
                // add this to the head of the list
                new_block.header_mut().next = previous.header;
                self.first = Some(new_block);
                return;
            }
            Relation::AdjacentBefore => {
                // This block is just before the first block in the list, so we
                // merge the two into a single block
                new_block.header_mut().next = previous.header;
                let merged = new_block.try_merge_next();
                self.first = Some(new_block);
                assert!(merged, "They were adjacent, they should merge");
                return;
            }
            Relation::Overlapping => {
                // These blocks both claim the same memory
                panic!("Overlapping memory blocks OH NO");
            }
            Relation::AdjacentAfter => {
                // This block is just after the first block in the list, so we
                // merge the two into a single block. This block isn't part of the list yet,
                // and 'previous' already correctly points to the next block, so all we need to do
                // is increase the 'previous' block size.
                previous.header_mut().size += size;
                // Now that 'previous' has grown, it's possible that 'previous'
                // is now adjacent to 'next', so we try and merge them. This may
                // or may not actually happen, and either way, we're left with a
                // valid list afterwards.
                previous.try_merge_next();
                return;
            }
            _ => {}
        }

        // Loop through the list of blocks, to find where this one should be
        // inserted. Once its place in the list is found, we merge with the
        // previous and/or next if we can, and if not, insert it into
        // the list.
        loop {
            // By construction, previous < new_block. Now we check previous.next
            // to see if previous < new_block < next, in which case we insert
            // and merge, or if next < new_block, we continue iterating through
            // the list.
            let next = match previous.next() {
                Some(n) => n,
                None => {
                    // previous < new_block, and nothing
                    previous.insert_merge(ptr, size);
                    return;
                }
            };

            if (next.header as *const u8) < ptr {
                // next < pointer, so we continue iterating
                previous = next;
                continue;
            }

            // If we are here, it means previous < ptr < next.
            // Time to insert_merge
            previous.insert_merge(ptr, size);
            return;
        }
    }

    pub fn len(&self) -> usize {
        let mut parent = match self.first {
            None => return 0,
            Some(p) => p,
        };

        let mut length = 1;
        while let Some(next) = parent.next() {
            length += 1;
            parent = next;
        }

        length
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_none()
    }
}

// Round up value to the nearest multiple of increment
fn round_up(value: usize, increment: usize) -> usize {
    if value == 0 {
        return 0;
    }
    increment * ((value - 1) / increment + 1)
}

pub trait HeapGrower {
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
    unsafe fn grow_heap(&mut self, size: usize) -> (*mut u8, usize);
}

#[derive(Default)]
struct UnixHeapGrower {
    // Just for tracking, not really needed
    pages: usize,
    growths: usize,
}

impl HeapGrower for UnixHeapGrower {
    unsafe fn grow_heap(&mut self, size: usize) -> (*mut u8, usize) {
        if size == 0 {
            return (null_mut(), 0);
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

        if ptr.is_null() {
            // panic!("No memory allocated!");
            return (ptr as *mut u8, 0);
        }

        self.pages += to_allocate / pagesize;
        self.growths += 1;

        (ptr as *mut u8, to_allocate)
    }
}

// A raw allocator, capable of growing the heap, returning pointers to new
// allocations, and tracking and reusing freed memory.
//
// Note: It never returns memory to the heap.
pub struct RawAlloc<G> {
    pub grower: G,
    pub blocks: BlockList,
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
    #[allow(dead_code)]
    pub fn new(grower: G) -> Self {
        RawAlloc {
            grower,
            blocks: BlockList::default(),
        }
    }

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

        // log::trace!(
        //     "Alignment: {}@{} -> {}@{}",
        //     layout.size(),
        //     layout.align(),
        //     aligned_layout.size(),
        //     aligned_layout.align()
        // );

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
        // log::trace!("Allocating {} bytes", needed_size);

        if let Some(ptr) = self.blocks.pop_size(needed_size) {
            // log::trace!("Popped off a block of size {} at {:?}", needed_size, ptr);
            return ptr;
        }

        let (ptr, size) = self.grower.grow_heap(needed_size);
        // log::trace!("Grew to size {}", needed_size);

        if size == needed_size {
            // log::trace!("    exactly as needed");
            return ptr;
        }

        let free_ptr = ptr.add(needed_size);
        if size >= needed_size + HEADER_SIZE {
            // log::trace!("Adding block of size {}", size - needed_size);
            self.blocks.add_block(free_ptr, size - needed_size);
        } else if size > needed_size {
            // Uh-oh. We have a bit of extra free memory, but not enough to add
            // a header and call it a new free block. This could happen if our
            // page size was not a multiple of 16. Weird.
            //
            // Log it and leak it, I guess...
            log::warn!("Leaking {} bytes at {:?}", size - needed_size, free_ptr);
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
        self.blocks.add_block(ptr, size);
    }
}

// A thread-safe allocator.
struct BasicAlloc<G> {
    // Values:
    // - 0: Untouched
    // - 1: Initialization in progress
    // - 2: Initialized
    init: AtomicU8,
    raw: MaybeUninit<Mutex<RawAlloc<G>>>,
}

impl<G: HeapGrower + Default> Default for BasicAlloc<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G> BasicAlloc<G> {
    pub const fn new() -> Self {
        BasicAlloc {
            init: AtomicU8::new(0),
            raw: MaybeUninit::uninit(),
        }
    }
}

impl<G: HeapGrower + Default> BasicAlloc<G> {
    pub fn get_raw(&self) -> MutexGuard<RawAlloc<G>> {
        // First, we check
        //
        // The ordering here is SeqCst because that's the safest, if not the
        // most efficient. This could probably be downgraded, but would require
        // some analysis and understanding to do so.
        let mut state = self.init.compare_and_swap(0, 1, Ordering::SeqCst);
        // log::info!("state: {}", state);
        if state == 0 {
            // We haven't initialized, so we do that now.
            let mx: &mut Mutex<RawAlloc<G>> = unsafe {
                // We cast the raw pointer to be
                let raw_loc: *const Mutex<RawAlloc<G>> = self.raw.as_ptr();
                let raw_mut: *mut Mutex<RawAlloc<G>> = raw_loc as *mut Mutex<RawAlloc<G>>;
                raw_mut.write(Mutex::new(RawAlloc::default()));
                raw_mut.as_mut().unwrap()
            };

            // Let other threads know that the mutex and raw allocator are now initialized,
            // and they are free to use the mutex to access the raw allocator
            self.init.store(2, Ordering::SeqCst);
            return mx.lock();
        }

        while state == 1 {
            // log::info!("Spinning!");
            // Spin while we wait for the state to become 2
            core::sync::atomic::spin_loop_hint();
            state = self.init.load(Ordering::SeqCst);
        }

        let ptr = unsafe { self.raw.as_ptr().as_ref().unwrap() };

        ptr.lock()
    }

    pub fn stats(&self) -> (Validity, Stats) {
        self.get_raw().stats()
    }
}

#[derive(Default)]
pub struct BasicUnixAlloc {
    alloc: BasicAlloc<UnixHeapGrower>,
}

impl BasicUnixAlloc {
    pub const fn new() -> Self {
        BasicUnixAlloc {
            alloc: BasicAlloc::new(),
        }
    }

    pub fn stats(&self) -> (Validity, Stats) {
        self.alloc.stats()
    }
}

unsafe impl GlobalAlloc for BasicUnixAlloc {
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

impl HeapGrower for ToyHeap {
    unsafe fn grow_heap(&mut self, size: usize) -> (*mut u8, usize) {
        if self.size + size > self.heap.len() {
            return (null_mut(), 0);
        }

        let allocating = round_up(size, self.page_size);
        let ptr = self.heap.as_mut_ptr().add(self.size);
        self.size += allocating;
        (ptr, allocating)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use test_env_log::test;

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
        assert!(allocator.blocks.first.is_some());

        let first = allocator
            .blocks
            .first
            .as_mut()
            .expect("This should not be null");
        assert_eq!(first.size(), layouts[1].size());
        let next = first.next();
        log::info!("dealloc: {}", allocator.blocks);
        // We should still have the remainder left over from the last page
        // allocation
        assert!(next.is_some());

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
