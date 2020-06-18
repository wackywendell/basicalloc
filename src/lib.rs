#![no_std]

//! A basic memory allocator.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::fmt;
use core::ops::Range;
use core::ptr::null_mut;

use log::debug;
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
            let self_size = self.size();
            let header = self.header_mut();
            header.size -= size;
            let ptr = (header as *mut FreeHeader as *mut u8).add(header.size);
            debug!(
                "Splitting {} bytes off from {:?}:{} to get {:?}",
                size,
                (header as *mut FreeHeader as *mut u8),
                self_size,
                ptr,
            );
            ptr
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

struct BlockList {
    first: Option<FreeBlock>,
}

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

impl BlockList {
    // Find and remove a chunk of size 'size' from the linked list
    fn pop_size(&mut self, size: usize) -> Option<*mut u8> {
        debug!("pop_size({})", size);

        let mut first = self.first?;
        debug!("  pop_size got first");
        if first.size() == size {
            debug!("  First block at {:?} is big enough", first.header);
            return Some(first.header as *mut u8);
        } else if first.size() >= size + HEADER_SIZE {
            let split = first.split(size);
            debug!(
                "  Split off from first block at {:?} to {:?}",
                first.header, split,
            );
            return Some(split);
        }

        let mut parent = first;
        loop {
            let mut next = parent.next()?;
            debug!("  Checking block at {:?} Size {}", next.header, next.size());

            if next.size() == size {
                let (ptr, _) = parent.pop_next();
                debug!("  Found correctly sized block at {:?}", ptr);
                return Some(ptr);
            }

            if next.size() < size + HEADER_SIZE {
                // This block is too small, skip it
                parent = parent.next().unwrap();
                continue;
            }

            // This block is bigger than we need, split it
            debug!("  Found big block at {:?}", next.header);
            return Some(next.split(size));
        }
    }

    // Add a block to the linked list. Takes ownership of ptr.
    unsafe fn add_block(&mut self, ptr: *mut u8, size: usize) {
        let mut new_block = FreeBlock::from_raw(ptr, None, size);

        let mut parent = match self.first {
            None => {
                self.first = Some(new_block);
                return;
            }
            Some(p) => p,
        };

        match new_block.relation(&parent) {
            Relation::Before => {
                new_block.header_mut().next = parent.header;
                self.first = Some(new_block);
                return;
            }
            Relation::AdjacentBefore => {
                new_block.header_mut().next = parent.header;
                let merged = new_block.try_merge_next();
                assert!(merged, "They were adjacent, they should merge");
                return;
            }
            Relation::Overlapping => {
                panic!("Overlapping memory blocks OH NO");
            }
            Relation::AdjacentAfter => {}
            _ => {}
        }

        loop {
            let next = match parent.next() {
                Some(n) => n,
                None => {
                    // We've reached the end of the list, let's append this to
                    // the list
                    parent.insert_merge(ptr, size);
                    return;
                }
            };

            if (next.header as *const u8) < ptr {
                parent = next;
                continue;
            }

            // If we are here, it means parent > ptr > next.
            // Time to insert_merge
            parent.insert_merge(ptr, size);
            return;
        }
    }

    #[allow(dead_code)]
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
}

trait HeapGrower {
    unsafe fn grow_heap(&mut self, size: usize) -> *mut u8;
}

struct UnixHeapGrower;

impl HeapGrower for UnixHeapGrower {
    unsafe fn grow_heap(&mut self, size: usize) -> *mut u8 {
        libc::sbrk(size as i32) as *mut u8
    }
}

struct BasicAlloc<G: HeapGrower> {
    grower: UnsafeCell<G>,
    blocks: UnsafeCell<BlockList>,
}

impl<G: HeapGrower + Default> Default for BasicAlloc<G> {
    fn default() -> Self {
        BasicAlloc {
            grower: UnsafeCell::from(G::default()),
            blocks: UnsafeCell::from(BlockList::default()),
        }
    }
}

impl<G: HeapGrower> BasicAlloc<G> {
    #[allow(dead_code)]
    pub fn new(grower: G) -> Self {
        BasicAlloc {
            grower: UnsafeCell::from(grower),
            blocks: UnsafeCell::from(BlockList::default()),
        }
    }

    unsafe fn new_block(&self, size: usize) -> *mut u8 {
        let grower = self.grower.get().as_mut().unwrap();
        grower.grow_heap(size)
    }

    fn aligned_size(layout: Layout) -> usize {
        // We align everything to 16 bytes, and all blocks are at least 16 bytes.
        // Its pretty wasteful, but easy!
        let layout = layout.align_to(16).expect("Whoa, serious memory issues");
        layout.pad_to_align().size() as usize
    }
}

unsafe impl<G: HeapGrower> GlobalAlloc for BasicAlloc<G> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let needed_size = BasicAlloc::<G>::aligned_size(layout);
        debug!("Allocating {} bytes", needed_size);

        let blocks_ptr = self.blocks.get();
        let blocks = match blocks_ptr.as_mut() {
            None => return self.new_block(needed_size),
            Some(b) => b,
        };

        if let Some(ptr) = blocks.pop_size(needed_size) {
            return ptr;
        }
        self.new_block(needed_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = BasicAlloc::<G>::aligned_size(layout);
        let blocks_ptr = self.blocks.get();
        let blocks = blocks_ptr
            .as_mut()
            .expect("Unexpected null pointer to BlockList");

        blocks.add_block(ptr, size);
    }
}

pub struct BasicUnixAlloc {
    alloc: BasicAlloc<UnixHeapGrower>,
}

unsafe impl GlobalAlloc for BasicUnixAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.alloc.dealloc(ptr, layout)
    }
}

pub struct ToyHeap {
    size: usize,
    heap: [u8; 1024],
}

impl Default for ToyHeap {
    fn default() -> Self {
        ToyHeap {
            size: 0,
            heap: [0; 1024],
        }
    }
}

impl HeapGrower for ToyHeap {
    unsafe fn grow_heap(&mut self, size: usize) -> *mut u8 {
        if self.size + size > self.heap.len() {
            panic!("Out of memory in this little toy heap!");
        }
        let ptr = self.heap.as_mut_ptr().add(self.size);
        self.size += size;
        ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use test_env_log::test;

    // Useful for debugging
    fn get_blocks<G: HeapGrower>(alloc: &BasicAlloc<G>) -> &BlockList {
        unsafe {
            alloc
                .blocks
                .get()
                .as_mut()
                .expect("There should be a block list here")
        }
    }

    #[test]
    fn test_basic() {
        let toy_heap = ToyHeap::default();
        let allocator = BasicAlloc::new(toy_heap);

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
            }
            pointers
        };

        for i in 0..BLOCKS - 1 {
            let l = layouts[i];
            let expected = unsafe { pointers[i].add(l.size()) };
            let found = pointers[i + 1];
            assert_eq!(expected, found);
        }

        let toy_heap = unsafe {
            allocator
                .grower
                .get()
                .as_mut()
                .expect("There should be a heap here")
        };
        // Toy heap should be the same size as the blocks requested
        assert_eq!(toy_heap.size, layouts.iter().map(|l| l.size()).sum());

        ////////////////////////////////////////////////////////////
        // Deallocation

        // Deallocate the second pointer
        unsafe { allocator.dealloc(pointers[1], layouts[1]) };

        // Check that the block list is as expected
        unsafe {
            let blocks = allocator
                .blocks
                .get()
                .as_mut()
                .expect("There should be a block list here");

            assert!(blocks.first.is_some());

            let first = blocks.first.as_mut().expect("This should not be null");
            assert_eq!(first.size(), layouts[1].size());
            assert!(first.next().is_none());

            // The block list now has 1 64-byte block on it
            log::info!("post-alloc: {}", blocks);
        };

        ////////////////////////////////////////////////////////////
        // Allocation with a block list
        unsafe {
            // Allocate 112 bytes, more than fits in the block on the block list
            let newp = allocator.alloc(Layout::from_size_align(112, 16).unwrap());
            assert_eq!(newp, pointers[2].add(layouts[2].size()));

            // Allocate 32 bytes, which should fit in the block
            let p32 = allocator.alloc(Layout::from_size_align(32, 16).unwrap());
            // The algorithm returns the second half of the block
            assert_eq!(p32, pointers[1].add(32));

            // We should now still have 32 bytes in 1 block in the block list

            // Allocate 8 bytes and another 16 bytes, which should both fit in the block
            // and completely consume it - because the 8 bytes should expand to 16
            log::info!("p32: {}", get_blocks(&allocator));
            let p8 = allocator.alloc(Layout::from_size_align(16, 4).unwrap());
            log::info!("p8: {}", get_blocks(&allocator));
            let p16 = allocator.alloc(Layout::from_size_align(8, 1).unwrap());
            // The algorithm returns the second half of the block
            log::info!("p16: {}", get_blocks(&allocator));
            assert_eq!(p8, pointers[1].add(16));
            assert_eq!(p16, pointers[1]);

            // And now our block list should be empty
            let blocks = allocator
                .blocks
                .get()
                .as_mut()
                .expect("There should be a block list here");

            assert!(blocks.first.is_none());
        };
    }
}
