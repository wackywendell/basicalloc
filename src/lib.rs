#![no_std]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ptr::null_mut;

#[derive(Clone)]
#[repr(C)]
struct MemoryHeader {
    next: *mut MemoryHeader,
    size: usize,
}

impl MemoryHeader {
    #[allow(clippy::cast_ptr_alignment)]
    unsafe fn from_raw(ptr: *mut u8, next: *mut MemoryHeader, size: usize) -> *mut MemoryHeader {
        let header = MemoryHeader { next, size };
        core::ptr::write_unaligned(ptr as *mut MemoryHeader, header);
        ptr as *mut MemoryHeader
    }

    unsafe fn data(&mut self) -> &mut [u8] {
        let ptr: *const MemoryHeader = self as *const MemoryHeader;

        let data_ptr = ptr.add(HEADER_SIZE) as *mut u8;

        core::slice::from_raw_parts_mut(data_ptr, self.size)
    }

    unsafe fn next_mut(&mut self) -> Option<&'static mut MemoryHeader> {
        self.next.as_mut()
    }
}

const HEADER_SIZE: usize = size_of::<MemoryHeader>();

struct BlockList {
    first: *mut MemoryHeader,
}

impl Default for BlockList {
    fn default() -> Self {
        BlockList { first: null_mut() }
    }
}

impl BlockList {
    // Find and remove a chunk of size 'size' from the linked list
    unsafe fn pop_size(&mut self, size: usize) -> Option<*mut u8> {
        let mut cur = self.first.as_mut()?;
        let mut parent: &mut *mut MemoryHeader = &mut self.first;
        loop {
            if cur.size < size {
                // Move down the list
                // set cur to next, or return None if cur.next is null
                let next = cur.next_mut()?;
                parent = &mut cur.next;
                cur = next;
                continue;
            }

            if cur.size + HEADER_SIZE == size {
                // Time to drop this block entirely, and return a pointer to it
                *parent = cur.next;
                // Convert cur from a pointer to the header to a pointer to the whole block
                return Some((cur as *mut MemoryHeader).cast());
            }

            // Time to cut this block into two pieces
            // We'll keep the first part the same, changing only the size,
            // and return the second half as the data
            cur.size -= size;
            let ptr: *mut u8 = cur.data().as_mut_ptr();
            return Some(ptr.add(cur.size));
        }
    }

    // Add a block to the linked list. Takes ownership of ptr.
    unsafe fn add_block(&mut self, ptr: *mut u8, size: usize) {
        if size < HEADER_SIZE {
            panic!("Can't recapture a block smaller than HEADER_SIZE");
        }
        let data_size = size - HEADER_SIZE;
        self.first = MemoryHeader::from_raw(ptr, self.first, data_size);
    }
}

trait HeapGrower {
    unsafe fn grow_heap(size: usize) -> *mut u8;
}

struct UnixHeapGrower;

impl HeapGrower for UnixHeapGrower {
    unsafe fn grow_heap(size: usize) -> *mut u8 {
        libc::sbrk(size as i32) as *mut u8
    }
}

struct BasicAlloc<G: HeapGrower> {
    grower: PhantomData<G>,
    blocks: UnsafeCell<BlockList>,
}

impl<G: HeapGrower> Default for BasicAlloc<G> {
    fn default() -> Self {
        BasicAlloc {
            grower: PhantomData,
            blocks: UnsafeCell::from(BlockList::default()),
        }
    }
}

impl<G: HeapGrower> BasicAlloc<G> {
    unsafe fn new_block(size: usize) -> *mut u8 {
        G::grow_heap(size)
    }

    fn aligned_size(layout: Layout) -> usize {
        // We align everything to 64 bytes.
        // Its pretty wasteful, but easy!
        let layout = layout.align_to(8).expect("Whoa, serious memory issues");
        layout.pad_to_align().size() as usize
    }
}

unsafe impl<G: HeapGrower> GlobalAlloc for BasicAlloc<G> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let needed_size = BasicAlloc::<G>::aligned_size(layout);

        let blocks_ptr = self.blocks.get();
        let blocks = match blocks_ptr.as_mut() {
            None => return BasicAlloc::<G>::new_block(needed_size),
            Some(b) => b,
        };

        if let Some(ptr) = blocks.pop_size(needed_size) {
            return ptr;
        }
        BasicAlloc::<G>::new_block(needed_size)
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
