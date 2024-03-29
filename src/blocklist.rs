//! A linked list of memory blocks
//!
//! This module provides two main abstractions:
//! [`Blocklist`](struct.BlockList.html), a linked list of memory blocks, and
//! [`FreeBlock`](struct.FreeBlock.html), a memory block in that list.
//!
//! ## [`FreeBlock`](struct.FreeBlock.html)
//!
//! The basic unit in this module is the [`FreeBlock`](struct.FreeBlock.html). A
//! `FreeBlock` is a safe wrapper around a pointer to a memory block "owned" by
//! that `FreeBlock`, in the sense that the `FreeBlock` should have exclusive
//! read-write privileges for that memory block. These memory blocks could be
//! managed by a memory allocator, e.g. sections of a static array or boxed
//! arrays (see e.g. [`ToyHeap`](`../allocators/struct.ToyHeap.html`)), or as is
//! the case in this package, free memory on the heap.
//!
//! ### Internals
//!
//! Each `FreeBlock` consists of a non-null pointer to a section of memory. That
//! memory starts with a [`FreeHeader`](struct.FreeHeader.html), which consists
//! of an `Option<FreeBlock>` (e.g., a nullable pointer to the next memory
//! block) and a `size` denoting the size of the current block.
//!
//! `FreeBlock` does not implement `Copy` or `Clone`, because it should be the
//! exclusive pointer to a block of memory. Safe methods returning a `FreeBlock`
//! are marked with `#[must_use]`, as dropping a `FreeBlock` leaks memory.
//! `FreeBlock` implements `Drop` with a `debug_assert!(false)`, as it should
//! never be simply dropped; internally, we use `mem::forget` when necessary.
//!
//! ## [`Blocklist`](struct.BlockList.html)
//!
//! A [`Blocklist`](struct.BlockList.html) is a linked list of memory blocks,
//! called [`FreeBlock`](struct.FreeBlock.html)s. The ownership model is that
//! the `BlockList` owns the first [`FreeBlock`](struct.FreeBlock.html), and
//! each [`FreeBlock`](struct.FreeBlock.html) owns the next.
//!
//! As a consequence, `IterMut` cannot be safely implemented on `BlockList`, as
//! the list does not own the blocks directly, but indirectly, and changes to
//! one block can affect the structure of the list. Instead, a general-purpose
//! `apply` function is implemented.

use core::fmt::{self, Display};
use core::ops::Range;
use core::ptr::NonNull;

use static_assertions::const_assert;

/// The header for our free blocks.
///
/// The header includes a pointer to the next free block, and the size of the
/// current block (including the header).
///
/// We use C representation and align to 16 bytes for... simplicity. This is
/// perhaps a stronger constraint that we need, but it does make things simple
/// and straightforward.
#[repr(C, align(16))]
pub struct FreeHeader {
    // A link to the next memory block. Note that `Option<FreeBlock>` is an
    // abstraction around a simple pointer.
    next: Option<FreeBlock>,
    // The size of the current block, header included. Should never be less than
    // `HEADER_SIZE`.
    size: usize,
}

/// We will align to 16 bytes and our headers will be given that much space
/// Similarly, all blocks will be at least 16 bytes large, even if they aren't
/// aware of it.
///
/// This is likely a stronger constraint than is entirely needed, but it does
/// simplify things.
const HEADER_SIZE: usize = 16;

// The constant we use for HEADER_SIZE needs to be bigger than the contents of a
// header, so we assert that here.
const_assert!(HEADER_SIZE >= core::mem::size_of::<FreeHeader>());

/// An enum for easy comparison of blocks and their order
pub enum Relation {
    Before,
    AdjacentBefore,
    Overlapping,
    AdjacentAfter,
    After,
}

impl FreeHeader {
    /// Construct a header from a freed memory block at `ptr`, with a link to
    /// the next in `next`, and the size of the block in `size`.
    ///
    /// # Safety
    ///
    /// This is unsafe because its manipulating raw, freed memory.
    ///
    /// To use this safely, `ptr` must point to memory of size `size` not in use
    /// by or accessible by any program logic.
    ///
    /// Further safety constraints are enforced by the invariants of `FreeBlock`
    /// and `BlockList`.
    #[allow(clippy::cast_ptr_alignment)]
    pub unsafe fn from_raw(
        ptr: NonNull<u8>,
        next: Option<FreeBlock>,
        size: usize,
    ) -> NonNull<FreeHeader> {
        let header = FreeHeader { next, size };
        let raw_ptr: NonNull<FreeHeader> = ptr.cast();
        core::ptr::write(ptr.as_ptr() as *mut FreeHeader, header);
        raw_ptr
    }
}

/// A `FreeBlock` is a wrapper around a pointer to a freed block to be
/// maintained in a [`BlockList`](struct.BlockList.html).
///
/// Invariants are enforced by / inherited from the NonNull strict.
///
/// Note that this is very similar to Box, except that it doesn't assume a heap
/// or memory allocator, so it doesn't implement Clone or Drop, and it also has
/// a 'next'.
pub struct FreeBlock {
    header: NonNull<FreeHeader>,
}

impl Drop for FreeBlock {
    fn drop(&mut self) {
        debug_assert!(false, "Leaking memory!!");
    }
}

impl FreeBlock {
    /// Construct a `FreeBlock` from raw parts: a freed memory block at `ptr` of
    /// size `size`. This will also write the header appropriately.
    ///
    /// # Safety
    ///
    /// This is unsafe because its manipulating raw, freed memory.
    ///
    /// To use this safely, `ptr` must point to memory of size `size` not in use
    /// by or accessible by any program logic.
    ///
    /// Further safety constraints are enforced by the invariants of `BlockList`.
    #[must_use]
    pub unsafe fn from_raw(ptr: NonNull<u8>, next: Option<FreeBlock>, size: usize) -> FreeBlock {
        debug_assert!(
            size >= HEADER_SIZE,
            "Can't recapture a block smaller than HEADER_SIZE"
        );
        let header = FreeHeader::from_raw(ptr, next, size);
        FreeBlock { header }
    }

    /// Get the memory covered by this block as a slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let size = self.header_view().size;
            core::slice::from_raw_parts(self.header.as_ptr() as *const u8, size)
        }
    }

    /// Get the pointer range covered by this block.
    pub fn as_range(&self) -> Range<*const u8> {
        unsafe {
            let size = self.header_view().size;
            let start = self.header.as_ptr() as *const u8;
            start..(start.add(size))
        }
    }

    /// Consume this block and return the range of memory covered, and the next
    /// block in the list.
    #[must_use]
    pub fn decompose(mut self) -> (Range<NonNull<u8>>, Option<FreeBlock>) {
        let next = self.take_next();
        let range = unsafe {
            let size = self.header_view().size;
            let start: NonNull<u8> = self.header.cast();
            let end: NonNull<u8> =
                NonNull::new_unchecked(self.header.as_ptr().add(size) as *mut u8);
            start..end
        };

        // We've decomposed self, so now we forget about it and do not call Drop.
        core::mem::forget(self);

        (range, next)
    }

    /// Compare two blocks to see how they are ordered.
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

    /// Get the next block over from this one.
    fn next(&self) -> Option<&Self> {
        (&self.header_view().next).into()
    }

    /// Get the next block over from this one.
    fn next_mut(&mut self) -> Option<&mut Self> {
        unsafe { (&mut self.header_mut().next).into() }
    }

    /// Remove the next, and return it
    #[must_use]
    fn take_next(&mut self) -> Option<Self> {
        unsafe { (&mut self.header_mut().next).take() }
    }

    /// Set this block's next to new_next, and return the old one
    #[must_use]
    fn replace_next(&mut self, new_next: FreeBlock) -> Option<Self> {
        unsafe { (&mut self.header_mut().next).replace(new_next) }
    }

    /// The size of the block, in bytes.
    pub fn size(&self) -> usize {
        self.header_view().size
    }

    /// An immutable pointer to the header
    pub fn header_view(&self) -> &FreeHeader {
        unsafe { self.header.as_ref() }
    }

    /// Get a mutable view of the header.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it allows modifying the size or pointer of
    /// a free block in safe code, which could lead to corruption.
    pub unsafe fn header_mut(&mut self) -> &mut FreeHeader {
        self.header.as_mut()
    }

    /// Remove the block after this one from the linked list, and return
    /// a pointer to that block and its size.
    ///
    /// As is required in a linked list, this will set self.next = next.next.
    ///
    /// If there is no next, returns (None, 0).
    #[must_use]
    pub fn pop_next(&mut self) -> Option<FreeBlock> {
        let mut next = match self.take_next() {
            None => {
                return None;
            }
            Some(n) => n,
        };

        // Update this block to look to next's next, cutting next out of the chain
        if let Some(next_next) = next.take_next() {
            assert!(self.replace_next(next_next).is_none());
        }

        Some(next)
    }

    /// Insert a new element, after this one, maintaining linked list invariants.
    ///
    /// `block` must have no `next`, or an assertion will fail.
    pub fn insert(&mut self, block: FreeBlock) {
        let next_next = self.replace_next(block);
        let inserting_next = match next_next {
            None => self.next_mut().unwrap().take_next(),
            Some(next_next_block) => self.next_mut().unwrap().replace_next(next_next_block),
        };
        assert!(inserting_next.is_none());
    }

    /// Insert a new element, after this one, maintaining linked list invariants
    /// and merging with either this item and/or the next, depending on
    /// adjacency.
    ///
    /// `block.next()` must be null, or an assertion will fail.
    pub fn insert_merge(&mut self, block: FreeBlock) -> usize {
        let this_end = self.as_range().end;
        let other_start = block.as_range().start;
        assert!(block.next().is_none());

        let (merges, try_next) = if this_end == other_start {
            // We can merge `block` into `self`, as it follows this one.
            let new_size = block.size();
            // Extend this block to cover that one
            unsafe {
                self.header_mut().size += new_size;
                // `block` has now been incorporated into `self`, so we forget
                // about it as a separate entity.
                core::mem::forget(block);
            }
            (1, self)
        } else {
            self.insert(block);
            (0, self.next_mut().unwrap())
        };

        merges + if try_next.try_merge_next() { 1 } else { 0 }
    }

    /// Split off part of this FreeBlock, and return a pointer to the split off
    /// data.
    ///
    /// The returned pointer is to a region of size 'size' that is no longer
    /// considered free.
    ///
    /// Panics if 'size' is greater than this block's size - HEADER_SIZE, as
    /// there is no way to split off a chunk that large while leaving behind a
    /// FreeBlock with an intact header.
    pub fn split(&mut self, size: usize) -> Range<NonNull<u8>> {
        debug_assert!(
            size + HEADER_SIZE <= self.header_view().size,
            "Can't split a block of size {} off of a block of size {} - need {} for header",
            size,
            self.header_view().size,
            HEADER_SIZE
        );

        unsafe {
            let self_size = self.size();
            let header = self.header_mut();
            header.size -= size;
            let start =
                NonNull::new_unchecked((header as *mut FreeHeader as *mut u8).add(header.size));
            let end = NonNull::new_unchecked((header as *mut FreeHeader as *mut u8).add(self_size));
            // log::trace!(
            //     "Splitting {} bytes off from {:?}:{} to get {:?}",
            //     size,
            //     (header as *mut FreeHeader as *mut u8),
            //     self_size,
            //     ptr,
            // );

            start..end
        }
    }

    /// Attempt to merge this block with the next.
    ///
    /// If the next block exaists, is adjacent, and exists directly after this
    /// block, the two will merge and this will return True; otherwise, this will
    /// return False.
    pub fn try_merge_next(&mut self) -> bool {
        let (next_start, next_size) = match self.next() {
            None => return false,
            Some(block) => (block.as_range().start, block.size()),
        };
        if self.as_range().end != next_start {
            return false;
        };

        unsafe {
            let header = self.header_mut();
            header.size += next_size;
            let mut next = header.next.take().unwrap();
            header.next = next.take_next();
            // We've incorporated the memory from 'next' into self, so we need
            // to forget about it as a separate entity
            core::mem::forget(next);
        }

        true
    }
}

/// A `BlockList` is a linked list of "free" blocks in memory.
///
/// Each block should be considered "owned" by the BlockList when inserted, and
/// do not hold any sort of payload. They may be split or merged internally.
///
/// In this module, thse memory blocks represent freed memory that has not been
/// returned to the OS, and provide a "pool" of available memory for reuse by
/// the allocator.
///
/// It maintains a few internal invariants:
///
/// - Each block should link to the next, with the last one linking to null.
/// - Each block should have a pointer < next.
/// - No two blocks should be precisely adjacent (those should be automatically
///   merged on insertion).
pub struct BlockList {
    first: Option<FreeBlock>,
}

pub struct BlockIter<'list> {
    next: Option<&'list FreeBlock>,
}

impl<'list> Iterator for BlockIter<'list> {
    type Item = &'list FreeBlock;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next.take()?;

        self.next = next.next();

        Some(next)
    }
}

// A BlockList is sendable - as long as the whole "chain" is maintained across
// threads, its fine.
//
// With some tweaking and atomic pointer swapping, we could make a thread-safe
// version of BlockList, but that has not been done here; hence, it implements
// Send but not Sync.
unsafe impl Send for FreeBlock {}

impl Default for BlockList {
    fn default() -> Self {
        BlockList { first: None }
    }
}

impl<'list> IntoIterator for &'list BlockList {
    type Item = &'list FreeBlock;
    type IntoIter = BlockIter<'list>;

    fn into_iter(self) -> Self::IntoIter {
        BlockIter {
            next: self.first.as_ref(),
        }
    }
}

impl fmt::Display for BlockList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockList(")?;
        let mut start = true;
        for block in self {
            if !start {
                write!(f, ", ")?;
            } else {
                start = false;
            }
            write!(f, "FreeBlock({:?}, {})", block.header, block.size())?;
        }

        write!(f, ")")
    }
}

/// Validity contains a representation of all invalid states found in a
/// BlockList.
#[derive(Default, Debug)]
pub struct Validity {
    /// Number of blocks overlapping other blocks.
    ///
    /// This likely indicates corruption.
    ///
    /// If there are also out of order blocks, this might undercount.
    pub overlaps: usize,

    /// Number of blocks that are directly adjacent to each other, and not
    /// merged. This shouldn't happen, but isn't totally corrupt.
    pub adjacents: usize,
    /// Number of blocks that do not have an address less than their next.
    ///
    /// This shouldn't occur.
    pub out_of_orders: usize,
}

impl Validity {
    /// Returns a boolean - a simple check if all cases are 0
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

impl Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{} empty blocks in allocator, total size {} bytes",
            self.length, self.size
        )
    }
}

/// State after a single "apply".
pub enum ApplyState<C, R> {
    // Keep going, and pass C into the next 'apply'
    Continue(C),
    // Finish iterating, and return the value R
    Finished(R),
    // Finish iterating, and return the value C
    Fail(C),
}

impl BlockList {
    pub const fn header_size() -> usize {
        HEADER_SIZE
    }

    pub fn iter(&self) -> BlockIter {
        BlockIter {
            next: self.first.as_ref(),
        }
    }

    /// Iterate through the blocklist, and apply a function at each step. This
    /// allows mutating the list as it is traversed, and replaces IterMut, which
    /// cannot be used due to the links between blocks.
    ///
    /// Note that any changes to any block's "next" will be followed at the next
    /// iteration.
    pub fn apply<C, R, F: FnMut(&mut FreeBlock, C) -> ApplyState<C, R>>(
        &mut self,
        start: C,
        mut pred: F,
    ) -> ApplyState<C, R> {
        let mut next = self.first.as_mut();

        let mut state = start;
        while let Some(block) = next.take() {
            state = match pred(block, state) {
                ApplyState::Continue(c) => c,
                ApplyState::Finished(r) => return ApplyState::Finished(r),
                ApplyState::Fail(c) => return ApplyState::Fail(c),
            };
            next = block.next_mut()
        }

        ApplyState::Continue(state)
    }

    /// Check current size of the list, and whether its valid.
    pub fn stats(&self) -> (Validity, Stats) {
        let mut validity: Validity = Default::default();
        let mut stats: Stats = Default::default();

        let mut previous: Option<&FreeBlock> = None;
        for next in self.iter() {
            match previous.map(|p| p.relation(next)) {
                Some(Relation::Before) => {
                    // This is valid, do nothing.
                }
                Some(Relation::AdjacentBefore) => {
                    // Right order, but these should be merged.
                    validity.adjacents += 1;
                }
                Some(Relation::Overlapping) => {
                    // This is really bad.
                    validity.overlaps += 1;
                }
                Some(Relation::AdjacentAfter) => {
                    // Wrong order, and these should be merged.
                    validity.out_of_orders += 1;
                    validity.adjacents += 1;
                }
                Some(Relation::After) => {
                    // Wrong order.
                    validity.out_of_orders += 1;
                }
                None => {
                    // This is the first in the list. Valid, do nothing.
                }
            }

            stats.length += 1;
            stats.size += next.size();
            previous = Some(next);
        }

        (validity, stats)
    }

    /// Find and remove a chunk of size 'size' from the linked list
    pub fn pop_size(&mut self, size: usize) -> Option<Range<NonNull<u8>>> {
        // First, check if we should pop off the first block
        let first_size = self.first.as_ref()?.size();
        if first_size == size {
            // The first block is just right, let's use it
            let (range, next) = self.first.take()?.decompose();
            self.first = next;
            return Some(range);
        } else if first_size >= size + HEADER_SIZE {
            // The first block is bigger than needed, let's use a portion of it
            let split = self.first.as_mut()?.split(size);
            return Some(split);
        }

        // Iterate through the rest of the blocks, and see if they fit
        let applied = self.apply((), |previous, ()| {
            let next_size: usize = match previous.next() {
                None => return ApplyState::Fail(()),
                Some(next) => next.size(),
            };
            // log::trace!("  Checking block at {:?} Size {}", next.header, next.size());

            if next_size == size {
                // This block is just right - let's pop it out of the chain and return it
                let block = previous.pop_next().unwrap();
                let (range, next) = block.decompose();
                assert!(next.is_none());
                return ApplyState::Finished(range);
                // log::trace!("  Found correctly sized block at {:?}", ptr);
            }

            if next_size < size + HEADER_SIZE {
                // This block is too small to be split, skip it
                return ApplyState::Continue(());
            }

            // This block is bigger than we need, split it
            // log::trace!("  Found big block at {:?}", next.header);
            let ptr = previous.next_mut().unwrap().split(size);
            ApplyState::Finished(ptr)
        });

        match applied {
            ApplyState::Continue(()) => None,
            ApplyState::Fail(()) => None,
            ApplyState::Finished(ptr) => Some(ptr),
        }
    }

    /// Add a block to the linked list. Takes ownership of ptr.
    ///
    /// # Safety
    ///
    /// `ptr` must point to valid, reachable memory of at least `size`, and
    /// ownership of that memory must be transferred to `BlockList` when this
    /// method is called.
    pub unsafe fn add_block(&mut self, ptr: NonNull<u8>, size: usize) {
        let mut new_block = FreeBlock::from_raw(ptr, None, size);

        let first: &FreeBlock = match self.first {
            None => {
                // There are no blocks in this list, so we make this the head of
                // the list and return
                self.first = Some(new_block);
                return;
            }
            Some(ref p) => p,
        };

        // We keep the list in sorted order, by pointer, to enable merging.
        match new_block.relation(first) {
            Relation::Before => {
                // This block is well before the first one in the list, so we
                // add this to the head of the list
                match self.first.take() {
                    None => {}
                    Some(b) => assert!(new_block.replace_next(b).is_none()),
                };
                self.first = Some(new_block);
                return;
            }
            Relation::AdjacentBefore => {
                // This block is just before the first block in the list, so we
                // merge the two into a single block
                match self.first.take() {
                    None => {}
                    Some(b) => assert!(new_block.replace_next(b).is_none()),
                };
                let merged = new_block.try_merge_next();
                self.first = Some(new_block);
                assert!(merged, "They were adjacent, they should merge");
                return;
            }
            Relation::Overlapping => {
                // These blocks both claim the same memory
                debug_assert!(false, "Overlapping memory blocks OH NO");
            }
            Relation::AdjacentAfter => {
                // This block is just after the first block in the list, so we
                // merge the two into a single block. This block isn't part of
                // the list yet, and 'previous' already correctly points to the
                // next block, so all we need to do is increase the 'previous'
                // block size.
                let first = self.first.as_mut().unwrap();
                first.header_mut().size += size;
                // Now we forget `block` as a separate entity, because its been
                // absorbed
                core::mem::forget(new_block);
                // Now that 'previous' has grown, it's possible that 'previous'
                // is now adjacent to 'next', so we try and merge them. This may
                // or may not actually happen, and either way, we're left with a
                // valid list afterwards.
                first.try_merge_next();
                return;
            }
            _ => {}
        }

        // Loop through the list of blocks, to find where this one should be
        // inserted. Once its place in the list is found, we merge with the
        // previous and/or next if we can, and if not, insert it into
        // the list.
        let applied = self.apply(new_block, |previous, new_block| {
            // By construction, previous < new_block. Now we check previous.next
            // to see if previous < new_block < next, in which case we insert
            // and merge, or if next < new_block, we continue iterating through
            // the list.
            let next = match previous.next() {
                Some(n) => n,
                None => {
                    // previous < new_block, and there's no more blocks after
                    // previous, so we insert here.
                    previous.insert_merge(new_block);
                    return ApplyState::Finished(());
                }
            };

            if next.header.cast() < ptr {
                // next < pointer, so we continue iterating
                return ApplyState::Continue(new_block);
            }

            // If we are here, it means previous < ptr < next.
            // Time to insert_merge
            previous.insert_merge(new_block);
            ApplyState::Finished(())
        });

        // And verify that we actually inserted
        match applied {
            ApplyState::Finished(()) => (),
            ApplyState::Fail(_) => unreachable!(),
            ApplyState::Continue(_) => unreachable!(),
        };
    }

    pub fn len(&self) -> usize {
        self.iter().count()
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_none()
    }
}
