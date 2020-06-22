use core::alloc::Layout;
use core::ptr::null_mut;

use basic_allocator::allocators::{RawAlloc, ToyHeap};

use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use test_env_log::test;

#[test]
fn test_stress() {
    let toy_heap = ToyHeap::default();
    let mut allocator = RawAlloc::new(toy_heap);
    let null_layout = Layout::new::<usize>();

    // Create a new array of pointers
    // Note: the null pointer means not allocated; the layout is meaningless
    let mut pointers: [(*mut u8, Layout); 128] = [(null_mut(), null_layout); 128];
    let mut _allocated_count: usize = 0;
    let mut allocated_size: usize = 0;
    let mut _freed_count: usize = 0;
    let mut freed_size: usize = 0;

    fn validate(allocator: &RawAlloc<ToyHeap>, allocated_size: usize, freed_size: usize) {
        let (validity, stats) = allocator.stats();
        log::info!(
            "Allocated: {}, Freed: {}; heap_size: {}; Validity: {:?}, Stats: {:?}",
            allocated_size,
            freed_size,
            allocator.grower.size,
            validity,
            stats,
        );
        log::info!("Blocks: {}", allocator.blocks);
        assert!(validity.is_valid());

        let found_heap_size = allocator.grower.size;
        let found_freed = stats.size;
        assert_eq!(allocated_size - freed_size, found_heap_size - found_freed);
    }

    let seed: u64 = rand::thread_rng().next_u64();
    log::info!("Using seed {}", seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let range = rand::distributions::Uniform::new_inclusive(1usize, 32);

    for _ in 0..1024 * 10 {
        let chosen = pointers.choose_mut(&mut rng).unwrap();
        let &mut (ptr, layout) = chosen;
        if ptr.is_null() {
            // Let's try allocating
            let new_size = range.sample(&mut rng) * range.sample(&mut rng);
            let &align = [1usize, 2, 4, 8, 16].choose(&mut rng).unwrap();
            let new_layout = Layout::from_size_align(new_size, align).unwrap();
            log::info!("Allocating {}@{}", new_layout.size(), new_layout.align());
            let new_ptr = unsafe { allocator.alloc(new_layout) };
            log::info!(
                "  Allocated {:?} {}@{}",
                new_ptr,
                new_layout.size(),
                new_layout.align()
            );
            *chosen = (new_ptr, new_layout);
            allocated_size += RawAlloc::<ToyHeap>::block_size(new_layout);
            _allocated_count += 1;
        } else {
            // Let's try freeing
            log::info!(
                "Deallocating {:?} {}@{}",
                ptr,
                layout.size(),
                layout.align()
            );
            unsafe { allocator.dealloc(ptr, layout) };
            *chosen = (null_mut(), null_layout);

            freed_size += RawAlloc::<ToyHeap>::block_size(layout);
            _freed_count += 1;
        }

        // And validate that everything is ok
        validate(&allocator, allocated_size, freed_size);
    }
}
