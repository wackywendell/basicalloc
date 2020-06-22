use basic_allocator::UnixAllocator;

// This is the magic line that creates a new UnixAllocator and uses it globally.
#[global_allocator]
static ALLOCATOR: UnixAllocator = UnixAllocator::new();

fn main() {
    env_logger::init();
    println!("Hello, World!");

    // Let's create a vec, and add a bunch of things to it - forcing some
    // allocations
    let mut v = vec![];
    for n in 0..(1024 * 1024) {
        log::debug!("Pushing {}", n);
        v.push(n);
    }

    // And let's print out some allocator stats
    let (validity, stats) = ALLOCATOR.stats();
    println!("Valid: {:?}", validity);
    println!("Stats: {:?}", stats);
}
