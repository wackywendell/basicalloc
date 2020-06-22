//! An example program that uses the provided allocator as the global allocator,
//! creates and destroys a large number of objects, and checks validity along
//! the way.

use basic_allocator::UnixAllocator;

use rand::distributions::{Distribution, Uniform};
use rand::{Rng, RngCore, SeedableRng};

// This is the magic line that creates a new UnixAllocator and uses it globally.
#[global_allocator]
static ALLOCATOR: UnixAllocator = UnixAllocator::new();

// Minimum number of allocations before we start deallocating
const MIN_ALLOCATIONS: usize = 1024;
// Total number of allocations / deallocations
const ALLOCATIONS: usize = 64 * 1024;
// Log_2 of the maximum sized array to allocate
const LOG2_MAX_SIZE: usize = 20;

#[derive(Default)]
struct RandomObjects {
    allocated: Vec<Vec<u64>>,
    log2_max_size: usize,
}

impl RandomObjects {
    fn new(log2_max_size: usize) -> Self {
        let max = if log2_max_size < 8 { 8 } else { log2_max_size };

        RandomObjects {
            allocated: Vec::new(),
            log2_max_size: max,
        }
    }

    fn create<R: Rng>(&mut self, rng: &mut R) {
        let range = Uniform::new_inclusive(8usize, self.log2_max_size);
        let new_size = (range.sample(rng) * range.sample(rng)) as u64;
        let obj: Vec<u64> = (0..new_size).collect();
        self.allocated.push(obj);
    }

    fn destroy<R: Rng>(&mut self, rng: &mut R) {
        if self.allocated.is_empty() {
            return;
        }
        let range = Uniform::new(0, self.allocated.len());
        let ix = range.sample(rng);
        let obj = self.allocated.swap_remove(ix);

        drop(obj);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.contains(&"--help".to_owned()) {
        println!(
            "USAGE: {} [ALLOCATIONS] [MIN_ALLOCATIONS] [LOG2_MAX_SIZE]",
            args[0]
        );
        return;
    }
    let mut allocations: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(ALLOCATIONS);
    let min_allocations: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(MIN_ALLOCATIONS);
    if allocations < min_allocations {
        allocations = min_allocations;
    }
    let log2_max_size: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(LOG2_MAX_SIZE);

    env_logger::init();
    println!("Running Stress Test.\n\nParameters:");
    println!("    {} total allocations", allocations);
    println!(
        "    {} allocations before any deallocations",
        min_allocations
    );
    println!("    2^{} max allocated object size", log2_max_size);

    let seed: u64 = rand::thread_rng().next_u64();
    log::info!("Using seed {}", seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut objects = RandomObjects::new(log2_max_size);

    let mut allocation_run: isize = min_allocations as isize;

    for i in 1..=allocations {
        // Decide if we should allocate some new objects, or destroy an old one.
        while allocation_run == 0 {
            let mut max_allocations = objects.allocated.len();
            if max_allocations < min_allocations {
                max_allocations = min_allocations;
            }
            let max_deallocations = objects.allocated.len() as isize;
            let range = Uniform::new(-(max_deallocations as isize), max_allocations as isize);
            allocation_run = range.sample(&mut rng);
            // println!("Running {} allocations", allocation_run);
        }

        if allocation_run > 0 {
            objects.create(&mut rng);
            allocation_run -= 1;
        } else {
            objects.destroy(&mut rng);
            allocation_run += 1;
        }

        let (validity, stats) = ALLOCATOR.stats();
        if i % 1024 == 0 {
            println!("Step {} / {}", i, allocations);
            let count = objects.allocated.len();
            let total_size: usize = objects.allocated.iter().map(|v| v.len()).sum();
            println!("    Allocated objects: {}, size: {}", count, total_size);
            println!("    Allocator stats: {:?}", stats);
            println!("    Allocations in progress: {}", allocation_run);
        }
        assert!(validity.is_valid());
    }

    while !objects.allocated.is_empty() {
        objects.destroy(&mut rng);
        let (validity, _) = ALLOCATOR.stats();
        assert!(validity.is_valid());
    }

    let (validity, stats) = ALLOCATOR.stats();
    println!("\nFinished.");
    println!("    Stats:    {:?}", stats);
    assert!(validity.is_valid());
}
