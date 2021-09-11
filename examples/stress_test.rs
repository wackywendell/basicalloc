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
const STEPS: usize = 64 * 1024;
// Log_2 of the maximum sized array to allocate
const LOG2_MAX_SIZE: usize = 20;

const USAGE: &str = "[STEPS] [MIN_ALLOCATIONS] [LOG2_MAX_SIZE]

In a short loop, this will allocate and deallocate a total of STEPS times, with deallocations
occurring only if at least MIN_ALLOCATIONS have occurred.

Allocations will be between 1 and 2^LOG2_MAX_SIZE-1 bytes (inclusive), approximately logarithmically spaced.";

#[derive(Default)]
struct RandomObjects {
    allocated: Vec<Vec<u8>>,
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

    // Generate a random size to allocate between 1 and 2^self.
    fn rand_size<R: Rng>(&self, rng: &mut R) -> usize {
        // Generate a power of 2 up to half the max
        let n1 = 1usize << (rng.gen_range(0usize..self.log2_max_size));
        // Generate a second number to add to it, up to doubling it, to make it not a power of 2
        let n2 = rng.gen_range(0..n1);
        let new_size = n1 + n2;

        let expected_max: usize = ((1 << (self.log2_max_size - 1)) - 1) * 2 + 1;
        assert!(new_size < expected_max);
        // println!(
        //     "Size: {} + {} = {} / 2^{} ({})",
        //     n1, n2, new_size, self.log2_max_size, expected_max,
        // );

        new_size
    }

    fn create<R: Rng>(&mut self, rng: &mut R) {
        let new_size = self.rand_size(rng);

        // Allocate as many bytes as expected
        let obj: Vec<u8> = Vec::with_capacity(new_size);
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
        println!("USAGE: {} {}", args[0], USAGE);
        return;
    }
    let mut steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(STEPS);
    let min_allocations: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(MIN_ALLOCATIONS);
    if steps < min_allocations {
        steps = min_allocations;
    }
    let log2_max_size_arg: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(LOG2_MAX_SIZE);

    let log2_max_size = if log2_max_size_arg <= usize::BITS as usize {
        log2_max_size_arg
    } else {
        println!(
            "Allocation size of 2^{} bytes too large, using 2^{}",
            log2_max_size_arg,
            usize::BITS,
        );
        usize::BITS as usize
    };

    env_logger::init();
    println!("Running Stress Test.\n\nParameters:");
    println!("    {} total allocation steps", steps);
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

    for i in 1..=steps {
        // Decide if we should allocate some new objects, or destroy an old one.
        match allocation_run.cmp(&0) {
            std::cmp::Ordering::Equal => {
                let mut max_allocations = objects.allocated.len();
                if max_allocations < min_allocations {
                    max_allocations = min_allocations;
                }
                let max_deallocations = objects.allocated.len() as isize;
                let range = Uniform::new(-(max_deallocations as isize), max_allocations as isize);
                while allocation_run == 0 {
                    allocation_run = range.sample(&mut rng);
                    // println!("Running {} allocations", allocation_run);
                }
            }
            std::cmp::Ordering::Greater => {
                objects.create(&mut rng);
                allocation_run -= 1;
            }
            std::cmp::Ordering::Less => {
                objects.destroy(&mut rng);
                allocation_run += 1;
            }
        }

        let (validity, stats) = ALLOCATOR.stats();
        if i % 1024 == 0 {
            println!("Step {} / {}", i, steps);
            let count = objects.allocated.len();
            let total_size: usize = objects.allocated.iter().map(|v| v.len()).sum();
            println!("    Allocated objects: {}, size: {}", count, total_size);
            println!("    Allocator stats: {}", stats);
            if allocation_run >= 0 {
                println!("    Allocations in progress: {}", allocation_run);
            } else {
                println!("    Deallocations in progress: {}", -allocation_run);
            }
        }
        assert!(validity.is_valid());
    }

    println!(
        "Finished allocating, deallocating {} objects",
        objects.allocated.len(),
    );
    while !objects.allocated.is_empty() {
        objects.destroy(&mut rng);
        let (validity, _) = ALLOCATOR.stats();
        assert!(validity.is_valid());
    }

    let (validity, stats) = ALLOCATOR.stats();
    println!("\nFinished.");
    println!("    Stats: {}", stats);
    assert!(validity.is_valid());
}
