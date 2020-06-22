//! An example program that uses the provided allocator as the global allocator,
//! creates and destroys a large number of objects, and checks validity along
//! the way.

use basic_allocator::UnixAllocator;

use rand::distributions::{Distribution, Uniform};
use rand::{Rng, RngCore, SeedableRng};

// This is the magic line that creates a new UnixAllocator and uses it globally.
#[global_allocator]
static ALLOCATOR: UnixAllocator = UnixAllocator::new();

const MIN_ALLOCATIONS: usize = 1024;
const ALLOCATIONS: usize = 64 * 1024;
const LOG2_MAX_SIZE: usize = 20;

#[derive(Default)]
struct RandomObjects {
    allocated: Vec<Vec<u64>>,
}

impl RandomObjects {
    fn create<R: Rng>(&mut self, rng: &mut R) {
        let range = Uniform::new_inclusive(8usize, LOG2_MAX_SIZE);
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
    env_logger::init();
    println!("Running Stress Test.");

    let seed: u64 = rand::thread_rng().next_u64();
    log::info!("Using seed {}", seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut objects = RandomObjects::default();

    for _ in 0..ALLOCATIONS {
        let mut allocate = true;

        if objects.allocated.len() >= MIN_ALLOCATIONS {
            allocate = rng.gen();
        }

        if allocate {
            objects.create(&mut rng);
        } else {
            objects.destroy(&mut rng);
        }

        let (validity, _) = ALLOCATOR.stats();
        assert!(validity.is_valid());
    }

    while !objects.allocated.is_empty() {
        objects.destroy(&mut rng);
        let (validity, _) = ALLOCATOR.stats();
        assert!(validity.is_valid());
    }

    let (validity, stats) = ALLOCATOR.stats();
    println!("Validity: {:?}", validity);
    println!("Stats:    {:?}", stats);
    assert!(validity.is_valid());
}
