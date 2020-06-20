extern crate basicalloc;

use basicalloc::BasicUnixAlloc;

#[global_allocator]
static A: BasicUnixAlloc = BasicUnixAlloc::new();

fn main() {
    println!("Hello, World!");
}
