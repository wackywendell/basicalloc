use basicalloc::BasicUnixAlloc;

#[global_allocator]
static A: BasicUnixAlloc = BasicUnixAlloc::new();

fn main() {
    env_logger::init();
    println!("Hello, World!");

    let s: String = "abc".to_owned();
    println!("Got a string {}", s);

    let mut v = vec![0, 1, 2, 3];
    for n in 10..2048 {
        // log::debug!("Pushing {}", n);
        v.push(n);
    }
    println!("Got a vec {}", v.len());
}
