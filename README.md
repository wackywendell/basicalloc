# A Basic Allocator

A home-grown memory allocator written entirely in Rust, meant primarily for
educational purposes. The code is heavily commented to explain how memory
allocation works at the OS level.

See the [documentation](https://docs.rs/basic_allocator) or
[the code itself](https://github.com/wackywendell/basicalloc) for details.

## How it works

The allocator requests virtual memory pages from the OS via `mmap`, then
manages them using a sorted linked list of free blocks. On allocation, it
searches the free list for a block that fits (splitting if too large). On
deallocation, it inserts the block back into the list and merges with adjacent
blocks to reduce fragmentation.

Rust's `GlobalAlloc::dealloc` passes the allocation size back to the allocator,
so unlike C's `malloc`/`free`, there's no need for per-allocation headers — the
header only exists on free blocks in the free list.

### Architecture

```
UnixAllocator              Thread-safe wrapper (spin lock)
  └─ RawAlloc              Allocation logic + free list
       ├─ BlockList         Sorted linked list of free blocks
       └─ MmapHeapGrower   Requests pages from the OS
            └─ syscall      Platform-specific mmap (asm or libc)
```

### Platform support

The `syscall` module provides `mmap` via inline assembly on:
- x86_64 Linux and macOS
- aarch64 Linux and macOS (Apple Silicon)

Alternatively, enable the `use_libc` feature to use libc's `mmap` on any
Unix platform.

## Usage

```rust
use basic_allocator::UnixAllocator;

#[global_allocator]
static ALLOCATOR: UnixAllocator = UnixAllocator::new();

fn main() {
    println!("It works!");
}
```

With `use_libc`:

```toml
[dependencies]
basic_allocator = { version = "0.1", features = ["use_libc"] }
```

## Development

### Quick start

```console
$ cargo test            # run tests
$ cargo run --example hello_world
$ cargo run --example grow_heap
$ cargo run --release --example stress_test
```

### Devcontainer (VS Code / Codespaces)

A devcontainer is included with Rust, rust-analyzer, just, and jj
pre-installed. Open the repo in VS Code and select "Reopen in Container", or
open it directly in GitHub Codespaces.

### Justfile

A [justfile](https://just.systems/) provides common recipes:

```console
$ just --list            # see all available recipes
$ just test              # run tests (default features)
$ just test-libc         # run tests with use_libc
$ just test-all          # both feature sets, tests + examples
$ just test-trace        # tests with RUST_LOG=trace output
$ just docker-test       # build + test in Docker (Linux x86_64)
$ just ci                # full CI: local + docker, both feature sets
```

### Logging

The allocator emits `log::trace!` messages during block splitting, merging, and
free list operations. These are zero-cost when no logger is installed. To see
them in tests:

```console
$ RUST_LOG=trace cargo test -- --nocapture
```

Or via the justfile:

```console
$ just test-trace
```

### Testing across platforms

Tests run on both macOS and Linux, with and without `use_libc`:

```console
$ just test-all          # local: both feature sets
$ just docker-test       # Docker: Linux x86_64, both feature sets
```

## Examples

- **`hello_world`** — Minimal example using `UnixAllocator` as the global
  allocator. Allocates a million-element Vec and prints allocator stats.
- **`grow_heap`** — Demonstrates the `HeapGrower` trait directly, showing
  how `MmapHeapGrower` requests memory from the OS.
- **`stress_test`** — Randomly allocates and deallocates objects of varying
  sizes, validating the free list integrity after each operation.
