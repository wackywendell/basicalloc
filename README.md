# A Basic Allocator

This package includes a home-grown memory allocator written entirely in Rust. It
is simple, and meant primarily for educational purposes.

This crate is heavily commented and documented. See the
[documentation](https://docs.rs/basic_allocator) or [the code
itself](https://github.com/wackywendell/basicalloc) for more details.

## Development

Development can happen locally on macOS or Linux using standard Rust tooling.
A [justfile](https://just.systems/) provides common recipes:

```console
$ just test          # run tests (default features)
$ just test-libc     # run tests with use_libc feature
$ just test-all      # both feature sets, tests + examples
$ just docker-test   # run tests in Docker (Linux x86_64)
$ just ci            # full CI: local + docker, both feature sets
$ just --list        # see all available recipes
```

A devcontainer configuration is included for VS Code / GitHub Codespaces.

## Examples

Several examples live in the `examples` directory:

```console
$ cargo run --example hello_world
$ cargo run --example grow_heap
$ cargo run --release --example stress_test
```