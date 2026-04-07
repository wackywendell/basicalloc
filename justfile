# List available recipes
default:
    @just --list

# Check compilation (default features)
check:
    cargo check

# Check compilation with use_libc
check-libc:
    cargo check --features use_libc

# Run tests (default features)
test:
    cargo test

# Run tests with use_libc
test-libc:
    cargo test --features use_libc

# Run tests with trace logging visible
test-trace:
    RUST_LOG=trace cargo test -- --nocapture

# Run all examples
examples:
    cargo run --example hello_world
    cargo run --example grow_heap
    cargo run --release --example stress_test -- 4096 512 16

# Run all examples with use_libc
examples-libc:
    cargo run --features use_libc --example hello_world
    cargo run --features use_libc --example grow_heap
    cargo run --features use_libc --release --example stress_test -- 4096 512 16

# Run everything locally: both feature sets, tests + examples
test-all: test test-libc examples examples-libc

# Shared volumes for caching cargo registry and builds across docker runs
docker-run := "docker run --rm -v basicalloc-target:/usr/src/basicalloc/target -v basicalloc-cargo:/usr/local/cargo/registry"

# Build the Docker CI image
docker-build:
    docker build --target ci -t basicalloc-ci .

# Run tests in Docker (Linux x86_64), both feature sets
docker-test: docker-build
    {{ docker-run }} basicalloc-ci cargo test
    {{ docker-run }} basicalloc-ci cargo test --features use_libc

# Run examples in Docker (Linux x86_64), both feature sets
docker-examples: docker-build
    {{ docker-run }} basicalloc-ci cargo run --example hello_world
    {{ docker-run }} basicalloc-ci cargo run --example grow_heap
    {{ docker-run }} basicalloc-ci cargo run --features use_libc --example hello_world
    {{ docker-run }} basicalloc-ci cargo run --features use_libc --example grow_heap

# Full CI-style check: local + docker, both feature sets
ci: test-all docker-test docker-examples
