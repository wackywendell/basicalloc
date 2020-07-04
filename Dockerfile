FROM rustlang/rust:nightly as base
# This mostly just works without the Dockerfile:
# docker  run  -v `pwd`:/$(basename $PWD) -w /$(basename $PWD) -it  rustlang/rust:nightly
#
# Alternatively, you can use this Dockerfile.
# For development:
# docker build --target dev -t basicallocdev . && docker run -v `pwd`:/usr/src/basicalloc -it basicallocdev
# For use:
# docker build -t basicalloc . && docker run -it basicalloc

# We don't really need strace, but it is useful for this
RUN apt-get update -y
RUN apt-get install -y strace
RUN cargo install cargo-build-deps


FROM base as builder
WORKDIR /usr/src/basicalloc
COPY Cargo.toml Cargo.lock ./

# Fetch dependencies to create docker cache layer.
# Workaround with empty main to pass the build, which must be purged after.
# https://github.com/rust-lang/cargo/issues/2644
RUN  mkdir -p ./src \
    && echo 'fn main() { println!("Dummy") }' > ./src/main.rs \
    && cargo build \
    && cargo build --release \
    && cargo build --features "use_libc" \
    && cargo build --release --features "use_libc" \
    && cargo test \
    && cargo test --features "use_libc" \
    && rm -r src/main.rs target/debug/.fingerprint/basic_allocator-* target/release/.fingerprint/basic_allocator-*

# Cache layer with only my code
COPY ./ ./

# The real build.
RUN cargo build --frozen
RUN cargo build --frozen --release
RUN cargo build --frozen --tests
RUN cargo build --frozen --features "use_libc"
RUN cargo build --frozen --release --features "use_libc"
RUN cargo build --frozen --tests --features "use_libc"


FROM base as dev
WORKDIR /usr/src/basicalloc
COPY --from=builder /usr/src/basicalloc/target /usr/src/basicalloc/target


# Fetch and build dependencies using cargo-build-deps
# https://github.com/nacardin/cargo-build-deps
# RUN cd /tmp && USER=root cargo new --bin basicalloc
# WORKDIR /tmp/basicalloc
# COPY Cargo.toml Cargo.lock ./
# RUN cargo build-deps --release
# RUN cargo build-deps
# COPY src /tmp/basicalloc/src
# RUN cargo build  --release
# RUN cargo build

FROM dev as default
COPY ./src ./src
COPY ./examples ./examples
COPY ./tests ./tests
