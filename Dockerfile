FROM mcr.microsoft.com/devcontainers/rust:latest AS base

# strace is useful for inspecting syscalls when debugging the allocator
RUN apt-get update -y && apt-get install -y strace && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/basicalloc

# Dev stage: interactive development tools + permissions for vscode user.
# Used by the devcontainer — source is mounted at runtime.
FROM base AS dev
RUN cargo install just jj-cli
# cargo install runs as root; fix permissions so the vscode user
# can write to the cargo registry and build cache.
RUN chown -R vscode:vscode /usr/local/cargo

# CI stage: just the environment, no baked-in build.
# Tests are run by the CI pipeline or justfile, not the Dockerfile.
FROM base AS ci
COPY . .
