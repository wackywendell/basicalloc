FROM rustlang/rust:nightly

# This mostly just works without the Dockerfile:
# docker  run  -v `pwd`:/$(basename $PWD) -w /$(basename $PWD) -it  rustlang/rust:nightly
#
# Alternatively, you can use this Dockerfile:
# docker build -t basicalloc . && docker run --rm -it basicalloc

WORKDIR /usr/src/basicalloc
COPY . .

# We don't really need strace, but it is useful for this
RUN apt-get update -y
RUN apt-get install -y strace

RUN cargo build