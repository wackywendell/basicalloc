# A Basic Allocator

This package includes a home-grown memory allocator written entirely in Rust. It
is simple, and meant primarily for educational purposes.

This crate is heavily commented and documented. See the
[documentation](https://docs.rs/basic_allocator) or [the code
itself](https://github.com/wackywendell/basicalloc) for more details.

## Development

Development can happen locally on an OSX or Linux machine, using the standard Rust frameworks. In addition, Docker can be used for linux:

```console
$ # Develop using the code mounted
$ docker build --target dev -t basicallocdev . && docker run -v `pwd`:/usr/src/basicalloc -it basicallocdev
[...]
root@0123456789ab:/usr/src/basicalloc# cargo test
[...]
```
