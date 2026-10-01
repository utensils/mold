#!/usr/bin/env bash
set -euo pipefail
# Standalone relay only: no GPU/Candle build, static libc for Amazon Linux.
relay_root=$(cd "$(dirname "$0")/../.." && pwd)
relay_image=rust:1.95-slim-bookworm@sha256:d7482085ff5b415f84dba5647ae71606650bdef00db7aeb69f4b3d170c3e4082
mkdir -p "$relay_root/tmp/relay-linux"
docker run --rm --platform linux/amd64 \
  -v "$relay_root:/src:ro" -v "$relay_root/tmp/relay-linux:/out" \
  "$relay_image" bash -euc '
    apt-get update -qq
    apt-get install -y --no-install-recommends musl-tools ca-certificates
    rustup target add x86_64-unknown-linux-musl
    cp -a /src/Cargo.toml /src/Cargo.lock /tmp/
    cp -a /src/crates /tmp/
    cd /tmp
    CARGO_TARGET_DIR=/out/target cargo build --locked --release -p mold-ai-relay --bin mold-relay --target x86_64-unknown-linux-musl
    install -m 755 /out/target/x86_64-unknown-linux-musl/release/mold-relay /out/mold-relay
    sha256sum /out/mold-relay > /out/mold-relay.sha256
  '
printf 'Built %s\n' "$relay_root/tmp/relay-linux/mold-relay"
