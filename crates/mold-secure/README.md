# Shared secure protocol

This GPU-independent crate supplies the same Noise and identity implementation
to Rust clients, the native Apple C ABI, and browser WebAssembly. The wire
contract is [SECURE-PROTOCOL.md](../../relay/aws/SECURE-PROTOCOL.md).

## Rebuilding browser artifacts

Generate `studio/api/generated/secure/` on **x86_64 Linux**, using the Rust
`x86_64-unknown-linux-gnu` host toolchain. The compiler and generator must match CI:

```bash
rustup toolchain install 1.98.0 --profile minimal --target wasm32-unknown-unknown
cargo +1.98.0 install wasm-bindgen-cli --version 0.2.118 --locked
bash scripts/build-secure-wasm.sh
```

Commit the generated artifacts with their source changes. CI rebuilds them with
`bash scripts/build-secure-wasm.sh --check` and requires an exact byte match.
macOS and other build hosts consume the checked-in artifacts when building web
or Tauri clients. Native Rust and Apple frameworks build directly from the crate
on their supported platforms; this restriction applies only to regenerating the
browser artifact.

Path remapping removes builder-specific checkout, Cargo home and Rust source
locations. A fixed compiler host is also necessary: cross-host builds had
identical code but differed in the 128-bit `TypeId` for Ed25519's boxed error.
CI and independent canonical Linux generation produced exactly the same bytes.
The script rejects other compiler hosts before producing artifacts, preserving
the cryptographic dependency settings and the exact-byte comparison.
