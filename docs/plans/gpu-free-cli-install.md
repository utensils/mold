# GPU-free Linux CLI distribution

## Problem and acceptance criteria

Linux installation currently requires a usable NVIDIA inventory and offers only
CUDA-linked archives. The AUR binary package also selects SM89 and requires the
wrong major CUDA runtime on current Arch. A remote client must install and run
`mold --version`, help, completions, model/host commands and HTTP generation
without NVIDIA hardware, drivers, toolkit or shared libraries.

## Implementation plan

1. Publish `mold-x86_64-unknown-linux-gnu-cpu.tar.gz` on both stable tags and
   rolling `latest`. Build on Ubuntu 22.04 without CUDA/Metal features, retaining
   `preview,discord,expand,webp,mp4,metrics,mdns,pulid` and the embedded web UI.
   Explicitly exclude cuda/cudnn/flash-attn/h3-cuda/metal and mesh features. Include the binary and all
   three shell completions. Verify ELF dependencies, run version/help/completions,
   and exercise an HTTP request against a local mock host in a GPU-free runtime.
   Include the archive in publication dependencies, uploads, SHA256SUMS and
   release descriptions. PR CI must build and smoke-test this shipping recipe.
2. Add `MOLD_BACKEND=auto|cpu|cuda` to the POSIX installer. Auto preserves CUDA
   selection when an inspectable compatible GPU exists, but chooses CPU when
   NVIDIA inventory is unavailable or no GPU is visible. Explicit CPU bypasses
   probing. Explicit CUDA and MOLD_CUDA_ARCH remain strict; contradictory CPU and
   CUDA-architecture settings fail. Unsupported/malformed/mixed visible CUDA
   fleets retain their existing actionable refusals and can opt into CPU.
   CPU downloads never fall back to CUDA or another release. Missing older
   CPU assets explain that a newer release or source build is needed. Keep
   checksum verification, replacement and CUDA compatibility guards intact.
3. Make `mold update` preserve the compiled backend: non-CUDA Linux builds
   select CPU without probing NVIDIA, CUDA builds retain existing selection.
   Keep package-manager ownership and missing-asset behavior intact.
4. Move `mold-ai-bin` to the GPU-free CPU archive, drop CUDA/NVIDIA dependencies,
   and update its checksum publisher and no-GPU Arch smoke harness. This is an
   intentional package migration: existing local GPU users must use the
   `mold-ai`/`mold-ai-git` source packages or architecture-specific release
   archives; those CUDA source recipes stay unchanged. Document this clearly.
   Keep package conflicts, versioned provides, and packaged completions.
   Publication waits for the new asset and rewrites real release checksums;
   never claim the old stable release contains an asset that is not published.
5. Update README, installation guide, AUR README, release notes/templates,
   release-maintainer rules and a changelog fragment. Explain remote usage with
   MOLD_HOST, automatic selection, forced CPU/CUDA, CPU local-generation limits,
   current Linux x86_64 scope, migration, and rollout (nightly after main build;
   stable/AUR after the next tag). Refresh relevant knowledge-graph entries
   without claiming a full graph regeneration.

## Validation and review gates

- Hermetic installer tests: absent/failing/empty NVIDIA probe, empty/-1
  visibility, CPU override on GPU host, explicit CUDA failures and conflicts,
  existing GPU/MIG/fleet cases, stable/nightly/pinned URLs, checksum failures,
  CPU 404 without CUDA fallback, successful extraction and executable install.
- Updater unit tests for CPU selection and exact-asset/no-fallback behavior.
- GPU-free build and real binary smoke test, including remote HTTP request;
  ELF audit rejects CUDA/NVIDIA dependencies. Arch package installs and executes
  without pulling CUDA, with completions present.
- Release contracts verify CPU build feature set, publication/checksum wiring,
  AUR source/checksum alignment, and PR-visible validation. Run relevant local
  release/doc/format checks and affected Rust tests.
- Independent subagent plan review first; resolve findings before coding.
- Claude reviews final implementation before PR creation; fix valid findings.
- Open conventional-commit PR, inspect bot review findings, await terminal CI
  on its exact head, merge as authorized and synchronize this worktree.

## Boundaries

No GPU runtime loader redesign, no new ARM artifacts, no claim of CPU inference
performance, no version bump or release tag created manually. This fixes the
client distribution path while retaining explicitly selected CUDA distributions.

## Plan review resolution

Independent GPT-5.6 Sol review completed before implementation. Accepted:

- Auto plus a nonempty architecture override means explicit CUDA intent. Only
  auto without an override falls back to CPU. Invalid backend values fail;
  non-auto backend selection on Darwin is rejected (its archive is Metal).
- CPU updater selection happens before any NVIDIA probing and ignores ambient
  CUDA overrides. Exact CPU assets have no legacy substitute.
- One reusable workflow builds, packages and verifies the same release recipe
  in PR CI, main and tag runs. Apple-first stable publication stays independent.
- Arch smoke asserts absence of CUDA packages/libraries both before and after
  installation; HTTP smoke asserts the request reached the mock and its result
  was rendered, preventing a local fallback from passing.
- GPU hosts without CUDA user-space libraries should explicitly choose CPU;
  auto does not attempt to diagnose or install driver/runtime libraries.
- Old tags remain immutable; missing CPU archives fail with newer-release or
  source-build guidance. AUR checksums are fetched after native publication.

The review reported no graph, but direct verification found the tracked `.ua/`
graph in this worktree. Its global baseline is stale; scoped source-verified
updates follow the repository's existing scopedUpdates convention.
