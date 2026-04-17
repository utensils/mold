# ── mold inference server ──
# Multi-stage Docker build for RunPod (and any NVIDIA GPU host).
#
# Build:
#   docker build -t mold-server .
#   docker build --build-arg CUDA_COMPUTE_CAP=90 -t mold-server-h100 .
#
# Run (local):
#   docker run --gpus all -p 7680:7680 mold-server
#
# CUDA_COMPUTE_CAP targets:
#   80 = Ampere (A100)          86 = Ampere (RTX 3090, A40)
#   89 = Ada Lovelace (RTX 4090, L40S)   90 = Hopper (H100)
#   120 = Blackwell (RTX 5090, B200)

# ── Stage 1a: Build web gallery SPA ─────────────────────────────────
# The gallery is served by `mold serve` as a SPA fallback. Building it
# in a dedicated stage keeps the Rust builder free of Node/bun tooling
# and avoids re-running `bun install` on every cargo cache invalidation.
FROM oven/bun:1.1-alpine AS web-builder
WORKDIR /web
COPY web/package.json web/bun.lock web/tsconfig.json web/tsconfig.app.json web/tsconfig.node.json web/vite.config.ts web/index.html ./
RUN bun install --frozen-lockfile
COPY web/src ./src
COPY web/public ./public
RUN bun run build

# ── Stage 1b: Build mold binary ─────────────────────────────────────
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ARG CUDA_COMPUTE_CAP=89
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for building.
# apt-get is retried with backoff because archive.ubuntu.com has flaked
# on GHA builders before (run 24552477974 failed fetching tini); a
# handful of retries is enough to ride out transient DNS / mirror
# outages without masking real failures.
RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        apt-get update && apt-get install -y --no-install-recommends \
            build-essential \
            pkg-config \
            libssl-dev \
            libwebp-dev \
            nasm \
            git \
            ca-certificates \
            curl \
        && break \
        || (echo "apt attempt $attempt failed, retrying in $((attempt * 5))s" && sleep $((attempt * 5))); \
    done; \
    rm -rf /var/lib/apt/lists/*

# Install Rust (stable)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy manifests first for layer caching of dependency builds
COPY Cargo.toml Cargo.lock ./
COPY crates/mold-core/Cargo.toml crates/mold-core/Cargo.toml
COPY crates/mold-inference/Cargo.toml crates/mold-inference/Cargo.toml
COPY crates/mold-server/Cargo.toml crates/mold-server/Cargo.toml
COPY crates/mold-cli/Cargo.toml crates/mold-cli/Cargo.toml
COPY crates/mold-discord/Cargo.toml crates/mold-discord/Cargo.toml
COPY crates/mold-tui/Cargo.toml crates/mold-tui/Cargo.toml

# Create stub source files so cargo can resolve and build dependencies
RUN mkdir -p crates/mold-core/src \
             crates/mold-inference/src \
             crates/mold-server/src \
             crates/mold-cli/src \
             crates/mold-discord/src \
             crates/mold-tui/src \
    && echo "// stub" > crates/mold-core/src/lib.rs \
    && echo "// stub" > crates/mold-inference/src/lib.rs \
    && echo "// stub" > crates/mold-server/src/lib.rs \
    && echo 'fn main() { println!("stub"); }' > crates/mold-cli/src/main.rs \
    && echo "// stub" > crates/mold-discord/src/lib.rs \
    && echo "// stub" > crates/mold-tui/src/lib.rs

# Build dependencies only (this layer is cached until Cargo.toml/lock changes)
RUN cargo build --release -p mold-ai --features cuda,expand,discord,tui,webp,mp4,metrics \
    || true

# Now copy the real source code
COPY crates/ crates/

# Touch source files to invalidate the stub builds but keep dep artifacts
RUN find crates/ -name "*.rs" -exec touch {} +

# Build the real binary
RUN cargo build --release -p mold-ai --features cuda,expand,discord,tui,webp,mp4,metrics

# Verify no unexpected missing libraries (libcuda.so.1 is expected to be
# absent — it's the NVIDIA driver, injected at runtime by the container toolkit)
RUN ! ldd /build/target/release/mold | grep "not found" | grep -v "libcuda.so"

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# cuBLAS and cuRAND are already in the runtime image.
# libcuda.so.1 (the NVIDIA driver) is injected at runtime by the NVIDIA
# Container Toolkit on GPU hosts like RunPod — it is never in the image.
# Same apt retry wrapper as the builder stage — the tini fetch in particular
# has timed out on GHA before.
RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        apt-get update && apt-get install -y --no-install-recommends \
            ca-certificates \
            libwebp7 \
            tini \
        && break \
        || (echo "apt attempt $attempt failed, retrying in $((attempt * 5))s" && sleep $((attempt * 5))); \
    done; \
    rm -rf /var/lib/apt/lists/*

# Copy the compiled binary
COPY --from=builder /build/target/release/mold /usr/local/bin/mold

# Copy the built web gallery SPA. `mold serve` picks this up via
# MOLD_WEB_DIR and serves it as the SPA fallback alongside /api/*.
COPY --from=web-builder /web/dist /opt/mold/web

# Copy the entrypoint script
COPY docker/start.sh /start.sh

# ── Environment variable defaults ──────────────────────────────────
# Core
ENV MOLD_LOG=info \
    MOLD_PORT=7680 \
    MOLD_DEFAULT_MODEL=flux2-klein:q8 \
    MOLD_EMBED_METADATA=1 \
    MOLD_WEB_DIR=/opt/mold/web
# Server
# MOLD_HOST=               (clients: remote server URL)
# MOLD_HOME=               (override base mold directory)
# MOLD_MODELS_DIR=         (override model storage)
# MOLD_OUTPUT_DIR=         (override output directory, empty to disable)
# MOLD_API_KEY=            (API key for server auth)
# MOLD_RATE_LIMIT=         (per-IP rate limit, e.g. "10/min")
# MOLD_CORS_ORIGIN=        (restrict CORS origin)
# MOLD_GALLERY_ALLOW_DELETE= (1 to enable DELETE /api/gallery/image/:name)
# MOLD_WEB_DIR=/opt/mold/web (override bundled web SPA location)
# Inference
# MOLD_EAGER=              (1 to keep all model components loaded)
# MOLD_OFFLOAD=            (1 to force CPU↔GPU block streaming)
# MOLD_T5_VARIANT=auto     (T5 encoder: auto, fp16, q8, q6, q5, q4, q3)
# MOLD_QWEN3_VARIANT=auto  (Qwen3 encoder: auto, bf16, q8, q6, iq4, q3)
# MOLD_SCHEDULER=          (ddim, euler-ancestral, uni-pc)
# MOLD_PREVIEW=            (1 to display images inline)
# Video
# MOLD_LTX_DEBUG=          (1 for per-step LTX Video diagnostics)
# MOLD_LTX2_DEBUG=         (1 for per-step LTX-2 / LTX-2.3 diagnostics)
# NOTE: LTX-2 / LTX-2.3 is CUDA-only — no Metal support, CPU is correctness-only.
# NOTE: LTX-2 Gemma text-encoder assets are HuggingFace-gated; set HF_TOKEN.
# Expansion
# MOLD_EXPAND=             (1 to enable LLM prompt expansion)
# MOLD_EXPAND_BACKEND=local
# MOLD_EXPAND_MODEL=qwen3-expand:q8
# MOLD_EXPAND_TEMPERATURE=0.7
# Discord
# MOLD_DISCORD_TOKEN=      (Discord bot token)
# MOLD_DISCORD_COOLDOWN=10
# MOLD_DISCORD_DAILY_QUOTA=

EXPOSE 7680

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]
CMD ["/start.sh"]
