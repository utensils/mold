import { describe, expect, it } from "vitest";
import {
  bestGpuBackend,
  gpuSnapshotsFromStatus,
  gpuSnapshotsFromWorkers,
  summarizeStatusGpuMemory,
} from "./gpuStatus";
import type { ServerStatus } from "./types";

function status(overrides: Partial<ServerStatus> = {}): ServerStatus {
  return {
    version: "0.20.2",
    models_loaded: [],
    uptime_secs: 1,
    ...overrides,
  };
}

describe("multi-GPU status adapters", () => {
  it("preserves every worker instead of collapsing to legacy gpu_info", () => {
    const value = status({
      gpu_info: {
        name: "NVIDIA RTX 3090",
        backend: "cuda",
        vram_total_mb: 24_000,
        vram_used_mb: 8_000,
      },
      gpus: [
        {
          ordinal: 0,
          name: "NVIDIA RTX 3090",
          vram_total_bytes: 24_000_000_000,
          vram_used_bytes: 8_000_000_000,
          state: "generating",
        },
        {
          ordinal: 3,
          name: "NVIDIA B200",
          vram_total_bytes: 80_000_000_000,
          vram_used_bytes: 20_000_000_000,
          state: "idle",
        },
      ],
    });

    expect(gpuSnapshotsFromStatus(value)).toEqual([
      expect.objectContaining({
        ordinal: 0,
        name: "NVIDIA RTX 3090",
        backend: "cuda",
        vram_total: 24_000_000_000,
        vram_used: 8_000_000_000,
      }),
      expect.objectContaining({
        ordinal: 3,
        name: "NVIDIA B200",
        backend: "cuda",
        vram_total: 80_000_000_000,
        vram_used: 20_000_000_000,
      }),
    ]);
    // The worker rows are raw bytes, so only the MiB round-trip moves:
    // 104e9 bytes is 99_182 MiB, not 104_000.
    expect(summarizeStatusGpuMemory(value)).toEqual({
      usedMb: 28_000_000_000 / 1024 ** 2,
      totalMb: 104_000_000_000 / 1024 ** 2,
    });
  });

  it("falls back to gpu_info for an older single-GPU server", () => {
    const value = status({
      gpu_info: {
        name: "Apple M3 Max",
        backend: "metal",
        vram_total_mb: 64_000,
        vram_used_mb: 16_000,
      },
    });

    expect(gpuSnapshotsFromStatus(value)).toEqual([
      expect.objectContaining({
        ordinal: 0,
        backend: "metal",
        vram_total: 67_108_864_000,
        vram_used: 16_777_216_000,
      }),
    ]);
    expect(summarizeStatusGpuMemory(value)).toEqual({
      usedMb: 16_000,
      totalMb: 64_000,
    });
  });

  it("selects CUDA capability even when a Metal device is listed first", () => {
    expect(bestGpuBackend([{ backend: "metal" }, { backend: "cuda" }])).toBe("cuda");
    expect(bestGpuBackend([{ backend: "metal" }])).toBe("metal");
    expect(bestGpuBackend([])).toBeNull();
  });
});

describe("the legacy MB fields are MEBIbytes", () => {
  /*
   * `/api/status.gpu_info.vram_total_mb` is filled by the server as
   * `total_bytes / (1024 * 1024)` (`mold-server/src/device_registry.rs`), so
   * it is MiB — which is also what nvidia-smi prints and what every Rust-side
   * fixture in the repo carries (24564 for a 4090, 49152 for two, 24576 for a
   * round 24 GiB). Reading it as decimal MB under-reported every legacy-status
   * GPU by 4.86%: a 24 GiB card came back as 22.9 GB of bytes.
   */
  it("converts a 4090's reported MiB back to the bytes it came from", () => {
    const [gpu] = gpuSnapshotsFromWorkers(
      { name: "NVIDIA GeForce RTX 4090", vram_total_mb: 24_576, vram_used_mb: 8_192 },
      null,
    );
    expect(gpu!.vram_total).toBe(24 * 1024 ** 3);
    expect(gpu!.vram_used).toBe(8 * 1024 ** 3);
  });

  it("round-trips bytes back to the same MiB the server sent", () => {
    const status = {
      gpu_info: { name: "NVIDIA GeForce RTX 4090", vram_total_mb: 24_564, vram_used_mb: 8_192 },
    } as never;
    expect(summarizeStatusGpuMemory(status)).toEqual({ usedMb: 8_192, totalMb: 24_564 });
  });
});
