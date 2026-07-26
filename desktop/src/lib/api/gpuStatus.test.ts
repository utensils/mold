import { describe, expect, it } from "vitest";
import {
  gpuSnapshotsFromStatus,
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
    expect(summarizeStatusGpuMemory(value)).toEqual({
      usedMb: 28_000,
      totalMb: 104_000,
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
        vram_total: 64_000_000_000,
        vram_used: 16_000_000_000,
      }),
    ]);
    expect(summarizeStatusGpuMemory(value)).toEqual({
      usedMb: 16_000,
      totalMb: 64_000,
    });
  });
});
