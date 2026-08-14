import { describe, expect, it } from "vitest";
import type { ResourceSnapshot, ServerStatus } from "../../types";
import { deriveHostCardGpu, deriveTelemetry } from "./machineTelemetry";

function statusWithGpus(gpus: NonNullable<ServerStatus["gpus"]>): ServerStatus {
  return {
    version: "test",
    models_loaded: [],
    busy: false,
    uptime_secs: 1,
    gpus,
  };
}

function resourcesWithGpu(
  backend: "metal" | "cuda",
  name: string,
): ResourceSnapshot {
  return {
    hostname: "test",
    timestamp: 1,
    gpus: [
      {
        ordinal: 0,
        name,
        backend,
        vram_total: 51_500_000_000,
        vram_used: 46_900_000_000,
        vram_used_by_mold: null,
        vram_used_by_other: null,
        gpu_utilization: null,
      },
    ],
    system_ram: {
      total: 51_500_000_000,
      used: 46_900_000_000,
      used_by_mold: 40_000_000_000,
      used_by_other: 6_900_000_000,
    },
    cpu: { cores: 16, usage_percent: 44 },
  };
}

describe("deriveTelemetry unified memory", () => {
  it("marks a Metal host unified so the panel can drop the duplicate RAM row", () => {
    const view = deriveTelemetry(
      null,
      resourcesWithGpu("metal", "Apple Metal GPU"),
    );
    expect(view.unifiedMemory).toBe(true);
  });

  it("keeps CUDA hosts and unknown hosts non-unified", () => {
    expect(
      deriveTelemetry(null, resourcesWithGpu("cuda", "NVIDIA L40S"))
        .unifiedMemory,
    ).toBe(false);
    expect(deriveTelemetry(null, null).unifiedMemory).toBe(false);
  });
});

describe("deriveHostCardGpu", () => {
  it("summarizes every homogeneous GPU and aggregates VRAM", () => {
    const summary = deriveHostCardGpu(
      statusWithGpus([
        {
          ordinal: 0,
          name: "NVIDIA GeForce RTX 3090",
          vram_total_bytes: 24_576_000_000,
          vram_used_bytes: 1_000_000_000,
          state: "idle",
        },
        {
          ordinal: 1,
          name: "NVIDIA GeForce RTX 3090",
          vram_total_bytes: 24_576_000_000,
          vram_used_bytes: 4_000_000_000,
          state: "generating",
        },
      ]),
    );

    expect(summary).toEqual({
      label: "2× NVIDIA GeForce RTX 3090",
      used: 5_000_000_000,
      total: 49_152_000_000,
    });
  });

  it("names heterogeneous GPUs instead of silently displaying ordinal zero", () => {
    const summary = deriveHostCardGpu(
      statusWithGpus([
        {
          ordinal: 0,
          name: "NVIDIA B200",
          vram_total_bytes: 192_000_000_000,
          vram_used_bytes: 20,
          state: "idle",
        },
        {
          ordinal: 1,
          name: "NVIDIA H100",
          vram_total_bytes: 80_000_000_000,
          vram_used_bytes: 10,
          state: "idle",
        },
      ]),
    );

    expect(summary?.label).toBe("NVIDIA B200 + NVIDIA H100");
    expect(summary?.total).toBe(272_000_000_000);
  });

  it("retains the legacy gpu_info fallback for older servers", () => {
    expect(
      deriveHostCardGpu({
        version: "old",
        models_loaded: [],
        busy: false,
        uptime_secs: 1,
        gpu_info: {
          name: "Apple M3 Ultra",
          vram_total_mb: 196_608,
          vram_used_mb: 32_768,
        },
      }),
    ).toEqual({
      label: "Apple M3 Ultra",
      used: 32_768_000_000,
      total: 196_608_000_000,
    });
  });
});
