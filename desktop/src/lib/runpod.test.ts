import { describe, expect, it } from "vitest";
import {
  friendlyRunPodError,
  isNetworkVolumeDatacenter,
  isRunPodHostUrl,
  podGpuName,
  podProxyUrl,
  rankRunPodGpus,
  runPodForHostUrl,
  runPodRegionLabel,
  type RunPodGpu,
  type RunPodPod,
  estimatedPodCost,
  liveAccruedCost,
  formatUsd,
  podHardwareSummary,
} from "./runpod";

const gpu = (
  displayName: string,
  stockStatus: string | null,
  memoryInGb: number | null,
): RunPodGpu => ({
  id: displayName,
  displayName,
  memoryInGb,
  stockStatus,
  secureCloud: true,
  communityCloud: true,
  available: stockStatus !== "None",
});

describe("RunPod presentation helpers", () => {
  it("ranks available GPUs by stock, then hourly-fit memory", () => {
    const ranked = rankRunPodGpus([
      gpu("A100 80GB", "Medium", 80),
      gpu("RTX 4090", "High", 24),
      gpu("RTX 5090", "High", 32),
      gpu("Unknown", "High", null),
      gpu("RTX 3090", "None", 24),
    ]);

    expect(ranked.map((entry) => entry.displayName)).toEqual([
      "RTX 4090",
      "RTX 5090",
      "Unknown",
      "A100 80GB",
      "RTX 3090",
    ]);
  });

  it("builds the RunPod HTTP proxy URL used by mold serve", () => {
    expect(podProxyUrl("abc123")).toBe("https://abc123-7680.proxy.runpod.net");
  });

  it("matches a running pod by proxy hostname across scheme and port differences", () => {
    const running = { id: "abc123", desiredStatus: "RUNNING" } as RunPodPod;
    const stopped = { id: "stopped", desiredStatus: "EXITED" } as RunPodPod;
    expect(runPodForHostUrl([stopped, running], "http://abc123-7680.proxy.runpod.net:80/api")).toBe(
      running,
    );
    expect(runPodForHostUrl([stopped], "https://stopped-7680.proxy.runpod.net")).toBeNull();
  });

  it("recognizes a pod proxy even after that pod is stopped", () => {
    expect(isRunPodHostUrl("abc123", "http://abc123-7680.proxy.runpod.net:80/api")).toBe(true);
    expect(isRunPodHostUrl("abc123", "https://another-7680.proxy.runpod.net")).toBe(false);
  });

  it("uses the top-level REST GPU when machine metadata omits it", () => {
    const pod = {
      machine: { gpuDisplayName: null, gpuTypeId: null, dataCenterId: null, location: "US" },
      gpu: { id: "NVIDIA RTX PRO 6000", displayName: "RTX PRO 6000 Blackwell", count: 1 },
    } as RunPodPod;
    expect(podGpuName(pod)).toBe("RTX PRO 6000 Blackwell");
  });

  it("uses current REST machine.gpuTypeId when display-name fields are absent", () => {
    const pod = {
      machine: {
        gpuDisplayName: null,
        gpuTypeId: "NVIDIA GeForce RTX 4090",
        dataCenterId: "EU-RO-1",
        location: "RO",
      },
      gpu: null,
    } as RunPodPod;
    expect(podGpuName(pod)).toBe("NVIDIA GeForce RTX 4090");
  });

  it("turns datacenter IDs into recognizable regions", () => {
    expect(runPodRegionLabel("CA-MTL-3", "Canada")).toBe("Montreal, Canada · CA-MTL-3");
    expect(runPodRegionLabel("EU-RO-1", "Europe")).toBe("Romania, Europe · EU-RO-1");
    expect(runPodRegionLabel("US-TX-3", "United States")).toBe("Texas, United States · US-TX-3");
  });

  it("limits network-volume creation to production-supported datacenters", () => {
    expect(isNetworkVolumeDatacenter("CA-MTL-1")).toBe(false);
    expect(isNetworkVolumeDatacenter("CA-MTL-3")).toBe(true);
    expect(isNetworkVolumeDatacenter("EU-RO-1")).toBe(true);
  });

  it("turns network-volume API payloads into an actionable message", () => {
    expect(
      friendlyRunPodError(
        'RunPod /networkvolumes 500 Internal Server Error: {"error":"create network volume: Data center \\"CA-MTL-1\\" not found or does not support network volumes. Available data centers: CA-MTL-3, EU-RO-1.","status":500}',
      ),
    ).toBe("CA-MTL-1 does not support network volumes. Choose a supported region and try again.");
  });
});

describe("estimatedPodCost", () => {
  it("multiplies uptime by the hourly rate", () => {
    expect(estimatedPodCost(0.44, 3600)).toBeCloseTo(0.44);
    expect(estimatedPodCost(1.2, 5400)).toBeCloseTo(1.8);
  });

  it("floors at zero for missing rates or fresh pods", () => {
    expect(estimatedPodCost(0, 3600)).toBe(0);
    expect(estimatedPodCost(0.44, 0)).toBe(0);
  });
});

describe("podHardwareSummary", () => {
  it("summarizes the useful hardware facts and skips unknowns", () => {
    const pod = {
      gpuCount: 2,
      vcpuCount: 16,
      memoryInGb: 62,
      volumeInGb: 40,
      machine: { dataCenterId: "EU-RO-1", gpuDisplayName: null, gpuTypeId: null, location: null },
    } as never;
    expect(podHardwareSummary(pod)).toBe("2× GPU · 16 vCPU · 62 GB RAM · 40 GB disk · EU-RO-1");
    const sparse = {
      gpuCount: 1,
      vcpuCount: 0,
      memoryInGb: 0,
      volumeInGb: 0,
      machine: null,
    } as never;
    expect(podHardwareSummary(sparse)).toBe("");
  });
});

describe("liveAccruedCost", () => {
  it("adds wall-clock time since the uptime snapshot to the accrued cost", () => {
    // 30 min of reported uptime, read 30 min ago → 1h billed at $1.20/hr = $1.20.
    const snapshotAt = 1_000_000;
    const now = snapshotAt + 30 * 60 * 1000;
    expect(liveAccruedCost(1.2, 1800, snapshotAt, now)).toBeCloseTo(1.2);
  });

  it("ignores a clock that runs backwards", () => {
    const snapshotAt = 5_000_000;
    expect(liveAccruedCost(1.2, 3600, snapshotAt, snapshotAt - 10_000)).toBeCloseTo(1.2);
  });

  it("is zero when the pod is not billing", () => {
    expect(liveAccruedCost(0, 3600, 0, 3_600_000)).toBe(0);
  });
});

describe("formatUsd", () => {
  it("renders a USD amount", () => {
    expect(formatUsd(0.69)).toBe("$0.69");
    expect(formatUsd(12)).toBe("$12.00");
  });
});
