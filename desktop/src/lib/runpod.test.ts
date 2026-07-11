import { describe, expect, it } from "vitest";
import {
  friendlyRunPodError,
  isNetworkVolumeDatacenter,
  podGpuName,
  podProxyUrl,
  rankRunPodGpus,
  runPodRegionLabel,
  type RunPodGpu,
  type RunPodPod,
} from "./runpod";

const gpu = (displayName: string, stockStatus: string | null, memoryInGb: number): RunPodGpu => ({
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
      gpu("RTX 3090", "None", 24),
    ]);

    expect(ranked.map((entry) => entry.displayName)).toEqual([
      "RTX 4090",
      "RTX 5090",
      "A100 80GB",
      "RTX 3090",
    ]);
  });

  it("builds the RunPod HTTP proxy URL used by mold serve", () => {
    expect(podProxyUrl("abc123")).toBe("https://abc123-7680.proxy.runpod.net");
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
