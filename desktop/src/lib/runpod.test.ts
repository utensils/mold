import { describe, expect, it } from "vitest";
import { podGpuName, podProxyUrl, rankRunPodGpus, type RunPodGpu, type RunPodPod } from "./runpod";

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
      machine: { gpuDisplayName: null, location: "US" },
      gpu: { id: "NVIDIA RTX PRO 6000", displayName: "RTX PRO 6000 Blackwell", count: 1 },
    } as RunPodPod;
    expect(podGpuName(pod)).toBe("RTX PRO 6000 Blackwell");
  });
});
