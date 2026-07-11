import { describe, expect, it } from "vitest";

import source from "./RunPodView.vue?raw";

describe("RunPod network volume UX", () => {
  it("supports lifecycle management and remembers the selected volume", () => {
    expect(source).toContain("createNetworkVolume");
    expect(source).toContain("updateNetworkVolume");
    expect(source).toContain("deleteNetworkVolume");
    expect(source).toContain("runpodNetworkVolumeId");
  });

  it("makes volume placement constraints visible", () => {
    expect(source).toContain('form.cloudType = "SECURE"');
    expect(source).toContain("form.datacenterId = volume.dataCenterId");
    expect(source).toContain("mounted at /workspace");
  });
});
