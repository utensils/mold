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

  it("does not offer unsupported stop or start actions for volume-backed pods", () => {
    const button = (label: string) => {
      const labelAt = source.indexOf(`\n                ${label}\n`);
      expect(labelAt).toBeGreaterThan(0);
      return source.slice(source.lastIndexOf("<button", labelAt), labelAt);
    };
    expect(button("Use in mold")).toContain("v-if=\"pod.desiredStatus === 'RUNNING'\"");
    expect(button("Use in mold")).not.toContain("!pod.networkVolume");
    expect(button("Stop")).toContain(
      "v-if=\"!pod.networkVolume && pod.desiredStatus === 'RUNNING'\"",
    );
    expect(button("Start")).toContain('v-else-if="!pod.networkVolume"');
    expect(source).toContain(
      "Delete the instance to stop compute; /workspace remains on the volume.",
    );
  });
});
