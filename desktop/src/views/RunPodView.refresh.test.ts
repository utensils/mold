import { describe, expect, it } from "vitest";
import source from "./RunPodView.vue?raw";

describe("RunPodView refresh feedback", () => {
  it("keeps refreshes visually silent", () => {
    expect(source).toContain('aria-label="Refresh RunPod status"');
    expect(source).toContain('@click="runpod.load()"');
    expect(source).not.toContain("Refreshing…");
    expect(source).not.toContain("manualRefreshing");
  });
});
