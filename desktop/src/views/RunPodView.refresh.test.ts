import { describe, expect, it } from "vitest";
import source from "./RunPodView.vue?raw";

describe("RunPodView refresh feedback", () => {
  it("keeps refreshes visually silent", () => {
    expect(source).toContain('aria-label="Refresh RunPod status"');
    expect(source).toContain('@click="runpod.load()"');
    expect(source).not.toContain("Refreshing…");
    expect(source).not.toContain("manualRefreshing");
  });

  it("waits with the house shimmer, not a shouted mono placeholder", () => {
    expect(source).toContain('data-test="runpod-loading"');
    expect(source).toContain("ms-shimmer");
    expect(source).not.toContain("LOADING RUNPOD");
  });

  it("wears the RunPod mark rather than a generic cloud", () => {
    expect(source).toContain('name="runpod"');
    expect(source).not.toContain('name="cloud"');
  });
});
