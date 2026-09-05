import { describe, expect, it } from "vitest";

import source from "./RunPodView.vue?raw";

describe("RunPod confirm-before-spend (G9)", () => {
  it("routes the Launch button through a confirm instead of provisioning directly", () => {
    // launch() only opens the dialog; the network call lives in confirmLaunch().
    expect(source).toMatch(/function launch\(\)\s*\{\s*confirmProvision\.value = true;\s*\}/);
    expect(source).toContain("async function confirmLaunch()");
    expect(source).toMatch(/confirmLaunch[\s\S]*await runpod\.create/);
    // The Launch button still triggers launch(), which now only opens the confirm.
    expect(source).toContain('@click="launch"');
  });

  it("gates resuming a stopped pod behind a confirm too", () => {
    expect(source).toContain('@click="requestStart(pod)"');
    expect(source).toContain("async function confirmStart()");
    expect(source).toMatch(/confirmStart[\s\S]*act\("start", pod\)/);
  });

  it("wires the confirm dialogs with blunt spend copy", () => {
    expect(source).toContain("<ConfirmDialog");
    expect(source).toContain('@confirm="confirmLaunch"');
    expect(source).toContain('@confirm="confirmStart"');
    expect(source).toContain("billing begins now");
    expect(source).toContain("Billing resumes immediately.");
  });

  it("shows a live running-cost meter on billing pods", () => {
    expect(source).toContain("<PodCostMeter");
    expect(source).toContain(':uptime-seconds="pod.uptimeSeconds"');
  });
});
