import { describe, expect, it } from "vitest";
import source from "./RunPodView.vue?raw";

describe("RunPodView logs", () => {
  it("opens the supported RunPod console instead of calling a nonexistent REST endpoint", () => {
    expect(source).toContain("View logs ↗");
    expect(source).toContain('@click="openConsole"');
    expect(source).not.toContain("runpodLogs");
    expect(source).not.toContain("showLogs");
  });
});
