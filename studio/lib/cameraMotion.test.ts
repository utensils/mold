import { describe, expect, it } from "vitest";
import { CAMERA_MOTION_PRESETS, cameraMotionMode } from "./cameraMotion";

describe("camera motion presets", () => {
  it("keeps the shared LTX-2 preset order and labels", () => {
    expect(CAMERA_MOTION_PRESETS).toEqual([
      { id: "dolly-in", label: "Dolly in" },
      { id: "dolly-left", label: "Dolly left" },
      { id: "dolly-out", label: "Dolly out" },
      { id: "dolly-right", label: "Dolly right" },
      { id: "jib-down", label: "Jib down" },
      { id: "jib-up", label: "Jib up" },
      { id: "static", label: "Static" },
    ]);
  });

  it("classifies empty, preset, and custom values", () => {
    expect(cameraMotionMode(null)).toBe("");
    expect(cameraMotionMode("")).toBe("");
    expect(cameraMotionMode("jib-up")).toBe("jib-up");
    expect(cameraMotionMode("/models/camera.safetensors")).toBe("custom");
  });
});
