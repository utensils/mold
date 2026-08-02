import { describe, expect, it } from "vitest";
import {
  CAMERA_MOTION_PRESETS,
  cameraMotionMode,
  isCameraMotionPreset,
  parseCameraControlAvailability,
} from "./cameraMotion";

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
    expect(isCameraMotionPreset("jib-up")).toBe(true);
    expect(isCameraMotionPreset("/models/camera.safetensors")).toBe(false);
    expect(cameraMotionMode(null)).toBe("");
    expect(cameraMotionMode("")).toBe("custom");
    expect(cameraMotionMode("   ")).toBe("custom");
    expect(cameraMotionMode("jib-up")).toBe("jib-up");
    expect(cameraMotionMode("/models/camera.safetensors")).toBe("custom");
  });

  it("reads a legacy bare-array response from an older host", () => {
    const controls = [{ id: "jib-up", label: "Jib up" }];
    expect(parseCameraControlAvailability(controls)).toEqual({
      controls,
      supported: true,
      unsupportedReason: null,
    });
  });

  it("treats an older host's empty array as unsupported with no reason", () => {
    // The bare array cannot say why, which is exactly why `?detail=1` exists.
    expect(parseCameraControlAvailability([])).toEqual({
      controls: [],
      supported: false,
      unsupportedReason: null,
    });
  });

  it("surfaces the host's own reason from the detail envelope", () => {
    expect(
      parseCameraControlAvailability({
        controls: [],
        supported: false,
        unsupported_reason:
          "camera-control presets are currently published for LTX-2 19B only",
      }),
    ).toEqual({
      controls: [],
      supported: false,
      unsupportedReason:
        "camera-control presets are currently published for LTX-2 19B only",
    });
  });

  it("reads a supported detail envelope", () => {
    const controls = [{ id: "dolly-in", label: "Dolly in" }];
    expect(
      parseCameraControlAvailability({ controls, supported: true }),
    ).toEqual({ controls, supported: true, unsupportedReason: null });
  });

  it("never throws on a malformed body", () => {
    for (const body of [null, undefined, 42, "nope", {}]) {
      expect(parseCameraControlAvailability(body)).toEqual({
        controls: [],
        supported: false,
        unsupportedReason: null,
      });
    }
  });
});
