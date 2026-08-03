import { describe, expect, it } from "vitest";
import {
  CAMERA_MOTION_PRESETS,
  cameraMotionFromLoraPath,
  cameraMotionLoraLabel,
  cameraMotionLoraPath,
  cameraMotionLoraSlotAvailable,
  cameraMotionMode,
  isCameraMotionPreset,
  normalizeCameraMotionLoraState,
  parseCameraControlAvailability,
  syncCameraMotionLora,
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

  it("maps camera choices onto their visible LoRA rows", () => {
    expect(cameraMotionLoraPath("dolly-in")).toBe("camera-control:dolly-in");
    expect(cameraMotionLoraPath(" /models/pan.safetensors ")).toBe(
      "/models/pan.safetensors",
    );
    expect(cameraMotionFromLoraPath("camera-control:dolly-in")).toBe(
      "dolly-in",
    );
    expect(cameraMotionLoraLabel("camera-control:dolly-in")).toBe(
      "Dolly in camera control",
    );
  });

  it("inserts, replaces, and removes the camera row without losing its strength", () => {
    const create = (path: string, scale: number) => ({ path, scale });
    const selected = syncCameraMotionLora(
      [{ path: "style.safetensors", scale: 0.7 }],
      null,
      "dolly-in",
      create,
    );
    expect(selected).toEqual([
      { path: "style.safetensors", scale: 0.7 },
      { path: "camera-control:dolly-in", scale: 1 },
    ]);

    selected[1]!.scale = 0.45;
    const switched = syncCameraMotionLora(
      selected,
      "dolly-in",
      "dolly-right",
      create,
    );
    expect(switched).toEqual([
      { path: "style.safetensors", scale: 0.7 },
      { path: "camera-control:dolly-right", scale: 0.45 },
    ]);
    expect(syncCameraMotionLora(switched, "dolly-right", null, create)).toEqual(
      [{ path: "style.safetensors", scale: 0.7 }],
    );
  });

  it("keeps the previous strength while a custom path is being entered", () => {
    const create = (path: string, scale: number) => ({ path, scale });
    const selected = [{ path: "camera-control:dolly-in", scale: 0.45 }];

    const editing = syncCameraMotionLora(selected, "dolly-in", "", create);
    expect(editing).toEqual(selected);
    expect(
      syncCameraMotionLora(editing, "", "/models/custom.safetensors", create),
    ).toEqual([{ path: "/models/custom.safetensors", scale: 0.45 }]);
  });

  it("does not displace an existing LoRA when all stack slots are occupied", () => {
    const full = ["one", "two", "three", "four"].map((path) => ({
      path,
      scale: 1,
    }));
    expect(cameraMotionLoraSlotAvailable(full, null, 4)).toBe(false);
    expect(
      syncCameraMotionLora(
        full,
        null,
        "dolly-in",
        (path, scale) => ({ path, scale }),
        4,
      ),
    ).toEqual(full);
  });

  it("normalizes restored camera aliases and legacy picker-only state", () => {
    const create = (path: string, scale: number) => ({ path, scale });
    expect(
      normalizeCameraMotionLoraState(
        [{ path: "camera-control:dolly-in", scale: 0.45 }],
        null,
        create,
        4,
      ),
    ).toEqual({
      loras: [{ path: "camera-control:dolly-in", scale: 0.45 }],
      cameraControl: "dolly-in",
    });
    expect(normalizeCameraMotionLoraState([], "jib-up", create, 4)).toEqual({
      loras: [{ path: "camera-control:jib-up", scale: 1 }],
      cameraControl: "jib-up",
    });
  });

  it("clears an unrepresentable legacy camera selection from a full stack", () => {
    const full = ["one", "two", "three", "four"].map((path) => ({
      path,
      scale: 1,
    }));
    expect(
      normalizeCameraMotionLoraState(
        full,
        "dolly-in",
        (path, scale) => ({ path, scale }),
        4,
      ),
    ).toEqual({ loras: full, cameraControl: null });
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
