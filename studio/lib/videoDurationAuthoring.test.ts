import { expect, it } from "vitest";
import {
  durationVideoCopies,
  initialVideoDuration,
  buildDurationVideoRequest,
} from "./videoDurationAuthoring";
it("initializes from the current short recipe without a global thirty-second default", () => {
  expect(initialVideoDuration(107, 24).durationMs).toBe("4458");
  expect(
    initialVideoDuration(Number.MAX_SAFE_INTEGER + 1, 24).durationMs,
  ).toBeNull();
});
it("keeps literal generation controls and optional scene prompts with exact durations", () => {
  const generation = {
    model: "installed",
    prompt: "",
    negative_prompt: String.raw`literal\new`,
    source_image: "pixels",
    frames: 107,
    fps: 24,
    batch_size: 1,
  };
  const draft = {
    durationMs: "30000",
    scenesEnabled: true,
    scenes: [
      { id: "one", prompt: String.raw`Scene\one`, durationMs: "10000" },
      { id: "two", prompt: "Scene two", durationMs: "20000" },
    ],
  };
  const request = buildDurationVideoRequest(draft, generation);
  expect(request.generation).toBe(generation);
  expect(request.scenes).toEqual([
    { prompt: String.raw`Scene\one`, duration_ms: "10000" },
    { prompt: "Scene two", duration_ms: "20000" },
  ]);
  expect(request.duration_ms).toBe("30000");
  expect(() =>
    buildDurationVideoRequest({ ...draft, durationMs: "29999" }, generation),
  ).toThrow("add up");
});
it("requires an explicit scene sum and never drops an invalid scene", () => {
  const draft = {
    durationMs: "1000",
    scenesEnabled: true,
    scenes: [{ id: "a", prompt: "", durationMs: "1000" }],
  };
  expect(() => buildDurationVideoRequest(draft, {})).toThrow("Every scene");
});
it("refuses a short-generation batch clamp instead of silently dropping video copies", () => {
  expect(() => durationVideoCopies(4, 1)).toThrow("4");
  expect(durationVideoCopies(4, 4)).toBe(4);
});
