import { expect, it } from "vitest";
import { publishedSeed, videoSettingsForReuse } from "./videoSettingsReuse";
it("restores requested added time, literal scenes and exact seed without linking the original", () => {
  const value = videoSettingsForReuse({
    frames: 480,
    fps: 24,
    video_job: {
      source_frames: "240",
      authoring: {
        duration_ms: "10000",
        boundary_frames: 124,
        input_frames: 107,
        seed: "18446744073709551615",
        scenes: [
          { prompt: "  first\\n\r\n", duration_ms: "4000" },
          { prompt: "last", duration_ms: "6000" },
        ],
      },
    },
  });
  expect(value?.draft).toMatchObject({
    durationMs: "10000",
    scenesEnabled: true,
    scenes: [
      { prompt: "  first\\n\r\n", durationMs: "4000" },
      { prompt: "last", durationMs: "6000" },
    ],
  });
  expect(value?.draft.extension).toBeUndefined();
  expect(value?.draft.reuseNotice).toContain("original video is not attached");
  expect(value?.frames).toBe(107);
  expect(value?.seed).toBe("18446744073709551615");
});
it("refuses incomplete parent provenance rather than guessing total length", () => {
  expect(() =>
    videoSettingsForReuse({ frames: 480, video_job: { id: "job" } }),
  ).toThrow(/older video/);
});
it("refuses scene drift and invalid decimal seeds before applying settings", () => {
  const a = {
    duration_ms: "1000",
    boundary_frames: 124,
    seed: "18446744073709551616",
    scenes: [{ prompt: "one", duration_ms: "999" }],
  };
  expect(() =>
    videoSettingsForReuse({ video_job: { authoring: a } }),
  ).toThrow();
});
it("ordinary outputs have no duration handoff", () => {
  expect(videoSettingsForReuse({ seed: 42 })).toBeNull();
});
it("increments a full u64 exactly and refuses overflow", async () => {
  const { incrementExactSeed, seedFromInput } =
    await import("./videoSettingsReuse");
  expect(incrementExactSeed("18446744073709551614", 1)).toBe(
    "18446744073709551615",
  );
  expect(() => incrementExactSeed("18446744073709551615", 1)).toThrow(/range/);
  expect(seedFromInput("18446744073709551615")).toBe("18446744073709551615");
  expect(seedFromInput("42")).toBe(42);
});
it("bounds scene bytes and the complete serialized authoring record without trimming", () => {
  const a = {
    duration_ms: "1000",
    boundary_frames: 124,
    seed: "1",
    scenes: [{ prompt: "é".repeat(38501), duration_ms: "1000" }],
  };
  expect(() => videoSettingsForReuse({ video_job: { authoring: a } })).toThrow(
    /prompt/,
  );
  const scenes = Array.from({ length: 20 }, () => ({
    prompt: "a".repeat(60000),
    duration_ms: "1000",
  }));
  expect(() =>
    videoSettingsForReuse({
      video_job: { authoring: { ...a, duration_ms: "20000", scenes } },
    }),
  ).toThrow(/size/);
});

it("displays authoritative decimal seeds instead of rounded JSON numbers", () => {
  expect(
    publishedSeed({
      seed: 11110663010851220000,
      video_job: { authoring: { seed: "11110663010851218914" } },
    }),
  ).toBe("11110663010851218914");
  expect(
    publishedSeed({
      seed: 18446744073709552000,
      chain: { stages: [{ seed: "18446744073709551615" }] },
    }),
  ).toBe("18446744073709551615");
  expect(publishedSeed({ seed: 42 })).toBe(42);
  expect(publishedSeed({ seed: 18446744073709552000 })).toBeNull();
});
