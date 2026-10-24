import { expect, it } from "vitest";
import {
  parseRequestedVideoDuration,
  formatRequestedVideoDuration,
  oneClipSceneSeconds,
  singleGenerationSeconds,
} from "./videoRequestedDuration";

it("keeps requested milliseconds exact rather than snapping to a model frame grid", () => {
  expect(parseRequestedVideoDuration("30")).toBe("30000");
  expect(parseRequestedVideoDuration("4.458")).toBe("4458");
  expect(parseRequestedVideoDuration("0.001")).toBe("1");
  expect(formatRequestedVideoDuration("4458")).toBe("4.458");
});
it("accepts long durations without a product time ceiling", () => {
  expect(parseRequestedVideoDuration("86400")).toBe("86400000");
  expect(parseRequestedVideoDuration("9007199254740.991")).toBe(
    "9007199254740991",
  );
});
it("refuses invalid, ambiguous and unrepresentable values without inventing zero", () => {
  for (const value of [
    "",
    " ",
    "0",
    "-1",
    "NaN",
    "Infinity",
    "1e3",
    "1.0001",
    "18446744073709551.616",
    "1,5",
  ])
    expect(parseRequestedVideoDuration(value)).toBeNull();
});

it("round-trips the full server millisecond range without JavaScript rounding", () => {
  expect(parseRequestedVideoDuration("18446744073709551.615")).toBe(
    "18446744073709551615",
  );
  expect(formatRequestedVideoDuration("18446744073709551615")).toBe(
    "18446744073709551.615",
  );
  expect(parseRequestedVideoDuration("9007199254740.992")).toBe(
    "9007199254740992",
  );
});

it("states the one-generation ceiling and the always-one-clip scene length", () => {
  const h3 = { fps: 24, single_clip_max_frames: 345, overlap_frames: 1 };
  expect(singleGenerationSeconds(h3)).toBe("14.375");
  // (345 - 1) / 24 = 14.33…, floored to a tenth so a scene this long never
  // spills into a second clip.
  expect(oneClipSceneSeconds(h3)).toBe("14.3");
  expect(oneClipSceneSeconds({ ...h3, overlap_frames: 9 })).toBe("14");
  expect(singleGenerationSeconds({ ...h3, fps: 0 })).toBeNull();
  expect(oneClipSceneSeconds({ ...h3, overlap_frames: 345 })).toBeNull();
  expect(
    oneClipSceneSeconds({ fps: 24, single_clip_max_frames: 345 }),
  ).toBeNull();
});
