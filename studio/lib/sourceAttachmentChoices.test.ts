import { expect, it } from "vitest";
import {
  imageAttachmentChoices,
  replaceEndpointKeyframe,
} from "./sourceAttachmentChoices";
const caps = {
  supportsKeyframes: false,
  supportsEndFrame: false,
  supportsSourceImage: true,
  sourceImageMode: "single",
  requiresSourceImage: false,
};
it("uses capability endpoints and keyframes without model-name routing", () => {
  expect(
    imageAttachmentChoices({
      ...caps,
      sourceImageMode: "h3-boundaries",
      requiresSourceImage: true,
    }),
  ).toEqual(["first"]);
  expect(
    imageAttachmentChoices({ ...caps, sourceImageMode: "h3-boundaries" }),
  ).toEqual(["first", "last"]);
  expect(imageAttachmentChoices({ ...caps, supportsEndFrame: true })).toEqual([
    "first",
    "last",
  ]);
  expect(imageAttachmentChoices({ ...caps, supportsKeyframes: true })).toEqual([
    "first",
    "last",
  ]);
  expect(
    imageAttachmentChoices({ ...caps, supportsSourceImage: false }),
  ).toEqual([]);
});
it("replaces one endpoint preserving intermediate keyframes, strengths, order and canvas", () => {
  const current = [
    { frame: 0, image: "old", strength: 0.3 },
    { frame: 12, image: "middle", strength: 0.8 },
    { frame: 24, image: "end" },
  ];
  expect(replaceEndpointKeyframe(current, 25, "first", "new")).toEqual([
    { frame: 0, image: "new", strength: 0.3 },
    current[1],
    current[2],
  ]);
  expect(
    replaceEndpointKeyframe(current.slice(0, 2), 25, "last", "new"),
  ).toEqual([current[0], current[1], { frame: 24, image: "new" }]);
  expect(() => replaceEndpointKeyframe(current, 0, "last", "new")).toThrow(
    "frame count",
  );
});

it("honors explicit endpoint refusal and required-first profiles before legacy hints", () => {
  const boundary = {
    mode: "hidden" as const,
    first_required: false,
    last_required: false,
    min_frames: 2,
    wire: "h3-endpoints",
  };
  expect(
    imageAttachmentChoices({
      ...caps,
      sourceImageMode: "h3-boundaries",
      supportsKeyframes: true,
      boundaryFrames: boundary,
    }),
  ).toEqual([]);
  expect(
    imageAttachmentChoices({
      ...caps,
      boundaryFrames: { ...boundary, mode: "adjustable", first_required: true },
    }),
  ).toEqual(["first"]);
});
