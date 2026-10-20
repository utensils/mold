import { describe, expect, it, vi } from "vitest";
import { prepareClipsInputs } from "./chainInputPreparation";
import type { SourceFitTransform } from "./sourceFit";
import { buildClipsRequest } from "./chainDraft";

const base = {
  model: "video",
  prompt: "Opening",
  width: 768,
  height: 768,
  frames: 107,
  steps: 3,
  source_image: "opening",
  source_image_name: "hero.png",
  source_fit: { mode: "pad-fit" as const },
  keyframes: [{ frame: 106, image: "closing", image_name: "ending.png" }],
  references: [
    {
      role: "reference_image",
      media: {
        authority: "inline",
        data: "reference",
        width: 1344,
        height: 768,
      },
    },
  ],
};
function dependencies() {
  return {
    current: () => true,
    ops: {
      imageSize: vi.fn(async () => ({ width: 1344, height: 768 })),
      fitImage: vi.fn(
        async (data: string, _transform: SourceFitTransform) =>
          `${data}-fitted`,
      ),
      buildMask: vi.fn(),
    },
  };
}
describe("authored clip input preparation", () => {
  it("fits opening and closing pixels once before stage expansion, retaining names, fit and inline references", async () => {
    const deps = dependencies();
    const prepared = await prepareClipsInputs(base, deps);
    expect(deps.ops.fitImage).toHaveBeenCalledTimes(2);
    expect(deps.ops.fitImage.mock.calls[0]?.[1]).toMatchObject({
      outputWidth: 768,
      outputHeight: 768,
      offsetY: 165,
    });
    expect(prepared).toMatchObject({
      source_image: "opening-fitted",
      source_image_name: "hero.png",
      source_fit: { mode: "pad-fit" },
      keyframes: [
        { frame: 106, image: "closing-fitted", image_name: "ending.png" },
      ],
      references: base.references,
    });
    expect(base.source_image).toBe("opening");
    expect(base.keyframes[0]?.image).toBe("closing");
    const wire = buildClipsRequest(
      prepared,
      [
        { id: "a", prompt: "Opening", frames: 107 },
        { id: "b", prompt: "Ending", frames: 124 },
      ],
      1,
    );
    expect(wire.stages[0]?.source_image).toBe("opening-fitted");
    expect(wire.stages[1]?.keyframes).toMatchObject([
      { frame: 123, image: "closing-fitted" },
    ]);
    expect(deps.ops.buildMask).not.toHaveBeenCalled();
  });
  it("refuses an invalidated draft before fitting the next endpoint", async () => {
    let current = true;
    const deps = dependencies();
    deps.current = () => current;
    deps.ops.imageSize.mockImplementationOnce(async () => {
      current = false;
      return { width: 1344, height: 768 };
    });
    await expect(prepareClipsInputs(base, deps)).rejects.toThrow("changed");
    expect(deps.ops.fitImage).not.toHaveBeenCalled();
    expect(deps.ops.imageSize).toHaveBeenCalledTimes(1);
  });
  it("does not decode or fit typed references or already-sized endpoints", async () => {
    const deps = dependencies();
    deps.ops.imageSize.mockResolvedValue({ width: 768, height: 768 });
    const prepared = await prepareClipsInputs(base, deps);
    expect(deps.ops.fitImage).not.toHaveBeenCalled();
    expect(deps.ops.imageSize.mock.calls).toEqual([["opening"], ["closing"]]);
    expect(prepared.references).toEqual(base.references);
  });
});

it("applies a pending reference crop at original resolution without changing the original, then repeats canonical metadata", async () => {
  const deps = dependencies();
  deps.ops.fitImage.mockResolvedValue("Y3JvcHBlZA==");
  const reference = {
    kind: "image" as const,
    mime_type: "image/png",
    width: 1344,
    height: 768,
    media: { authority: "inline" as const, data: "b3JpZ2luYWw=" },
    provenance: { name: "hero.png", sha256: "original-sha" },
  };
  const state = {
    firstFrame: null,
    lastFrame: null,
    references: [
      { reference, crop: { x: 100, y: 0, width: 768, height: 768 } },
    ],
  };
  const prepared = await prepareClipsInputs(
    {
      ...base,
      source_image: undefined,
      keyframes: [],
      references: [reference],
    },
    deps,
    state,
  );
  expect(prepared.references[0]).toMatchObject({
    width: 768,
    height: 768,
    media: { authority: "inline", data: "Y3JvcHBlZA==" },
    provenance: { name: "hero.png", crop: { x: 100, width: 768 } },
  });
  expect(state.references[0]?.reference.media.data).toBe("b3JpZ2luYWw=");
  expect(state.references[0]?.crop).toBeDefined();
  const wire = buildClipsRequest(
    prepared,
    [
      { id: "a", prompt: "A", frames: 107 },
      { id: "b", prompt: "B", frames: 107 },
    ],
    0,
  );
  expect(wire.stages[0]?.generation_references).toEqual(
    wire.stages[1]?.generation_references,
  );
  expect(deps.ops.imageSize).not.toHaveBeenCalled();
});

it("refuses a present future fit policy instead of silently cropping", async () => {
  const deps = dependencies();
  await expect(
    prepareClipsInputs({ ...base, source_fit: { mode: "future-fit" } }, deps),
  ).rejects.toThrow("Choose a source fit option again");
  expect(deps.ops.imageSize).not.toHaveBeenCalled();
});
it("refuses missing source upscaler rather than applying only its inner fit", async () => {
  await expect(
    prepareClipsInputs(
      {
        ...base,
        source_fit: {
          mode: "upscale-then-fit",
          upscalerModel: "",
          fit: { mode: "pad-fit" },
        },
      },
      dependencies(),
    ),
  ).rejects.toThrow("Choose a source upscaler");
});

it("shares one bounded preprocessing result for identical first and last endpoints", async () => {
  const deps = { ...dependencies(), upscale: vi.fn(async () => "upscaled") };
  const prepared = await prepareClipsInputs(
    {
      ...base,
      keyframes: [{ frame: 106, image: "opening" }],
      source_fit: {
        mode: "upscale-then-fit",
        upscalerModel: "upscaler",
        fit: { mode: "pad-fit" },
      },
    },
    deps,
  );
  expect(prepared.keyframes[0]?.image).toBe(prepared.source_image);
  expect(deps.upscale).toHaveBeenCalledTimes(1);
  expect(deps.ops.fitImage).toHaveBeenCalledTimes(1);
});

it("never revives parked or mismatched cropped references", async () => {
  const deps = dependencies();
  const state = {
    firstFrame: null,
    lastFrame: null,
    references: [
      {
        reference: {
          kind: "image" as const,
          mime_type: "image/png",
          width: 1344,
          height: 768,
          media: { authority: "inline" as const, data: "b3JpZ2luYWw=" },
        },
        crop: { x: 0, y: 0, width: 768, height: 768 },
      },
    ],
  };
  const { references: _parked, ...withoutReferences } = base;
  deps.ops.imageSize.mockResolvedValue({ width: 768, height: 768 });
  const plain = await prepareClipsInputs(withoutReferences, deps, state);
  expect(plain).not.toHaveProperty("references");
  const different = await prepareClipsInputs(
    {
      ...withoutReferences,
      references: [
        {
          ...state.references[0]!.reference,
          media: { authority: "inline", data: "different" },
        },
      ],
    },
    deps,
    state,
  );
  expect(different.references[0]?.media.data).toBe("different");
  expect(deps.ops.fitImage).not.toHaveBeenCalled();
});
