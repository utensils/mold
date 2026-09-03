import { describe, expect, it } from "vitest";
import {
  applyCreateDrop,
  routeCreateDrop,
  type CreateDropContext,
} from "./createImageDrop";
import { __testing__ as generateFormTesting } from "../composables/useGenerateForm";
import type { SourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import type { GenerateFormState, SourceImageState } from "../types";

const single: SourceMediaPlan = {
  kind: "single",
  required: false,
  endFrame: true,
  video: false,
};
const attachments: SourceMediaPlan = {
  kind: "attachments",
  max: 4,
  required: false,
  primary: null,
};
const klein: SourceMediaPlan = {
  kind: "single-or-references",
  single: { required: false, endFrame: false, video: false },
  references: { max: 4, maxPixelsSingle: null, maxPixelsMulti: null },
};

function state(): GenerateFormState {
  return generateFormTesting.defaultForm();
}

function context(
  plan: SourceMediaPlan,
  overrides: Partial<CreateDropContext> = {},
): CreateDropContext {
  return {
    plan,
    referenceMax: plan.kind === "attachments" ? plan.max : 4,
    refusalReason: null,
    identityVisible: false,
    openingVisible: false,
    ...overrides,
  };
}

function image(filename = "dropped.png"): SourceImageState {
  return {
    kind: "upload",
    filename,
    base64: `BYTES_${filename}`,
    width: 1024,
    height: 768,
    mime: "image/png",
  };
}

describe("routeCreateDrop", () => {
  it("sends a drop with no well under it to the plan's default", () => {
    expect(routeCreateDrop(state(), context(single))).toBe("source");
    expect(routeCreateDrop(state(), context(attachments))).toBe("references");
  });

  it("refuses a model that takes no image, in the server's own sentence", () => {
    expect(
      routeCreateDrop(
        state(),
        context(
          { kind: "none" },
          { refusalReason: "This model does not accept reference images." },
        ),
      ),
    ).toEqual({ refused: "This model does not accept reference images." });
  });

  it("counts an exclusive recipe's references from their own store", () => {
    const s = state();
    s.referenceImages = [image("a.png"), image("b.png")];
    // References hold the media, so an unaimed drop keeps appending to them.
    expect(routeCreateDrop(s, context(klein))).toBe("references");
    // …and a full strip refuses rather than silently dropping the file.
    s.referenceImages = [
      image("a.png"),
      image("b.png"),
      image("c.png"),
      image("d.png"),
    ];
    expect(routeCreateDrop(s, context(klein))).toMatchObject({
      refused: expect.stringContaining("at most 4"),
    });
  });
});

describe("applyCreateDrop", () => {
  it("writes the source well and records the last write", async () => {
    const s = state();
    expect(await applyCreateDrop(s, "source", image(), context(single))).toBe(
      null,
    );
    expect(s.imageAttachments).toHaveLength(1);
    expect(s.imageAttachments[0]?.filename).toBe("dropped.png");
    expect(s.sourceFitPolicy).toEqual({ mode: "crop-fill" });
    expect(s.exclusiveWell).toBe("source");
  });

  it("APPENDS to a strip instead of replacing it", async () => {
    const s = state();
    s.imageAttachments = [image("target.png"), image("ref.png")];
    await applyCreateDrop(s, "references", image(), context(attachments));
    expect(s.imageAttachments.map((m) => m.filename)).toEqual([
      "target.png",
      "ref.png",
      "dropped.png",
    ]);
  });

  it("parks rather than discards on an exclusive recipe, both ways", async () => {
    const s = state();
    s.imageAttachments = [image("source.png")];
    await applyCreateDrop(s, "references", image("ref.png"), context(klein));
    // The source is PARKED, not discarded — it comes back when the strip
    // clears — and the references are the active well.
    expect(s.imageAttachments.map((m) => m.filename)).toEqual(["source.png"]);
    expect(s.referenceImages?.map((m) => m.filename)).toEqual(["ref.png"]);
    expect(s.exclusiveWell).toBe("references");

    await applyCreateDrop(s, "source", image("other.png"), context(klein));
    expect(s.referenceImages?.map((m) => m.filename)).toEqual(["ref.png"]);
    expect(s.imageAttachments.map((m) => m.filename)).toEqual(["other.png"]);
    expect(s.exclusiveWell).toBe("source");
  });

  it("reaches the end frame, the identity photo and the sequence opening", async () => {
    const s = state();
    await applyCreateDrop(s, "end", image("close.png"), context(single));
    expect(s.endFrame?.filename).toBe("close.png");

    await applyCreateDrop(s, "identity", image("face.png"), context(single));
    expect(s.identityImage?.filename).toBe("face.png");

    const draft = {
      openingImage: null as { filename: string; base64: string | null } | null,
    };
    await applyCreateDrop(
      s,
      "opening",
      image("open.png"),
      context(single),
      draft,
    );
    expect(draft.openingImage).toEqual({
      filename: "open.png",
      base64: "BYTES_open.png",
    });
  });

  it("writes H3 boundaries to the authoring state, never the strip", async () => {
    const s = state();
    expect(
      await applyCreateDrop(s, "h3-first", image(), context(single)),
    ).toBeNull();
    expect(s.imageAttachments).toHaveLength(0);
    expect(s.h3Authoring?.firstFrame?.data).toBe("BYTES_dropped.png");
  });
});
