import { describe, expect, it } from "vitest";
import {
  referenceCountRefusal,
  resolveDropTarget,
  NO_IMAGE_INPUT_REFUSAL,
  type DropRoutingState,
} from "./imageDropRouting";
import type { SourceMediaPlan } from "./sourceMediaPlan";

const single: SourceMediaPlan = {
  kind: "single",
  required: false,
  endFrame: true,
  video: true,
};
const attachments: SourceMediaPlan = {
  kind: "attachments",
  max: 4,
  required: false,
  primary: null,
};
const qwen: SourceMediaPlan = {
  kind: "attachments",
  max: null,
  required: true,
  primary: "target",
};
const klein: SourceMediaPlan = {
  kind: "single-or-references",
  single: { required: false, endFrame: false, video: false },
  references: { max: 4, maxPixelsSingle: null, maxPixelsMulti: null },
};
const h3Boundaries: SourceMediaPlan = {
  kind: "h3-boundaries",
  requiredEndpoint: null,
};
const h3References: SourceMediaPlan = { kind: "h3-references" };

function state(overrides: Partial<DropRoutingState> = {}): DropRoutingState {
  return {
    hasSource: false,
    referenceCount: 0,
    identityVisible: false,
    openingVisible: false,
    h3FirstPresent: false,
    ...overrides,
  };
}

describe("resolveDropTarget without a well under the cursor", () => {
  it("routes each plan kind to its own default well", () => {
    expect(resolveDropTarget(single, null, state())).toBe("source");
    expect(resolveDropTarget(attachments, null, state())).toBe("references");
    expect(resolveDropTarget(qwen, null, state())).toBe("references");
    expect(resolveDropTarget(h3Boundaries, null, state())).toBe("h3-first");
    expect(
      resolveDropTarget(h3Boundaries, null, state({ h3FirstPresent: true })),
    ).toBe("h3-last");
    expect(resolveDropTarget(h3References, null, state())).toBe("h3-reference");
  });

  it("refuses a model that takes no image at all", () => {
    expect(resolveDropTarget({ kind: "none" }, null, state())).toEqual({
      refused: NO_IMAGE_INPUT_REFUSAL,
    });
    // …with the profile's own sentence when the recipe supplied one.
    expect(
      resolveDropTarget(
        { kind: "none" },
        null,
        state({
          refusalReason: "This model does not accept reference images.",
        }),
      ),
    ).toEqual({ refused: "This model does not accept reference images." });
  });

  it("sends a Klein drop to the well that already holds media", () => {
    expect(resolveDropTarget(klein, null, state())).toBe("source");
    expect(resolveDropTarget(klein, null, state({ referenceCount: 2 }))).toBe(
      "references",
    );
    expect(resolveDropTarget(klein, null, state({ hasSource: true }))).toBe(
      "source",
    );
    // Both present: the last write is the active well, and a drop with no
    // well under the cursor lands there.
    expect(
      resolveDropTarget(
        klein,
        null,
        state({ hasSource: true, referenceCount: 1, lastWrite: "references" }),
      ),
    ).toBe("references");
  });
});

describe("resolveDropTarget with a well under the cursor", () => {
  it("lets the hovered well win over the plan default", () => {
    expect(resolveDropTarget(single, "end", state())).toBe("end");
    expect(resolveDropTarget(klein, "references", state())).toBe("references");
    // Klein in the other direction: dropping on Source while references are
    // attached parks the references rather than appending to them.
    expect(
      resolveDropTarget(klein, "source", state({ referenceCount: 2 })),
    ).toBe("source");
    expect(resolveDropTarget(h3References, "h3-reference", state())).toBe(
      "h3-reference",
    );
    expect(resolveDropTarget(h3Boundaries, "h3-last", state())).toBe("h3-last");
  });

  it("reaches the identity and sequence wells, which no default ever picks", () => {
    expect(
      resolveDropTarget(single, "identity", state({ identityVisible: true })),
    ).toBe("identity");
    expect(
      resolveDropTarget(single, "opening", state({ openingVisible: true })),
    ).toBe("opening");
    // A well this form is not rendering falls back to the plan default
    // instead of writing a field nobody can see.
    expect(resolveDropTarget(single, "identity", state())).toBe("source");
    expect(resolveDropTarget(single, "opening", state())).toBe("source");
  });

  it("falls back to the plan default for a well the plan does not render", () => {
    expect(resolveDropTarget(attachments, "source", state())).toBe(
      "references",
    );
    expect(resolveDropTarget(single, "references", state())).toBe("source");
    expect(resolveDropTarget(single, "h3-first", state())).toBe("source");
    expect(resolveDropTarget(h3Boundaries, "references", state())).toBe(
      "h3-first",
    );
    // Qwen's Target IS the shared source well, so it stays reachable.
    expect(resolveDropTarget(qwen, "source", state())).toBe("source");
    // …and an end-frame well only exists where the plan renders one.
    expect(resolveDropTarget(single, "end", state())).toBe("end");
    expect(
      resolveDropTarget(
        { kind: "single", required: false, endFrame: false, video: false },
        "end",
        state(),
      ),
    ).toBe("source");
  });
});

describe("resolveDropTarget strip bounds", () => {
  it("refuses a strip that is already at the advertised ceiling", () => {
    expect(
      resolveDropTarget(attachments, null, state({ referenceCount: 4 })),
    ).toEqual({ refused: referenceCountRefusal(4) });
    expect(
      resolveDropTarget(
        attachments,
        "references",
        state({ referenceCount: 4 }),
      ),
    ).toEqual({ refused: referenceCountRefusal(4) });
    expect(
      resolveDropTarget(klein, "references", state({ referenceCount: 4 })),
    ).toEqual({ refused: referenceCountRefusal(4) });
  });

  it("keeps appending while an unbounded strip grows", () => {
    expect(resolveDropTarget(qwen, null, state({ referenceCount: 12 }))).toBe(
      "references",
    );
  });

  it("bounds the H3 reference panel by its own budget", () => {
    expect(
      resolveDropTarget(
        h3References,
        null,
        state({ h3ReferenceCount: 9, h3ReferenceMax: 9 }),
      ),
    ).toEqual({ refused: referenceCountRefusal(9) });
    expect(
      resolveDropTarget(
        h3References,
        null,
        state({ h3ReferenceCount: 8, h3ReferenceMax: 9 }),
      ),
    ).toBe("h3-reference");
  });
});
