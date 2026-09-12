import { describe, expect, it } from "vitest";
import {
  hostSelectionLabel,
  preparedExpansionStaleReasons,
  quickExpansionStaleReasons,
  type PreparedExpansionStaleBatch,
  type PreparedExpansionStaleInputs,
  type QuickExpansionStaleInputs,
  type QuickExpansionStaleSnapshot,
} from "./preparedExpansion";

/*
 * The stale-reason rule is ONE rule. Desktop and web both read it, so its
 * sentences are pinned HERE rather than in either surface's suite — a reason
 * that reads differently on two screens is the bug this hoist retires.
 */

const ROUTE = {
  hostId: "studio",
  label: "Studio 4090",
  kind: "remote",
  instanceId: "inst-1",
  target: { baseUrl: "http://studio:7680", apiKey: "k" },
};

function batch(
  overrides: Partial<PreparedExpansionStaleBatch> = {},
): PreparedExpansionStaleBatch {
  return {
    sourcePrompt: "a lighthouse",
    model: "flux-dev:q8",
    family: "flux",
    task: "text-to-image",
    requestedCount: 3,
    selectedHostPolicy: null,
    route: ROUTE,
    ...overrides,
  };
}

function current(
  overrides: Partial<PreparedExpansionStaleInputs> = {},
): PreparedExpansionStaleInputs {
  return {
    sourcePrompt: "a lighthouse",
    model: "flux-dev:q8",
    family: "flux",
    task: "text-to-image",
    requestedCount: 3,
    selectedHostPolicy: null,
    readyHostIds: new Set(["studio"]),
    hostLabels: new Map([["studio", "Studio 4090"]]),
    hostTargets: new Map([["studio", { ...ROUTE.target, kind: "remote" }]]),
    ...overrides,
  };
}

describe("preparedExpansionStaleReasons", () => {
  it("says nothing while the form still matches the reviewed work", () => {
    expect(preparedExpansionStaleReasons(batch(), current())).toEqual([]);
  });

  it("names the style, its family, the conditioning, the count and the machine in the lexicon's words", () => {
    const reasons = preparedExpansionStaleReasons(
      batch(),
      current({
        sourcePrompt: "a lighthouse at dusk",
        model: "sdxl-base:fp16",
        family: "sdxl",
        task: "image-to-video",
        requestedCount: 5,
        selectedHostPolicy: "capable",
      }),
    );
    expect(reasons).toEqual([
      "Source prompt changed after these variations were prepared.",
      'Style changed from "flux-dev:q8" to "sdxl-base:fp16".',
      'Style family changed from "flux" to "sdxl".',
      "Conditioning changed from text-to-image to image-to-video.",
      "Batch changed from 3 to 5.",
      "Machine selection changed from Auto to Most capable.",
    ]);
    // The retired words never come back through the shared rule.
    for (const reason of reasons) {
      expect(reason).not.toMatch(/\bhost\b/i);
      expect(reason).not.toMatch(/\bmodel\b/i);
    }
  });

  it("prefers a display name for a style when the caller supplies one", () => {
    expect(
      preparedExpansionStaleReasons(
        batch(),
        current({
          model: "sdxl-base:fp16",
          family: "sdxl",
          modelLabels: new Map([
            ["flux-dev:q8", "Photoreal"],
            ["sdxl-base:fp16", "Illustration"],
          ]),
        }),
      )[0],
    ).toBe('Style changed from "Photoreal" to "Illustration".');
  });

  it("reports an unreachable machine by its frozen label", () => {
    expect(
      preparedExpansionStaleReasons(
        batch(),
        current({ readyHostIds: new Set<string>() }),
      ),
    ).toEqual(["Studio 4090 is no longer reachable."]);
  });

  it("reports a reachable machine whose connection details moved", () => {
    expect(
      preparedExpansionStaleReasons(
        batch(),
        current({
          hostTargets: new Map([
            [
              "studio",
              {
                baseUrl: "http://studio:7681",
                apiKey: "k",
                kind: "remote",
              },
            ],
          ]),
        }),
      ),
    ).toEqual(["Studio 4090's connection details changed."]);
  });

  it("reports conditioning media only once a fingerprint was frozen", () => {
    expect(
      preparedExpansionStaleReasons(
        batch(),
        current({ conditioningFingerprint: "sha-2" }),
      ),
    ).toEqual([]);
    expect(
      preparedExpansionStaleReasons(
        batch({ conditioningFingerprint: "sha-1" }),
        current({ conditioningFingerprint: "sha-2" }),
      ),
    ).toEqual([
      "Conditioning media changed after these variations were prepared.",
    ]);
  });

  it("watches the remix dimensions only for a remix batch", () => {
    expect(
      preparedExpansionStaleReasons(
        batch({ kind: "remix", dimensions: ["lighting"] }),
        current({ dimensions: ["camera"] }),
      ),
    ).toEqual([
      "Remix dimensions changed after these variations were prepared.",
    ]);
    expect(
      preparedExpansionStaleReasons(
        batch({ dimensions: ["lighting"] }),
        current({ dimensions: ["camera"] }),
      ),
    ).toEqual([]);
  });
});

describe("quickExpansionStaleReasons", () => {
  const snapshot: QuickExpansionStaleSnapshot = {
    expandedPrompt: "a lighthouse in storm light",
    model: "flux-dev:q8",
    family: "flux",
    task: "text-to-image",
  };
  const live: QuickExpansionStaleInputs = { ...snapshot };

  it("says nothing while the rewrite is still what the composer shows", () => {
    expect(quickExpansionStaleReasons(snapshot, live)).toEqual([]);
  });

  it("names a hand edit, a style change and a conditioning change", () => {
    expect(
      quickExpansionStaleReasons(snapshot, {
        ...live,
        expandedPrompt: "hand edited",
        model: "sdxl-base:fp16",
        family: "sdxl",
        task: "image-to-video",
      }),
    ).toEqual([
      "Expanded prompt changed after it was prepared.",
      'Style changed from "flux-dev:q8" to "sdxl-base:fp16".',
      'Style family changed from "flux" to "sdxl".',
      "Conditioning changed from text-to-image to image-to-video.",
    ]);
  });
});

describe("hostSelectionLabel", () => {
  it("names the two automatic policies and looks a pinned machine up", () => {
    expect(hostSelectionLabel(null)).toBe("Auto");
    expect(hostSelectionLabel("capable")).toBe("Most capable");
    expect(
      hostSelectionLabel("studio", new Map([["studio", "Studio 4090"]])),
    ).toBe("Studio 4090");
    expect(hostSelectionLabel("ghost")).toBe("ghost");
  });
});
