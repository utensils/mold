import { describe, expect, it } from "vitest";
import type { PreparedExpansionBatch, QuickExpansionSnapshot } from "../lib/preparedExpansion";
import {
  capturePreparedSubmission,
  captureQuickSubmission,
  preparedSubmissionIsCurrent,
  quickSubmissionIsCurrent,
} from "./mobileGenerationSnapshot";

function prepared(): PreparedExpansionBatch {
  return {
    kind: "remix",
    batchId: "batch-1",
    sourcePrompt: "source",
    rootPrompt: "root",
    sourceKind: "current",
    model: "flux:test",
    family: "flux",
    task: "text-to-image",
    requestedCount: 2,
    stylePreset: null,
    selectedHostPolicy: "studio",
    dimensions: ["composition"],
    prompts: [
      { id: "one", text: " first " },
      { id: "two", text: "second" },
    ],
    route: {
      hostId: "studio",
      label: "Studio",
      kind: "remote",
      target: { baseUrl: "http://studio.test:7680", apiKey: "secret" },
    },
  };
}

function quick(): QuickExpansionSnapshot {
  return {
    requestToken: 7,
    originalPrompt: "source",
    expandedPrompt: "expanded",
    model: "flux:test",
    family: "flux",
    task: "text-to-image",
    stylePreset: null,
    selectedHostPolicy: "studio",
    route: prepared().route,
  };
}

describe("mobile generation submission snapshots", () => {
  it("freezes trimmed prepared prompts, route credentials, and remix provenance", () => {
    const source = prepared();
    const snapshot = capturePreparedSubmission(source)!;
    source.prompts[0]!.text = "changed";
    source.route.target.apiKey = "changed";

    expect(snapshot.prompts).toEqual(["first", "second"]);
    expect(snapshot.route.target.apiKey).toBe("secret");
    expect(snapshot.promptTransforms).toEqual([
      expect.objectContaining({
        operation: "remix",
        root_prompt: "root",
        dimensions: ["composition"],
      }),
      expect.objectContaining({
        operation: "remix",
        root_prompt: "root",
        dimensions: ["composition"],
      }),
    ]);
  });

  it("rejects prepared work after prompt identity or staleness changes", () => {
    const current = prepared();
    const snapshot = capturePreparedSubmission(current)!;
    expect(preparedSubmissionIsCurrent(snapshot, current, [])).toBe(true);
    current.prompts[1]!.text = "different";
    expect(preparedSubmissionIsCurrent(snapshot, current, [])).toBe(false);
    expect(preparedSubmissionIsCurrent(snapshot, prepared(), ["host changed"])).toBe(false);
  });

  it("freezes quick routing and rejects a replaced request token", () => {
    const current = quick();
    const snapshot = captureQuickSubmission(current)!;
    current.route.target.apiKey = "changed";

    expect(snapshot.route.target.apiKey).toBe("secret");
    expect(quickSubmissionIsCurrent(snapshot, current, [])).toBe(true);
    expect(quickSubmissionIsCurrent(snapshot, { ...current, requestToken: 8 }, [])).toBe(false);
  });
});
