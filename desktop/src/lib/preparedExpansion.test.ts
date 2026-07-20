import { describe, expect, it } from "vitest";
import type { HostRoute } from "../stores/hosts";
import {
  PreparationRequestGuard,
  createPreparedExpansionBatch,
  quickExpansionStaleReasons,
  preparedExpansionStaleReasons,
  validateExpandedPrompts,
  type PreparedExpansionInputs,
} from "./preparedExpansion";

const route: HostRoute = {
  hostId: "studio-4090",
  label: "Studio 4090",
  kind: "remote",
  target: { baseUrl: "http://studio:7680", apiKey: "secret" },
};

const inputs: PreparedExpansionInputs = {
  sourcePrompt: "a lighthouse at dusk",
  model: "flux-dev:q8",
  family: "flux",
  requestedCount: 3,
  selectedHostPolicy: null,
};

describe("prepared expansion validation", () => {
  it("accepts exactly N non-empty prompts and trims their edges", () => {
    expect(validateExpandedPrompts([" one ", "two", " three"], 3)).toEqual(["one", "two", "three"]);
  });

  it("rejects short, long, and blank responses without silently resizing", () => {
    expect(() => validateExpandedPrompts(["one", "two"], 3)).toThrow(
      "Expected exactly 3 non-empty prompts, but the host returned 2.",
    );
    expect(() => validateExpandedPrompts(["one", "two", "three", "four"], 3)).toThrow(
      "Expected exactly 3 non-empty prompts, but the host returned 4.",
    );
    expect(() => validateExpandedPrompts(["one", "  ", "three"], 3)).toThrow("Prompt 2 was empty");
  });
});

describe("prepared expansion lifecycle", () => {
  it("keeps stable prompt ids and the complete route/input provenance", () => {
    const batch = createPreparedExpansionBatch(
      inputs,
      route,
      ["one", "two", "three"],
      7,
      "prepared-stable-id",
    );

    expect(batch.prompts.map((prompt) => prompt.id)).toEqual([
      "prepared-7-1",
      "prepared-7-2",
      "prepared-7-3",
    ]);
    expect(batch).toMatchObject({ ...inputs, route, batchId: "prepared-stable-id" });
  });

  it("names every upstream mismatch while preserving edited prompts", () => {
    const batch = createPreparedExpansionBatch(inputs, route, ["edited one", "two", "three"], 1);
    const reasons = preparedExpansionStaleReasons(batch, {
      sourcePrompt: "a lighthouse at dawn",
      model: "sdxl-base:fp16",
      family: "sdxl",
      requestedCount: 5,
      selectedHostPolicy: "capable",
      readyHostIds: new Set([route.hostId]),
      hostLabels: new Map([[route.hostId, route.label]]),
    });

    expect(reasons).toEqual([
      "Source prompt changed after these variations were prepared.",
      'Model changed from "flux-dev:q8" to "sdxl-base:fp16".',
      'Model family changed from "flux" to "sdxl".',
      "Batch changed from 3 to 5.",
      "Host selection changed from Auto to Most capable.",
    ]);
    expect(batch.prompts[0]!.text).toBe("edited one");
  });

  it("marks only the frozen host unavailable without re-resolving Auto", () => {
    const batch = createPreparedExpansionBatch(inputs, route, ["one", "two", "three"], 1);

    expect(
      preparedExpansionStaleReasons(batch, {
        ...inputs,
        readyHostIds: new Set(),
        hostLabels: new Map([[route.hostId, route.label]]),
      }),
    ).toEqual(["Studio 4090 is no longer reachable."]);
  });

  it("marks changed connection details stale instead of using an old route", () => {
    const batch = createPreparedExpansionBatch(inputs, route, ["one", "two", "three"], 1);

    expect(
      preparedExpansionStaleReasons(batch, {
        ...inputs,
        readyHostIds: new Set([route.hostId]),
        hostLabels: new Map([[route.hostId, route.label]]),
        hostTargets: new Map([
          [route.hostId, { baseUrl: "http://studio-new:7680", apiKey: "secret", kind: "remote" }],
        ]),
      }),
    ).toEqual(["Studio 4090's connection details changed."]);
  });

  it("lets only the newest request apply and invalidates a discarded request", () => {
    const guard = new PreparationRequestGuard();
    const first = guard.begin();
    const second = guard.begin();
    expect(guard.isCurrent(first)).toBe(false);
    expect(guard.isCurrent(second)).toBe(true);

    guard.invalidate();
    expect(guard.isCurrent(second)).toBe(false);
  });

  it("applies prepared-style stale checks to a frozen quick expansion route", () => {
    expect(
      quickExpansionStaleReasons(
        {
          requestToken: 7,
          originalPrompt: "a lighthouse",
          expandedPrompt: "a detailed lighthouse",
          model: inputs.model,
          family: inputs.family,
          selectedHostPolicy: null,
          route,
        },
        {
          expandedPrompt: "edited after expand",
          model: inputs.model,
          family: inputs.family,
          selectedHostPolicy: "capable",
          readyHostIds: new Set([route.hostId]),
          hostLabels: new Map([[route.hostId, route.label]]),
          hostTargets: new Map([[route.hostId, { ...route.target, kind: route.kind }]]),
        },
      ),
    ).toEqual([
      "Expanded prompt changed after it was prepared.",
      "Host selection changed from Auto to Most capable.",
    ]);
  });
});
