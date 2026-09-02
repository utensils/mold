import { describe, expect, it } from "vitest";
import type { HostRoute } from "../stores/hosts";
import {
  PreparationRequestGuard,
  createPreparedExpansionBatch,
  quickExpansionStaleReasons,
  quickExpansionRouteIsCurrent,
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
  task: "text-to-image",
  requestedCount: 3,
  stylePreset: null,
  selectedHostPolicy: null,
};

describe("prepared expansion validation", () => {
  it("accepts exactly N non-empty prompts and trims their edges", () => {
    expect(validateExpandedPrompts([" one ", "two", " three"], 3)).toEqual(["one", "two", "three"]);
  });

  it("unwraps singleton JSON arrays returned as individual variations", () => {
    expect(validateExpandedPrompts(['["one"]', '["two"]', '["three"]'], 3)).toEqual([
      "one",
      "two",
      "three",
    ]);
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

  // A recipe that IGNORES the prompt gets ONE advisory answer from the host
  // (the guide's image-preparation advice) whatever count was requested. Any
  // other short answer is still a malformed response.
  it("accepts the single advisory answer when the recipe ignores the prompt", () => {
    expect(validateExpandedPrompts([" prepare the image "], 3, { promptIgnored: true })).toEqual([
      "prepare the image",
    ]);
    expect(() => validateExpandedPrompts(["one", "two"], 3, { promptIgnored: true })).toThrow(
      "Expected exactly 3 non-empty prompts, but the host returned 2.",
    );
    expect(() => validateExpandedPrompts(["one"], 3)).toThrow(
      "Expected exactly 3 non-empty prompts, but the host returned 1.",
    );
    expect(() => validateExpandedPrompts(["  "], 3, { promptIgnored: true })).toThrow(
      "Prompt 1 was empty",
    );
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
      task: "image-to-video",
      requestedCount: 5,
      stylePreset: null,
      selectedHostPolicy: "capable",
      readyHostIds: new Set([route.hostId]),
      hostLabels: new Map([[route.hostId, route.label]]),
    });

    expect(reasons).toEqual([
      "Source prompt changed after these variations were prepared.",
      'Model changed from "flux-dev:q8" to "sdxl-base:fp16".',
      'Model family changed from "flux" to "sdxl".',
      "Conditioning changed from text-to-image to image-to-video.",
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

  it("marks a changed frozen host instance identity stale", () => {
    const batch = createPreparedExpansionBatch(
      inputs,
      { ...route, instanceId: "server-A" },
      ["one", "two", "three"],
      1,
    );

    expect(
      preparedExpansionStaleReasons(batch, {
        ...inputs,
        readyHostIds: new Set([route.hostId]),
        hostLabels: new Map([[route.hostId, route.label]]),
        hostTargets: new Map([
          [route.hostId, { ...route.target, kind: route.kind, instanceId: "server-B" }],
        ]),
      }),
    ).toEqual(["Studio 4090's connection details changed."]);
  });

  it("treats late instance identity enrichment as compatible", () => {
    const batch = createPreparedExpansionBatch(
      inputs,
      { ...route, instanceId: null },
      ["one", "two", "three"],
      1,
    );

    expect(
      preparedExpansionStaleReasons(batch, {
        ...inputs,
        readyHostIds: new Set([route.hostId]),
        hostLabels: new Map([[route.hostId, route.label]]),
        hostTargets: new Map([
          [route.hostId, { ...route.target, kind: route.kind, instanceId: "server-A" }],
        ]),
      }),
    ).toEqual([]);
  });

  it("names style changes, removals, and additions as specifically named stale work", () => {
    const host = {
      readyHostIds: new Set([route.hostId]),
      hostLabels: new Map([[route.hostId, route.label]]),
    };
    const styled = createPreparedExpansionBatch(
      { ...inputs, stylePreset: "cinematic" },
      route,
      ["one", "two", "three"],
      1,
    );

    expect(
      preparedExpansionStaleReasons(styled, { ...inputs, ...host, stylePreset: "anime" }),
    ).toEqual(["Style changed from Cinematic to Anime."]);
    expect(
      preparedExpansionStaleReasons(styled, { ...inputs, ...host, stylePreset: null }),
    ).toEqual(["Style Cinematic was removed after these variations were prepared."]);

    const unstyled = createPreparedExpansionBatch(inputs, route, ["one", "two", "three"], 1);
    expect(
      preparedExpansionStaleReasons(unstyled, { ...inputs, ...host, stylePreset: "anime" }),
    ).toEqual(["Style Anime was added after these variations were prepared."]);
  });

  it("treats a legacy style id and its canonical twin as the same frozen style", () => {
    const styled = createPreparedExpansionBatch(
      { ...inputs, stylePreset: "photographic" },
      route,
      ["one", "two", "three"],
      1,
    );

    expect(
      preparedExpansionStaleReasons(styled, {
        ...inputs,
        stylePreset: "photoreal",
        readyHostIds: new Set([route.hostId]),
        hostLabels: new Map([[route.hostId, route.label]]),
      }),
    ).toEqual([]);
  });

  it("lets only the newest request apply and invalidates a discarded request", () => {
    const guard = new PreparationRequestGuard();
    const first = guard.begin();
    const firstSignal = guard.signalFor(first);
    const second = guard.begin();
    const secondSignal = guard.signalFor(second);
    expect(guard.isCurrent(first)).toBe(false);
    expect(firstSignal.aborted).toBe(true);
    expect(guard.isCurrent(second)).toBe(true);
    expect(secondSignal.aborted).toBe(false);

    guard.invalidate();
    expect(guard.isCurrent(second)).toBe(false);
    expect(secondSignal.aborted).toBe(true);
    expect(() => guard.signalFor(second)).toThrow("no longer current");
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
          task: inputs.task,
          stylePreset: null,
          selectedHostPolicy: null,
          route,
        },
        {
          expandedPrompt: "edited after expand",
          model: inputs.model,
          family: inputs.family,
          task: inputs.task,
          selectedHostPolicy: "capable",
          readyHostIds: new Set([route.hostId]),
          hostLabels: new Map([[route.hostId, route.label]]),
          hostTargets: new Map([[route.hostId, { ...route.target, kind: route.kind }]]),
        },
      ),
    ).toEqual(["Expanded prompt changed after it was prepared."]);
  });

  it("treats a host-only quick change as route release, not semantic staleness", () => {
    const snapshot = {
      requestToken: 7,
      originalPrompt: "a lighthouse",
      expandedPrompt: "a cinematic lighthouse",
      model: inputs.model,
      family: inputs.family,
      task: inputs.task,
      stylePreset: null,
      selectedHostPolicy: null,
      route,
    };
    const current = {
      expandedPrompt: snapshot.expandedPrompt,
      model: snapshot.model,
      family: snapshot.family,
      task: snapshot.task,
      selectedHostPolicy: "new-host",
      readyHostIds: new Set([route.hostId]),
      hostLabels: new Map([[route.hostId, route.label]]),
    };
    expect(quickExpansionStaleReasons(snapshot, current)).toEqual([]);
    expect(quickExpansionRouteIsCurrent(snapshot, current)).toBe(false);
  });

  it("keeps a quick expansion fresh when an unknown instance identity becomes known", () => {
    expect(
      quickExpansionStaleReasons(
        {
          requestToken: 7,
          originalPrompt: "a lighthouse",
          expandedPrompt: "a detailed lighthouse",
          model: inputs.model,
          family: inputs.family,
          task: inputs.task,
          stylePreset: null,
          selectedHostPolicy: null,
          route: { ...route, instanceId: null },
        },
        {
          expandedPrompt: "a detailed lighthouse",
          model: inputs.model,
          family: inputs.family,
          task: inputs.task,
          selectedHostPolicy: null,
          readyHostIds: new Set([route.hostId]),
          hostLabels: new Map([[route.hostId, route.label]]),
          hostTargets: new Map([
            [route.hostId, { ...route.target, kind: route.kind, instanceId: "server-A" }],
          ]),
        },
      ),
    ).toEqual([]);
  });
});
