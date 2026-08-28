import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_EXPAND_MODEL,
  expandModelId,
  expansionPolicyForSelection,
  parseMissingExpandModel,
  resolveExpansionRoute,
  type ExpansionCandidate,
} from "./expansionRouting";

function candidate(
  hostId: string,
  overrides: Partial<ExpansionCandidate> = {},
): ExpansionCandidate {
  return { hostId, ready: true, ...overrides };
}

/** Deterministic stand-in for `pickAutoHost` / `pickMostCapableHost`. */
const firstAlphabetically = (ids: readonly string[]) =>
  [...ids].sort()[0] ?? null;

describe("resolveExpansionRoute", () => {
  it("keeps the generation route when that host has the expand model", () => {
    const decision = resolveExpansionRoute(
      { kind: "auto" },
      { hostId: "mac" },
      [
        candidate("mac", { modelPresent: true }),
        candidate("plato", { modelPresent: true }),
      ],
      firstAlphabetically,
    );
    expect(decision).toEqual({ kind: "generation" });
  });

  it("keeps the generation route when the host's expand capability is unknown", () => {
    const decision = resolveExpansionRoute(
      { kind: "auto" },
      { hostId: "mac" },
      [
        candidate("mac", { modelPresent: null }),
        candidate("plato", { modelPresent: true }),
      ],
      firstAlphabetically,
    );
    expect(decision).toEqual({ kind: "generation" });
  });

  it("keeps the generation route for a host that was never read at all", () => {
    const decision = resolveExpansionRoute(
      { kind: "auto" },
      { hostId: "unknown-host" },
      [candidate("plato", { modelPresent: true })],
      firstAlphabetically,
    );
    expect(decision).toEqual({ kind: "generation" });
  });

  it("reroutes to a ranked host that has the model when the generation host does not", () => {
    const rank = vi.fn(firstAlphabetically);
    const decision = resolveExpansionRoute(
      { kind: "auto" },
      { hostId: "mac" },
      [
        candidate("mac", { modelPresent: false }),
        candidate("plato", { modelPresent: true }),
        candidate("hal9000", { modelPresent: true }),
      ],
      rank,
    );
    expect(decision).toEqual({ kind: "reroute", hostId: "hal9000" });
    expect(rank).toHaveBeenCalledWith(["plato", "hal9000"]);
  });

  it("uses the same ranking under Most capable", () => {
    const decision = resolveExpansionRoute(
      { kind: "capable" },
      { hostId: "mac" },
      [
        candidate("mac", { modelPresent: false }),
        candidate("plato", { modelPresent: true }),
        candidate("hal9000", { modelPresent: true }),
      ],
      (ids) => (ids.includes("plato") ? "plato" : null),
    );
    expect(decision).toEqual({ kind: "reroute", hostId: "plato" });
  });

  it("never leaves a pinned host", () => {
    const rank = vi.fn(firstAlphabetically);
    const decision = resolveExpansionRoute(
      { kind: "pinned", hostId: "mac" },
      { hostId: "mac" },
      [
        candidate("mac", { modelPresent: false }),
        candidate("plato", { modelPresent: true }),
      ],
      rank,
    );
    expect(decision).toEqual({ kind: "missing" });
    expect(rank).not.toHaveBeenCalled();
  });

  it("ignores hosts that are unreachable or have expansion unconfigured", () => {
    const decision = resolveExpansionRoute(
      { kind: "auto" },
      { hostId: "mac" },
      [
        candidate("mac", { modelPresent: false }),
        candidate("down", { modelPresent: true, ready: false }),
        candidate("unconfigured", { modelPresent: true, configured: false }),
      ],
      firstAlphabetically,
    );
    expect(decision).toEqual({ kind: "missing" });
  });

  it("does not treat an unread peer as a candidate that has the model", () => {
    const decision = resolveExpansionRoute(
      { kind: "auto" },
      { hostId: "mac" },
      [candidate("mac", { modelPresent: false }), candidate("plato")],
      firstAlphabetically,
    );
    expect(decision).toEqual({ kind: "missing" });
  });

  it("reports the generation route when the ranker picks it back", () => {
    const decision = resolveExpansionRoute(
      { kind: "auto" },
      { hostId: "mac" },
      [
        candidate("mac", { modelPresent: false }),
        candidate("plato", { modelPresent: true }),
      ],
      () => "mac",
    );
    expect(decision).toEqual({ kind: "generation" });
  });

  it("routes expansion even when no generation route resolved", () => {
    const decision = resolveExpansionRoute(
      { kind: "auto" },
      null,
      [candidate("plato", { modelPresent: true })],
      firstAlphabetically,
    );
    expect(decision).toEqual({ kind: "reroute", hostId: "plato" });
  });

  it("is missing when nothing is routable at all", () => {
    expect(
      resolveExpansionRoute({ kind: "auto" }, null, [], firstAlphabetically),
    ).toEqual({
      kind: "missing",
    });
  });
});

describe("expansionPolicyForSelection", () => {
  it("maps the desktop pref shape", () => {
    expect(expansionPolicyForSelection(null)).toEqual({ kind: "auto" });
    expect(expansionPolicyForSelection("capable")).toEqual({ kind: "capable" });
    expect(expansionPolicyForSelection("plato")).toEqual({
      kind: "pinned",
      hostId: "plato",
    });
  });

  it("maps the web sentinel shape", () => {
    const sentinels = { auto: "auto", capable: "capable" };
    expect(expansionPolicyForSelection("auto", sentinels)).toEqual({
      kind: "auto",
    });
    expect(expansionPolicyForSelection("capable", sentinels)).toEqual({
      kind: "capable",
    });
    expect(expansionPolicyForSelection("plato", sentinels)).toEqual({
      kind: "pinned",
      hostId: "plato",
    });
  });
});

describe("parseMissingExpandModel", () => {
  it("extracts legacy and model-qualified missing-model errors", () => {
    expect(
      parseMissingExpandModel(
        "local expand model not found — run: mold pull qwen3-expand",
      ),
    ).toBe("qwen3-expand");
    expect(
      parseMissingExpandModel(
        "local expand model 'qwen3-expand:q8' not found — run: mold pull qwen3-expand:q8",
      ),
    ).toBe("qwen3-expand:q8");
  });

  it("does not turn output-count recovery advice into a missing-model state", () => {
    expect(
      parseMissingExpandModel(
        "expected exactly 10 distinct non-empty prompts, but the expansion backend returned 9. " +
          "The model may need re-downloading: mold pull qwen3-expand",
      ),
    ).toBeNull();
  });
});

describe("expandModelId", () => {
  it("prefers the model the host names", () => {
    expect(expandModelId({ model: "qwen3-expand:q8" })).toBe("qwen3-expand:q8");
  });

  it("falls back to the manifest default on servers that name none", () => {
    expect(expandModelId(null)).toBe(DEFAULT_EXPAND_MODEL);
    expect(expandModelId({})).toBe(DEFAULT_EXPAND_MODEL);
    expect(expandModelId({ model: "  " })).toBe(DEFAULT_EXPAND_MODEL);
  });
});
