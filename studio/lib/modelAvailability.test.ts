import { describe, expect, it } from "vitest";
import { modelAvailabilityTag } from "./modelAvailability";

/** The web/phone shape: every machine is a peer, none is home. */
const fleet = [
  { id: "studio", label: "Studio" },
  { id: "plato", label: "plato" },
];

/** The desktop shape: the built-in engine is listed like any other machine. */
const withLocal = [
  { id: "local", label: "This Mac" },
  { id: "hal9000-7680", label: "hal9000" },
  { id: "bender-7680", label: "bender" },
];

describe("modelAvailabilityTag", () => {
  it("stays quiet when no reachable machine has the model", () => {
    expect(modelAvailabilityTag([], fleet)).toBeNull();
    expect(modelAvailabilityTag(["gone-machine"], fleet)).toBeNull();
  });

  it("stays quiet when every reachable machine has the model", () => {
    expect(modelAvailabilityTag(["studio", "plato"], fleet)).toBeNull();
    expect(modelAvailabilityTag(["local", "hal9000-7680", "bender-7680"], withLocal)).toBeNull();
  });

  it("stays quiet with a single reachable machine, which has nothing to say", () => {
    expect(modelAvailabilityTag(["studio"], [fleet[0]!])).toBeNull();
  });

  it("names the one machine that has the model", () => {
    expect(modelAvailabilityTag(["studio"], fleet)).toBe("Studio");
    expect(modelAvailabilityTag(["hal9000-7680"], withLocal)).toBe("hal9000");
  });

  it("names the built-in engine like any other machine, with no home rule", () => {
    expect(modelAvailabilityTag(["local"], withLocal)).toBe("This Mac");
  });

  it("counts machines when several but not all have the model", () => {
    expect(modelAvailabilityTag(["hal9000-7680", "bender-7680"], withLocal)).toBe("2 machines");
    expect(modelAvailabilityTag(["local", "bender-7680"], withLocal)).toBe("2 machines");
  });

  it("ignores ids the caller's reachable list does not contain", () => {
    expect(modelAvailabilityTag(["studio", "gone-machine"], fleet)).toBe("Studio");
  });

  it("never says host", () => {
    const emitted = [
      modelAvailabilityTag(["studio"], fleet),
      modelAvailabilityTag(["hal9000-7680", "bender-7680"], withLocal),
    ];
    for (const tag of emitted) expect(tag).not.toMatch(/\bhosts?\b/i);
  });
});
