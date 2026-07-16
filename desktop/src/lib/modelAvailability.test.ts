import { describe, expect, it } from "vitest";
import { modelAvailabilityFromQuery } from "./modelAvailability";

describe("modelAvailabilityFromQuery", () => {
  it("keeps the legacy catalog deep link focused on available models", () => {
    expect(modelAvailabilityFromQuery({ tab: "catalog" })).toBe("available");
  });

  it("accepts the explicit availability query", () => {
    expect(modelAvailabilityFromQuery({ availability: "installed" })).toBe("installed");
  });

  it("defaults unknown queries to the combined library", () => {
    expect(modelAvailabilityFromQuery({ tab: "wat" })).toBe("all");
  });
});
