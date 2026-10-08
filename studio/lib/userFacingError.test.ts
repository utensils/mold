import { describe, expect, it } from "vitest";
import fixtures from "../../docs/contracts/user-errors.json";
import { userFacingError } from "./userFacingError";

describe("cross-surface error contract", () => {
  for (const fixture of fixtures) {
    it(fixture.raw.slice(0, 90) || "empty diagnostic", () => {
      expect(userFacingError(fixture.raw)).toBe(fixture.message);
      expect(userFacingError(fixture.message)).toBe(fixture.message);
    });
  }
});
