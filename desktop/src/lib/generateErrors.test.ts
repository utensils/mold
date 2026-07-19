import { describe, expect, it } from "vitest";
import { isMissingModelError } from "./generateErrors";

describe("isMissingModelError", () => {
  it("matches the SSE 404 and server unknown-model shapes", () => {
    expect(isMissingModelError("SSE request failed with HTTP 404")).toBe(true);
    expect(isMissingModelError("model 'jibmix-flux:fp8' not found")).toBe(true);
  });

  it("leaves other failures alone", () => {
    expect(isMissingModelError("SSE request failed with HTTP 500")).toBe(false);
    expect(isMissingModelError("Cancelled")).toBe(false);
    expect(isMissingModelError(null)).toBe(false);
    expect(isMissingModelError("HTTP 4040")).toBe(false);
  });
});
