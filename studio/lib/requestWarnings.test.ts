import { describe, expect, it } from "vitest";
import { requestWarningsFromHeaders } from "./requestWarnings";

describe("requestWarningsFromHeaders", () => {
  it("keeps a semicolon inside one advisory", () => {
    const headers = new Headers({
      "x-mold-request-warning":
        "Tags were ignored; the gallery database is disabled.",
    });
    expect(requestWarningsFromHeaders(headers)).toEqual([
      "Tags were ignored; the gallery database is disabled.",
    ]);
  });

  it("preserves separate values when the runtime exposes getAll", () => {
    expect(
      requestWarningsFromHeaders({
        get: () => "combined fallback",
        getAll: () => ["first advisory", "second advisory"],
      }),
    ).toEqual(["first advisory", "second advisory"]);
  });

  it("falls back when a browser-compatible getAll rejects ordinary headers", () => {
    expect(
      requestWarningsFromHeaders({
        get: () => "one combined advisory",
        getAll: () => {
          throw new TypeError('Only "set-cookie" is supported.');
        },
      }),
    ).toEqual(["one combined advisory"]);
  });
});
