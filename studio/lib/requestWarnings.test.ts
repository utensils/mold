import { describe, expect, it } from "vitest";
import {
  requestWarningsFromCompleteEvent,
  requestWarningsFromHeaders,
} from "./requestWarnings";

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

describe("requestWarningsFromCompleteEvent", () => {
  it("reads the advisories a streaming render produced", () => {
    expect(
      requestWarningsFromCompleteEvent({
        image: "",
        request_warnings: [
          "3 faces were detected in the identity image; conditioning on the largest one",
        ],
      }),
    ).toEqual([
      "3 faces were detected in the identity image; conditioning on the largest one",
    ]);
  });

  it("is silent on an ordinary render and on an older server", () => {
    expect(requestWarningsFromCompleteEvent({ image: "" })).toEqual([]);
    expect(requestWarningsFromCompleteEvent({ request_warnings: [] })).toEqual(
      [],
    );
  });

  // The field is additive, so a print must never be lost to a shape surprise.
  it("never throws on a malformed payload", () => {
    expect(requestWarningsFromCompleteEvent(null)).toEqual([]);
    expect(requestWarningsFromCompleteEvent("complete")).toEqual([]);
    expect(
      requestWarningsFromCompleteEvent({ request_warnings: "one" }),
    ).toEqual([]);
    expect(
      requestWarningsFromCompleteEvent({
        request_warnings: [1, "  ", " kept "],
      }),
    ).toEqual(["kept"]);
  });
});
