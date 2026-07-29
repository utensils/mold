import { describe, expect, it } from "vitest";
import { ApiError } from "./client";
import { describeTransportError, isTransportFailure } from "./errors";

describe("isTransportFailure", () => {
  it("recognizes fetch TypeErrors regardless of engine message", () => {
    expect(isTransportFailure(new TypeError("Load failed"))).toBe(true);
    expect(isTransportFailure(new TypeError("Failed to fetch"))).toBe(true);
    expect(isTransportFailure(new TypeError("anything the engine says"))).toBe(true);
  });

  it("recognizes known transport strings on plain errors and strings", () => {
    expect(isTransportFailure(new Error("The network connection was lost."))).toBe(true);
    expect(isTransportFailure("Load failed")).toBe(true);
    expect(isTransportFailure("The request timed out.")).toBe(true);
    expect(isTransportFailure("A server with the specified hostname could not be found.")).toBe(
      true,
    );
    expect(isTransportFailure("The Internet connection appears to be offline.")).toBe(true);
  });

  it("never classifies HTTP replies or domain errors as transport failures", () => {
    expect(isTransportFailure(new ApiError("model not found", 404))).toBe(false);
    expect(isTransportFailure(new ApiError("", 500))).toBe(false);
    expect(isTransportFailure(new Error("host ran out of memory"))).toBe(false);
    expect(isTransportFailure("Cancelled")).toBe(false);
  });
});

describe("describeTransportError", () => {
  it("names the host when the connection itself failed", () => {
    expect(describeTransportError(new TypeError("Load failed"), "halcyon")).toBe(
      "Couldn’t reach halcyon. Check the connection and try again.",
    );
    expect(describeTransportError("The network connection was lost.", "Studio")).toBe(
      "Couldn’t reach Studio. Check the connection and try again.",
    );
  });

  it("falls back to a generic host name when none is known", () => {
    expect(describeTransportError(new TypeError("Load failed"))).toBe(
      "Couldn’t reach the host. Check the connection and try again.",
    );
    expect(describeTransportError(new TypeError("Load failed"), "  ")).toBe(
      "Couldn’t reach the host. Check the connection and try again.",
    );
  });

  it("passes server-authored ApiError detail through verbatim", () => {
    expect(
      describeTransportError(
        new ApiError("local expand model not found — run: mold pull qwen3-expand", 422),
        "Studio",
      ),
    ).toBe("local expand model not found — run: mold pull qwen3-expand");
  });

  it("gives an empty-bodied HTTP failure a visible sentence", () => {
    expect(describeTransportError(new ApiError("", 500), "Studio")).toBe(
      "Studio hit an internal error. Try again.",
    );
  });

  it("directs auth failures at the API key even when the body is empty", () => {
    expect(describeTransportError(new ApiError("", 401), "Studio")).toBe(
      "Studio didn’t accept the API key. Update it in Machines and try again.",
    );
    expect(describeTransportError(new ApiError("unauthorized", 403), "Studio")).toBe(
      "Studio didn’t accept the API key. Update it in Machines and try again.",
    );
  });

  it("maps SSE open failures by their embedded HTTP status", () => {
    expect(describeTransportError(new Error("SSE request failed with HTTP 401"), "Studio")).toBe(
      "Studio didn’t accept the API key. Update it in Machines and try again.",
    );
    expect(describeTransportError(new Error("SSE request failed with HTTP 503"), "Studio")).toBe(
      "Studio is unavailable or still starting. Try again shortly.",
    );
  });

  it("describes malformed replies without dumping parser internals", () => {
    expect(
      describeTransportError(new SyntaxError("JSON Parse error: Unrecognized token"), "Studio"),
    ).toBe("Studio sent an unexpected reply.");
  });

  it("treats aborts as cancellation", () => {
    const abort = new Error("The operation was aborted.");
    abort.name = "AbortError";
    expect(describeTransportError(abort, "Studio")).toBe("Cancelled");
  });

  it("passes unknown error copy through and covers empty messages", () => {
    expect(describeTransportError(new Error("host ran out of memory"), "Studio")).toBe(
      "Studio ran out of memory. Try a smaller model, image size, or batch.",
    );
    expect(describeTransportError(new Error(""), "Studio")).toBe(
      "Something went wrong talking to Studio.",
    );
    expect(describeTransportError("Studio is offline", "Studio")).toBe("Studio is offline");
  });
});
