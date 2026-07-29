import { describe, expect, it } from "vitest";
import {
  copyableError,
  describeTransportError,
  isTransportFailure,
} from "./errors";

describe("describeTransportError", () => {
  it("turns raw SSE status failures into recovery copy", () => {
    expect(
      describeTransportError(
        new Error("SSE request failed with HTTP 404"),
        "plato",
      ),
    ).toBe(
      "plato couldn’t find that model or job. It may have been removed or the host may have restarted.",
    );
  });

  it("directs authentication failures to Machines", () => {
    expect(
      describeTransportError(
        { status: 401, message: "Unauthorized" },
        "Studio",
      ),
    ).toBe(
      "Studio didn’t accept the API key. Update it in Machines and try again.",
    );
  });

  it("uses structured server codes for queue and memory failures", () => {
    expect(
      describeTransportError(
        { status: 429, message: "full", body: { code: "QUEUE_FULL" } },
        "plato",
      ),
    ).toBe("plato is busy right now. Wait a moment and try again.");
    expect(
      describeTransportError(
        {
          status: 500,
          message: "CUDA out of memory while allocating tensor",
          body: { code: "INSUFFICIENT_MEMORY" },
        },
        "plato",
      ),
    ).toBe(
      "plato ran out of memory. Try a smaller model, image size, or batch.",
    );
  });

  it("humanizes engine network and parser messages", () => {
    expect(describeTransportError(new TypeError("Load failed"), "plato")).toBe(
      "Couldn’t reach plato. Check the connection and try again.",
    );
    expect(
      describeTransportError(new SyntaxError("Unexpected token '<'"), "plato"),
    ).toBe("plato sent an unexpected reply.");
    expect(isTransportFailure("stream closed before completion")).toBe(true);
  });

  it("keeps useful domain copy and preserves raw diagnostics for copy", () => {
    const raw = "SSE request failed with HTTP 404";
    const friendly = describeTransportError(raw, "plato");
    expect(copyableError(raw, friendly)).toBe(
      `${friendly}\n\nTechnical details: ${raw}`,
    );
    expect(
      describeTransportError("Model download was cancelled", "plato"),
    ).toBe("Model download was cancelled");
  });
});
