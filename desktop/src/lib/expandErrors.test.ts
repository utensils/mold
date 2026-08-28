import { describe, expect, it } from "vitest";
import { parseMissingExpandModel } from "./expandErrors";

describe("parseMissingExpandModel", () => {
  it("extracts the model from the engine's not-found message", () => {
    expect(
      parseMissingExpandModel("local expand model not found — run: mold pull qwen3-expand"),
    ).toBe("qwen3-expand");
  });

  it("handles tagged and catalog ids", () => {
    expect(
      parseMissingExpandModel(
        "local expand model 'qwen3-expand:q4' not found — run: mold pull qwen3-expand:q4",
      ),
    ).toBe("qwen3-expand:q4");
    expect(
      parseMissingExpandModel(
        "local expand model 'hf:org/repo' not found — run: mold pull hf:org/repo",
      ),
    ).toBe("hf:org/repo");
  });

  it("returns null for unrelated errors", () => {
    expect(parseMissingExpandModel("inference failed: out of memory")).toBeNull();
    expect(
      parseMissingExpandModel(
        "The model may need re-downloading: mold pull qwen3-expand",
      ),
    ).toBeNull();
  });
});
