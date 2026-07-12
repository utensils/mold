import { describe, expect, it } from "vitest";
import { modelSource } from "./modelSource";

describe("modelSource", () => {
  it("maps catalog prefixes and hf_repo to their source", () => {
    expect(modelSource({ name: "cv:2319074" })).toBe("civitai");
    expect(modelSource({ name: "hf:Qwen/Qwen-Image" })).toBe("hf");
    expect(modelSource({ name: "flux-dev:q8", hf_repo: "black-forest-labs/FLUX.1-dev" })).toBe(
      "hf",
    );
    expect(modelSource({ name: "my-merge", hf_repo: null })).toBe("local");
    expect(modelSource({ name: "my-merge" })).toBe("local");
  });
});
