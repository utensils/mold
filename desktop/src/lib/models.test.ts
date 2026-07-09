import { describe, expect, it } from "vitest";
import { groupInstalledModels, quantTag } from "./models";
import type { ModelEntry } from "./api/types";

function model(name: string, family: string, disk: number, loaded = false): ModelEntry {
  return {
    name,
    family,
    size_gb: disk / 1e9,
    is_loaded: loaded,
    hf_repo: "",
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
    disk_usage_bytes: disk,
  };
}

describe("quantTag", () => {
  it("returns the part after the colon, or null", () => {
    expect(quantTag("flux-dev:q8")).toBe("q8");
    expect(quantTag("sdxl:fp16")).toBe("fp16");
    expect(quantTag("real-esrgan-x4plus")).toBeNull();
  });
});

describe("groupInstalledModels", () => {
  const models = [
    model("sdxl:fp16", "sdxl", 6_900_000_000),
    model("flux-schnell:q4", "flux", 6_400_000_000),
    model("flux-dev:q8", "flux", 11_800_000_000, true),
    model("real-esrgan-x4plus:fp16", "real-esrgan", 67_000_000),
    model("qwen3-expand:q4", "qwen3-expand", 2_000_000_000),
  ];

  it("groups generation families, sorted by family then name", () => {
    const { families } = groupInstalledModels(models);
    expect(families.map(([f]) => f)).toEqual(["flux", "sdxl"]);
    expect(families[0]![1].map((m) => m.name)).toEqual(["flux-dev:q8", "flux-schnell:q4"]);
  });

  it("splits utility families out and name-sorts them", () => {
    const { utility } = groupInstalledModels(models);
    expect(utility.map((m) => m.name)).toEqual(["qwen3-expand:q4", "real-esrgan-x4plus:fp16"]);
  });

  it("reports the largest disk footprint for the usage-bar denominator", () => {
    expect(groupInstalledModels(models).maxDiskBytes).toBe(11_800_000_000);
  });

  it("handles an empty list", () => {
    expect(groupInstalledModels([])).toEqual({ families: [], utility: [], maxDiskBytes: 0 });
  });
});
