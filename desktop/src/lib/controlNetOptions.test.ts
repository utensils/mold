import { describe, expect, it } from "vitest";
import { buildControlNetOptions } from "./controlNetOptions";
import type { CatalogEntry, ModelEntry } from "./api/types";

function model(part: Partial<ModelEntry>): ModelEntry {
  return {
    name: "controlnet-canny-sd15",
    family: "controlnet",
    downloaded: true,
    ...part,
  } as ModelEntry;
}

function entry(part: Partial<CatalogEntry>): CatalogEntry {
  return {
    id: "cv:1",
    source: "civitai",
    name: "cv-control",
    family: "controlnet",
    kind: "control-net",
    nsfw: false,
    installed: true,
    primary_path: "/models/cv-control.safetensors",
    ...part,
  } as CatalogEntry;
}

describe("buildControlNetOptions", () => {
  it("lists installed controlnet-family models, downloaded first then by name", () => {
    const options = buildControlNetOptions(
      [
        model({ name: "zeta-missing", downloaded: false }),
        model({ name: "controlnet-openpose-sd15", downloaded: true }),
        model({ name: "controlnet-canny-sd15", downloaded: true }),
      ],
      [],
      "",
    );
    expect(options.map((o) => o.value)).toEqual([
      "controlnet-canny-sd15",
      "controlnet-openpose-sd15",
      "zeta-missing",
    ]);
  });

  it("ignores models from other families", () => {
    const options = buildControlNetOptions(
      [model({ name: "flux-dev:q8", family: "flux" }), model({ name: "cn", family: "controlnet" })],
      [],
      "",
    );
    expect(options.map((o) => o.value)).toEqual(["cn"]);
  });

  it("disables not-downloaded models and marks them missing (web parity)", () => {
    const options = buildControlNetOptions([model({ name: "cn", downloaded: false })], [], "");
    expect(options).toEqual([{ value: "cn", label: "cn (missing)", disabled: true }]);
  });

  it("appends installed catalog control-net entries valued by primary_path", () => {
    const options = buildControlNetOptions(
      [model({ name: "builtin-cn" })],
      [entry({ name: "cv-control", primary_path: "/models/cv-control.safetensors" })],
      "",
    );
    expect(options).toEqual([
      { value: "builtin-cn", label: "builtin-cn", disabled: false },
      { value: "/models/cv-control.safetensors", label: "cv-control", disabled: false },
    ]);
  });

  it("skips catalog entries that are not installed or lack a primary path", () => {
    const options = buildControlNetOptions(
      [],
      [
        entry({ id: "cv:1", name: "not-installed", installed: false }),
        entry({ id: "cv:2", name: "no-path", primary_path: null }),
      ],
      "",
    );
    expect(options).toEqual([]);
  });

  it("dedupes catalog entries that collide with a built-in model name or value", () => {
    const options = buildControlNetOptions(
      [model({ name: "controlnet-canny-sd15" })],
      [
        entry({ id: "cv:1", name: "controlnet-canny-sd15", primary_path: "/m/canny.safetensors" }),
        entry({ id: "cv:2", name: "dup-value", primary_path: "/m/pose.safetensors" }),
        entry({ id: "cv:3", name: "dup-value-2", primary_path: "/m/pose.safetensors" }),
      ],
      "",
    );
    expect(options.map((o) => o.value)).toEqual(["controlnet-canny-sd15", "/m/pose.safetensors"]);
  });

  it("keeps an unknown current value selectable (web parity trailing option)", () => {
    const options = buildControlNetOptions([model({ name: "cn" })], [], "hand-typed-model");
    expect(options.at(-1)).toEqual({
      value: "hand-typed-model",
      label: "hand-typed-model",
      disabled: false,
    });
  });

  it("does not duplicate a current value already covered by an option", () => {
    const options = buildControlNetOptions([model({ name: "cn" })], [], "cn");
    expect(options.map((o) => o.value)).toEqual(["cn"]);
  });

  it("adds no trailing option for an empty current value", () => {
    expect(buildControlNetOptions([], [], "")).toEqual([]);
    expect(buildControlNetOptions([], [], "   ")).toEqual([]);
  });
});
