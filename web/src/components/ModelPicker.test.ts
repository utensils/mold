import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelPicker from "./ModelPicker.vue";
import type { ModelInfoExtended } from "../types";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return {
    ...actual,
    downloadsStreamUrl: vi.fn(() => "/api/downloads/stream"),
    fetchDownloads: vi.fn(async () => ({
      active: null,
      queued: [],
      history: [],
    })),
  };
});

class MockEventSource {
  constructor(_url: string) {}
  addEventListener(_type: string, _listener: EventListener) {}
  close() {}
}

vi.stubGlobal("EventSource", MockEventSource);

function model(
  name: string,
  overrides: Partial<ModelInfoExtended> = {},
): ModelInfoExtended {
  return {
    name,
    family: overrides.family ?? "flux",
    size_gb: overrides.size_gb ?? 8,
    is_loaded: false,
    last_used: null,
    hf_repo: overrides.hf_repo ?? "example/repo",
    downloaded: overrides.downloaded ?? true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 28,
    default_guidance: 3.5,
    description: overrides.description ?? "",
    ...overrides,
  };
}

function mountPicker(models: ModelInfoExtended[], modelValue = "") {
  return mount(ModelPicker, {
    props: { models, modelValue },
    global: {
      stubs: {
        RouterLink: {
          props: ["to"],
          template: '<a :href="to"><slot /></a>',
        },
      },
    },
  });
}

describe("ModelPicker", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("shows downloaded standalone generation models by default", () => {
    const w = mountPicker([
      model("flux-dev:q8", { downloaded: true }),
      model("flux-dev:q4", { downloaded: false }),
      model("real-esrgan-x4plus", { family: "upscaler", downloaded: true }),
      model("qwen3-expand", { family: "qwen3-expand", downloaded: true }),
      model("sdxl-canny-controlnet", {
        family: "controlnet",
        downloaded: true,
      }),
    ]);

    expect(w.text()).toContain("flux-dev:q8");
    expect(w.text()).not.toContain("flux-dev:q4");
    expect(w.text()).not.toContain("real-esrgan");
    expect(w.text()).not.toContain("qwen3-expand");
    expect(w.text()).not.toContain("controlnet");
    expect(w.find("#mold-show-all-models").exists()).toBe(false);
    expect(
      w.findAll("button").some((button) => button.text() === "Download"),
    ).toBe(false);
  });

  it("shows a compact catalog empty state when no generation model is downloaded", () => {
    const w = mountPicker([
      model("flux-dev:q8", { downloaded: false }),
      model("real-esrgan-x4plus", { family: "upscaler", downloaded: true }),
    ]);

    expect(w.find("[data-test='model-picker-empty']").exists()).toBe(true);
    expect(w.find("[data-test='model-catalog-link']").attributes("href")).toBe(
      "/catalog",
    );
  });

  it("groups models by family and persists collapsed family state", async () => {
    const w = mountPicker([
      model("flux-dev:q8", { family: "flux" }),
      model("sdxl:fp16", { family: "sdxl" }),
    ]);

    expect(w.text()).toContain("FLUX");
    expect(w.text()).toContain("SDXL");

    await w.find("[data-test='family-toggle-flux']").trigger("click");
    expect(w.text()).not.toContain("flux-dev:q8");
    expect(
      localStorage.getItem("mold.generate.modelPicker.collapsedFamilies"),
    ).toBe(JSON.stringify(["flux"]));
  });

  it("preserves select behavior for downloaded models", async () => {
    const selected = model("flux-dev:q8", { downloaded: true });
    const w = mountPicker([selected]);

    await w.find("[data-test='model-option-flux-dev:q8']").trigger("click");

    expect(w.emitted("update:modelValue")?.[0]).toEqual(["flux-dev:q8"]);
    expect(w.emitted("select")?.[0]).toEqual([selected]);
  });

  it("applies compact boolean filters and multi-sort controls", async () => {
    const w = mountPicker([
      model("flux-schnell:q4", { family: "flux", size_gb: 4 }),
      model("flux-dev:q8", { family: "flux", size_gb: 8 }),
      model("sdxl:fp16", { family: "sdxl", size_gb: 7 }),
    ]);

    await w.find("[data-test='filter-family']").setValue("flux");
    await w.find("[data-test='filter-quantization']").setValue("q8");
    expect(w.text()).toContain("flux-dev:q8");
    expect(w.text()).not.toContain("flux-schnell:q4");
    expect(w.text()).not.toContain("sdxl:fp16");

    await w.find("[data-test='filter-downloaded']").setValue("any");
    await w.find("[data-test='filter-mode']").setValue("any");
    expect(w.text()).toContain("flux-schnell:q4");
    expect(w.text()).toContain("flux-dev:q8");
    expect(w.text()).not.toContain("sdxl:fp16");

    await w.find("[data-test='sort-size']").trigger("click");
    const rows = w
      .findAll("[data-test='model-option']")
      .map((row) => row.text());
    expect(rows[0]).toContain("flux-schnell:q4");
  });
});
