import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import type { ModelInfoExtended } from "../../types";
import InstalledModelRow from "./InstalledModelRow.vue";

function makeModel(over: Partial<ModelInfoExtended> = {}): ModelInfoExtended {
  return {
    name: "flux-schnell:q8",
    family: "flux",
    size_gb: 12.3,
    is_loaded: false,
    last_used: null,
    hf_repo: "black-forest-labs/FLUX.1-schnell",
    downloaded: true,
    default_steps: 4,
    default_guidance: 0,
    default_width: 1024,
    default_height: 1024,
    description: "",
    ...over,
  };
}

describe("InstalledModelRow", () => {
  it("renders the model name and family · size", () => {
    const w = mount(InstalledModelRow, { props: { model: makeModel() } });
    expect(w.text()).toContain("flux-schnell:q8");
    expect(w.text()).toContain("flux");
    expect(w.text()).toContain("12.3 GB");
  });

  it("shows the ★ loaded badge only when the model is loaded", () => {
    const loaded = mount(InstalledModelRow, {
      props: { model: makeModel({ is_loaded: true }) },
    });
    expect(loaded.find("[data-test=loaded-badge]").exists()).toBe(true);
    expect(loaded.text()).toMatch(/★ loaded/);

    const idle = mount(InstalledModelRow, {
      props: { model: makeModel({ is_loaded: false }) },
    });
    expect(idle.find("[data-test=loaded-badge]").exists()).toBe(false);
  });

  it("emits open when the row is clicked", async () => {
    const w = mount(InstalledModelRow, { props: { model: makeModel() } });
    await w.find("[data-test=installed-row]").trigger("click");
    expect(w.emitted("open")).toBeTruthy();
  });
});
