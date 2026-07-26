import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import CreateModelPicker from "./CreateModelPicker.vue";
import type { ModelInfoExtended } from "../../types";

function model(overrides: Partial<ModelInfoExtended> = {}): ModelInfoExtended {
  return {
    name: "cv:23423432",
    family: "sdxl",
    size_gb: 6.5,
    is_loaded: false,
    last_used: null,
    hf_repo: "",
    default_steps: 25,
    default_guidance: 5,
    default_width: 1024,
    default_height: 1024,
    description: "RealVisXL V5.0 by SG161222",
    downloaded: true,
    ...overrides,
  };
}

describe("CreateModelPicker", () => {
  it("shows a human-readable catalog name while preserving the id as its value", () => {
    const wrapper = mount(CreateModelPicker, {
      props: { models: [model()], model: "cv:23423432" },
      global: { stubs: { RouterLink: { template: "<a><slot /></a>" } } },
    });

    const option = wrapper.get("option");
    expect(option.text()).toBe("RealVisXL V5.0 by SG161222");
    expect(option.attributes("value")).toBe("cv:23423432");
  });

  it("selects the empty-state option when a restored model is unavailable", () => {
    const wrapper = mount(CreateModelPicker, {
      props: {
        models: [],
        model: "ltx-2-19b-dev:fp8",
        emptyLabel: "No sequence models installed",
      },
      global: { stubs: { RouterLink: { template: "<a><slot /></a>" } } },
    });

    const select = wrapper.get("select").element as HTMLSelectElement;
    expect(select.value).toBe("");
    expect(wrapper.get("option").text()).toBe("No sequence models installed");
  });
});
