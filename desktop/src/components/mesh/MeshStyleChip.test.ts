import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

// The chip wraps Generate's own `ModelPicker`, which reaches the host stores
// and the router; this file is about the chip's trigger, not the menu.
vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));

import MeshStyleChip from "./MeshStyleChip.vue";
import type { ModelEntry } from "../../lib/api/types";

const entry = (over: Partial<ModelEntry> = {}) =>
  ({
    name: "hunyuan3d-mini-turbo:fp16",
    family: "hunyuan3d",
    downloaded: true,
    ...over,
  }) as ModelEntry;

function chip(props: Partial<Record<string, unknown>> = {}) {
  return mount(MeshStyleChip, {
    props: {
      models: [entry()],
      selected: null,
      label: "3-D style",
      placeholder: "Choose a 3-D style",
      kicker: "3-D object styles",
      browseTarget: "/models?type=mesh",
      ...props,
    },
    global: { plugins: [createPinia()], stubs: { RouterLink: true } },
  });
}

describe("MeshStyleChip", () => {
  beforeEach(() => setActivePinia(createPinia()));

  /*
   * Both defects this component has shipped came from deriving a visible
   * string from its label: `Choose a ${label.toLowerCase()}` rendered
   * "Choose a 3-d style", and the same derivation mangled the test id to
   * `--d-style`. "3-D" is the lexicon's spelling wherever a kind is named.
   */
  it("says the kind the way the lexicon spells it, with nothing picked", () => {
    const wrapper = chip();
    expect(wrapper.text()).toContain("Choose a 3-D style");
    expect(wrapper.text()).not.toContain("3-d style");
    wrapper.unmount();
  });

  it("gives the chip a test id that survives the digit in 3-D", () => {
    const wrapper = chip();
    expect(wrapper.find("[data-test='mesh-style-chip-3-d-style']").exists()).toBe(true);
    expect(wrapper.html()).not.toContain("mesh-style-chip--d-style");
    wrapper.unmount();
  });

  /* Plain name in sans, technical truth in mono, on the same row. */
  it("shows the style's plain name over its id once one is picked", () => {
    const wrapper = chip({
      selected: entry({ display_name: "Hunyuan3D mini Turbo" }),
    });
    expect(wrapper.get(".ms-chip__label").text()).toBe("Hunyuan3D mini Turbo");
    expect(wrapper.get(".ms-chip__id").text()).toBe("hunyuan3d-mini-turbo:fp16");
    wrapper.unmount();
  });

  /*
   * A display name that is just the id is not a plain name — the family's
   * friendly label stands in, exactly as `StylePicker` resolves it.
   */
  it("falls back to the family when the display name is only the id", () => {
    const wrapper = chip({ selected: entry({ display_name: "hunyuan3d-mini-turbo:fp16" }) });
    expect(wrapper.get(".ms-chip__label").text()).not.toBe("hunyuan3d-mini-turbo:fp16");
    expect(wrapper.get(".ms-chip__id").text()).toBe("hunyuan3d-mini-turbo:fp16");
    wrapper.unmount();
  });

  it("names itself for assistive tech whatever is picked", () => {
    const wrapper = chip();
    const button = wrapper.get("button");
    expect(button.attributes("aria-label")).toBe("3-D style");
    expect(button.attributes("aria-haspopup")).toBe("listbox");
    wrapper.unmount();
  });
});
