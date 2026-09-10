import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import CreateStylePicker from "./CreateStylePicker.vue";
import type { ModelInfoExtended } from "../../types";

/*
 * Web's Create style control: the chip that names the current style, and the
 * SHARED `@studio/components/StyleMenu.vue` it opens in a popover — the same
 * list desktop's composer chip opens. It replaced a native `<select>`, which
 * could say a family name and an id and nothing else.
 */

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

const stubs = { RouterLink: { template: "<a><slot /></a>" } };

// The panel teleports to <body>; a stray one would answer the next query.
afterEach(() => {
  document.body.innerHTML = "";
});

function mountPicker(props: Record<string, unknown> = {}) {
  return mount(CreateStylePicker, {
    props: { models: [model()], model: "cv:23423432", ...props },
    global: { stubs },
    attachTo: document.body,
  });
}

const menu = () =>
  document.body.querySelector('[data-test="model-picker-menu"]');
const rows = () => [
  ...document.body.querySelectorAll('[data-test="model-option-name"]'),
];

async function open(wrapper: ReturnType<typeof mountPicker>) {
  await wrapper.get('[data-test="style-chip"]').trigger("click");
  return menu();
}

describe("CreateStylePicker chip", () => {
  it("names the style in plain words and keeps the exact id in mono", () => {
    const wrapper = mountPicker();
    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe(
      "RealVisXL V5.0 by SG161222",
    );
    expect(wrapper.get('[data-test="style-chip-id"]').text()).toBe(
      "cv:23423432",
    );
    wrapper.unmount();
  });

  it("falls back to the family's friendly label for a bare manifest id", () => {
    const wrapper = mountPicker({
      models: [model({ name: "flux-dev:q8", family: "flux", description: "" })],
      model: "flux-dev:q8",
    });
    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe(
      "FLUX",
    );
    expect(wrapper.get('[data-test="style-chip-id"]').text()).toBe(
      "flux-dev:q8",
    );
    wrapper.unmount();
  });

  it("asks for a style when nothing is chosen", () => {
    const wrapper = mountPicker({ models: [], model: "" });
    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe(
      "Choose a style",
    );
    wrapper.unmount();
  });

  it("is a chip and a listbox, never a native select", async () => {
    const wrapper = mountPicker();
    expect(wrapper.find("select").exists()).toBe(false);
    expect(
      wrapper.get('[data-test="style-chip"]').attributes("aria-expanded"),
    ).toBe("false");
    await open(wrapper);
    expect(
      wrapper.get('[data-test="style-chip"]').attributes("aria-expanded"),
    ).toBe("true");
    expect(menu()?.getAttribute("role")).toBe("listbox");
    wrapper.unmount();
  });
});

describe("CreateStylePicker menu", () => {
  it("opens the shared family-grouped list and picks a row", async () => {
    const flux = model({
      name: "flux-dev:q8",
      family: "flux",
      description: "",
    });
    const wrapper = mountPicker({ models: [model(), flux] });

    expect(menu()).toBeNull();
    await open(wrapper);
    expect(
      [...document.body.querySelectorAll(".ms-model__group")].map(
        (n) => n.textContent,
      ),
    ).toEqual(["SDXL", "FLUX"]);

    (rows()[1] as HTMLElement).closest("button")!.click();
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("select")?.[0]?.[0]).toMatchObject({
      name: "flux-dev:q8",
    });
    // Picking closes it.
    expect(menu()).toBeNull();
    wrapper.unmount();
  });

  it("keeps a downloaded-but-unrunnable style visible and refuses it by name", async () => {
    const wrapper = mountPicker({
      models: [model({ runtime_available: false })],
    });
    await open(wrapper);
    expect(
      document.body.querySelector('[data-test="model-disabled-reason"]')
        ?.textContent,
    ).toContain("Download only");
    (rows()[0] as HTMLElement).closest("button")!.click();
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("select")).toBeUndefined();
    wrapper.unmount();
  });

  it("keeps a restored style the fleet doesn't have as the phantom row", async () => {
    const wrapper = mountPicker({
      models: [],
      model: "ltx-2-19b-dev:fp8",
      missingModel: "ltx-2-19b-dev:fp8",
      emptyLabel: "No sequence models installed",
    });
    expect(wrapper.get('[data-test="style-chip-id"]').text()).toBe(
      "ltx-2-19b-dev:fp8",
    );
    await open(wrapper);
    const phantom = document.body.querySelector(
      '[data-test="model-option-missing"]',
    );
    expect(phantom?.textContent).toContain("ltx-2-19b-dev:fp8");
    expect(phantom?.textContent).toContain("Not on this machine — get it");
    wrapper.unmount();
  });

  it("says the section is empty in the page's own words", async () => {
    const wrapper = mountPicker({
      models: [],
      model: "",
      emptyLabel: "No sequence models installed",
    });
    await open(wrapper);
    expect(
      document.body
        .querySelector('[data-test="model-picker-empty"]')
        ?.textContent?.trim(),
    ).toBe("No sequence models installed");
    wrapper.unmount();
  });

  it("shows no availability tag on web, which has no rule for one yet", async () => {
    const wrapper = mountPicker();
    await open(wrapper);
    expect(
      document.body.querySelector('[data-test="model-availability"]'),
    ).toBeNull();
    wrapper.unmount();
  });

  /*
   * The popover teleports the menu to <body> and never moves focus into it,
   * and a list of eight or fewer styles carries no filter field to land on.
   * So without this a keyboard user could open the menu and then walk nowhere:
   * ↓ moved no row and Tab left the menu behind. Focus goes to the menu root
   * on open, AND the chip forwards its own keys, so both paths operate it.
   */
  it("walks and picks from the keyboard with too few styles for a filter field", async () => {
    const flux = model({
      name: "flux-dev:q8",
      family: "flux",
      description: "",
    });
    const wrapper = mountPicker({ models: [model(), flux] });

    const chip = wrapper.get('[data-test="style-chip"]');
    await chip.trigger("click");
    expect(
      document.body.querySelector('[data-test="model-filter"]'),
    ).toBeNull();
    expect(document.activeElement).toBe(menu());

    await chip.trigger("keydown", { key: "ArrowDown" });
    await chip.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("select")?.[0]?.[0]).toMatchObject({
      name: "flux-dev:q8",
    });
    wrapper.unmount();
  });

  it("walks from the menu root without the chip counting the same key twice", async () => {
    const flux = model({
      name: "flux-dev:q8",
      family: "flux",
      description: "",
    });
    const wrapper = mountPicker({ models: [model(), flux] });
    await open(wrapper);

    // Dispatched inside the menu, so it bubbles to the chip's handler too.
    const root = menu() as HTMLElement;
    for (const key of ["ArrowDown", "Enter"]) {
      root.dispatchEvent(
        new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
      );
    }
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("select")?.[0]?.[0]).toMatchObject({
      name: "flux-dev:q8",
    });
    wrapper.unmount();
  });

  it("closes on Escape and hands focus back to the chip", async () => {
    const wrapper = mountPicker();
    await open(wrapper);
    expect(menu()).not.toBeNull();

    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await wrapper.vm.$nextTick();
    expect(menu()).toBeNull();
    expect(document.activeElement).toBe(
      wrapper.get('[data-test="style-chip"]').element,
    );
    wrapper.unmount();
  });

  it("names the active row for a screen reader", async () => {
    const wrapper = mountPicker({
      models: [model(), model({ name: "flux-dev:q8", family: "flux" })],
    });
    const chip = wrapper.get('[data-test="style-chip"]');
    await chip.trigger("click");

    const root = menu() as HTMLElement;
    expect(chip.attributes("aria-controls")).toBe(root.id);
    const first = root.querySelector(".ms-model__option") as HTMLElement;
    expect(root.getAttribute("aria-activedescendant")).toBe(first.id);
    expect(first.id).not.toBe("");
    wrapper.unmount();
  });

  it("sends Browse more to the output kind's own Styles filter", async () => {
    const wrapper = mountPicker({ browseTo: "/models?type=video" });
    expect(
      wrapper.get('[data-test="browse-styles"]').attributes("href") ?? "",
    ).toBeDefined();
    await open(wrapper);
    (
      document.body.querySelector('[data-test="browse-catalog"]') as HTMLElement
    ).click();
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("browse")?.[0]).toEqual(["/models?type=video"]);
    expect(menu()).toBeNull();
    wrapper.unmount();
  });
});
