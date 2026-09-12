import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { nextTick } from "vue";
import StyleMenu from "./StyleMenu.vue";
import styleMenuSource from "./StyleMenu.vue?raw";
import type { StyleMenuModel } from "../lib/styleMenu";

/*
 * The shared style LIST — the one menu desktop, web and the phone all open.
 * The popover/sheet around it belongs to each host; everything asserted here
 * is the part that must read identically on all three.
 */

function model(overrides: Partial<StyleMenuModel> = {}): StyleMenuModel {
  return {
    name: "flux-dev:q8",
    family: "flux",
    description: "",
    disk_usage_bytes: 6_500_000_000,
    is_loaded: false,
    ...overrides,
  };
}

function many(count: number): StyleMenuModel[] {
  return Array.from({ length: count }, (_, i) =>
    model({ name: `sdxl-${i}:q8`, family: "sdxl" }),
  );
}

function mountMenu(props: Record<string, unknown> = {}) {
  return mount(StyleMenu, {
    props: { models: [model()], selected: null, ...props },
    attachTo: document.body,
  });
}

const rowText = (wrapper: ReturnType<typeof mountMenu>) =>
  wrapper.findAll("[data-test='model-option-name']").map((n) => n.text());

describe("StyleMenu rows", () => {
  it("groups rows by family in the order the host handed them", () => {
    const wrapper = mountMenu({
      models: [
        model({ name: "flux-dev:q8", family: "flux" }),
        model({ name: "z-image:bf16", family: "z-image" }),
        model({ name: "flux2-klein:q4", family: "flux2" }),
      ],
    });

    expect(wrapper.findAll(".ms-model__group").map((n) => n.text())).toEqual([
      "FLUX",
      "Z-Image",
      "Flux.2",
    ]);
    expect(
      wrapper.findAll("[data-test='model-option-id']").map((n) => n.text()),
    ).toEqual(["flux-dev:q8", "z-image:bf16", "flux2-klein:q4"]);
  });

  it("says the plain name in sans and the technical truth in mono", () => {
    const wrapper = mountMenu({
      models: [
        model({
          name: "cv:23423432",
          family: "sdxl",
          description: "RealVisXL V5.0",
          is_loaded: true,
        }),
      ],
    });

    expect(wrapper.get("[data-test='model-option-name']").text()).toBe(
      "RealVisXL V5.0",
    );
    expect(wrapper.get("[data-test='model-option-id']").text()).toBe(
      "cv:23423432",
    );
    expect(wrapper.get("[data-test='model-option-size']").text()).toBe(
      "6.5 GB",
    );
    expect(wrapper.get("[data-test='model-option-loaded']").text()).toBe(
      "on GPU",
    );
  });

  it("marks the current style", () => {
    const current = model({ name: "z-image:bf16", family: "z-image" });
    const wrapper = mountMenu({
      models: [model(), current],
      selected: current,
    });

    const marks = wrapper.findAll("[data-test='model-option-current']");
    expect(marks).toHaveLength(1);
    expect(marks[0]!.text()).toBe("✓");
  });

  /*
   * The lexicon is description-first (§2: "Photoreal — best quality ·
   * `flux-dev:q4`"; a bare id is a "never say" as the primary label). A
   * manifest style's `modelDisplayName` IS its id, so the rows used to read
   * `flux-schnell:q8` over `flux-schnell:q8 23.1 GB` — the id twice — while
   * the chip above them already said the description. One rule now:
   * `styleDisplayName`.
   */
  it("leads a manifest style with its description, not its id", () => {
    const wrapper = mountMenu({
      models: [
        model({
          name: "flux-schnell:q8",
          family: "flux",
          description: "FLUX.1 Schnell Q8 — fast 4-step, general purpose",
        }),
      ],
    });

    expect(wrapper.get("[data-test='model-option-name']").text()).toBe(
      "FLUX.1 Schnell Q8 — fast 4-step, general purpose",
    );
    expect(wrapper.get("[data-test='model-option-id']").text()).toBe(
      "flux-schnell:q8",
    );
    expect(
      wrapper.find("[data-test='model-option-description']").exists(),
    ).toBe(false);
  });

  it("falls back to the family's own name when a style describes itself with nothing", () => {
    const wrapper = mountMenu({
      models: [
        model({ name: "wan22-ti2v-5b:fp16", family: "wan", description: "" }),
      ],
    });
    expect(wrapper.get("[data-test='model-option-name']").text()).toBe(
      "Wan Video",
    );
    expect(wrapper.get("[data-test='model-option-id']").text()).toBe(
      "wan22-ti2v-5b:fp16",
    );
  });

  /*
   * `styleDisplayName` ranks the description ABOVE `display_name`, so a
   * catalog row carrying both would silently lose its curated name. The
   * second line is where that name goes — never a repeat of the title or of
   * the id beneath it.
   */
  it("keeps the curated name the description outranked, and never repeats one", () => {
    const wrapper = mountMenu({
      models: [
        model({
          name: "cv:1",
          family: "sdxl",
          display_name: "Studio style",
          description: "RealVisXL V5.0 by SG161222",
        }),
      ],
    });

    expect(wrapper.get("[data-test='model-option-name']").text()).toBe(
      "RealVisXL V5.0 by SG161222",
    );
    expect(wrapper.get("[data-test='model-option-description']").text()).toBe(
      "Studio style",
    );
  });

  it("says nothing on the second line when the row has one name", () => {
    const wrapper = mountMenu({
      models: [
        model({
          name: "flux-dev:q8",
          family: "flux",
          description: "Detailed still images",
        }),
        model({ name: "cv:1", family: "sdxl", description: "RealVisXL V5.0" }),
      ],
    });

    expect(wrapper.findAll("[data-test='model-option-description']")).toEqual(
      [],
    );
    expect(
      wrapper.findAll("[data-test='model-option-name']").map((n) => n.text()),
    ).toEqual(["Detailed still images", "RealVisXL V5.0"]);
  });

  it("renders the host's glyph slot beside every row", () => {
    const wrapper = mount(StyleMenu, {
      props: { models: [model()], selected: null },
      slots: { glyph: '<i class="glyph-stub" />' },
    });
    expect(wrapper.findAll(".glyph-stub")).toHaveLength(1);
  });

  it("draws a source glyph per row by default, following modelSource", () => {
    const wrapper = mountMenu({
      models: [
        model({ name: "cv:8001", family: "sdxl" }),
        model({
          name: "flux-dev:q8",
          family: "flux",
          hf_repo: "black-forest-labs/FLUX.1-dev",
        }),
        model({ name: "my-lora.safetensors", family: "sdxl" }),
      ],
    });
    const byId = new Map(
      wrapper
        .findAll("[data-test='model-option-id']")
        .map((idNode, i) => [
          idNode.text(),
          wrapper
            .findAll("[data-test='model-option-name']")
            [i]!.element.closest(".ms-model__option")!
            .querySelector("svg")!
            .getAttribute("data-source"),
        ]),
    );
    expect(byId.get("cv:8001")).toBe("civitai");
    expect(byId.get("flux-dev:q8")).toBe("hf");
    expect(byId.get("my-lora.safetensors")).toBe("local");
  });

  it("draws no default glyph when the host fills the slot itself", () => {
    const wrapper = mount(StyleMenu, {
      props: { models: [model()], selected: null },
      slots: { glyph: '<i class="glyph-stub" />' },
    });
    expect(wrapper.find("svg[data-source]").exists()).toBe(false);
  });
});

describe("StyleMenu filter", () => {
  it("offers the filter field only once the list passes eight rows", () => {
    expect(
      mountMenu({ models: many(8) })
        .find("[data-test='model-filter']")
        .exists(),
    ).toBe(false);
    expect(
      mountMenu({ models: many(9) })
        .find("[data-test='model-filter']")
        .exists(),
    ).toBe(true);
  });

  /* Everything a person might have read on the row, or remembered from the
   * command line, reaches the filter — the title, the id, and the family. */
  it("narrows on the id, the description, the plain name and the family label", async () => {
    const wrapper = mountMenu({
      models: [
        ...many(8),
        model({
          name: "flux-dev:q8",
          family: "flux",
          description: "Photoreal — best quality",
        }),
        model({ name: "wan22-ti2v-5b:fp16", family: "wan", description: "" }),
      ],
    });

    await wrapper.get("[data-test='model-filter']").setValue("wan video");
    expect(rowText(wrapper)).toEqual(["Wan Video"]);

    await wrapper.get("[data-test='model-filter']").setValue("flux-dev");
    expect(rowText(wrapper)).toEqual(["Photoreal — best quality"]);

    await wrapper.get("[data-test='model-filter']").setValue("photoreal");
    expect(rowText(wrapper)).toEqual(["Photoreal — best quality"]);
  });

  it("says a filter matched nothing in different words from an empty section", async () => {
    const empty = mountMenu({
      models: [],
      emptyLabel: "No clip styles are ready.",
    });
    expect(empty.get("[data-test='model-picker-empty']").text()).toBe(
      "No clip styles are ready.",
    );

    const filtered = mountMenu({
      models: many(9),
      emptyLabel: "No clip styles are ready.",
    });
    await filtered.get("[data-test='model-filter']").setValue("nothing");
    expect(filtered.get("[data-test='model-picker-empty']").text()).toBe(
      "No style matches “nothing”.",
    );
  });

  it("focuses the filter when the host asks it to", async () => {
    const wrapper = mountMenu({ models: many(9), autofocusFilter: true });
    await nextTick();
    expect(document.activeElement).toBe(
      wrapper.get("[data-test='model-filter']").element,
    );

    const quiet = mountMenu({ models: many(9) });
    await nextTick();
    expect(document.activeElement).not.toBe(
      quiet.get("[data-test='model-filter']").element,
    );
  });
});

describe("StyleMenu keyboard", () => {
  const walk = async (wrapper: ReturnType<typeof mountMenu>, key: string) =>
    wrapper.get("[data-test='model-picker-menu']").trigger("keydown", { key });

  it("wraps ↑/↓ around the list and picks the active row with Enter", async () => {
    const models = [
      model({ name: "a:q8", family: "flux" }),
      model({ name: "b:q8", family: "flux" }),
    ];
    const wrapper = mountMenu({ models });

    await walk(wrapper, "ArrowDown");
    await walk(wrapper, "Enter");
    expect(wrapper.emitted("pick")?.[0]?.[0]).toMatchObject({ name: "b:q8" });

    await walk(wrapper, "ArrowDown");
    await walk(wrapper, "Enter");
    expect(wrapper.emitted("pick")?.[1]?.[0]).toMatchObject({ name: "a:q8" });

    await walk(wrapper, "ArrowUp");
    await walk(wrapper, "Enter");
    expect(wrapper.emitted("pick")?.[2]?.[0]).toMatchObject({ name: "b:q8" });
  });

  it("opens on the selected row so Enter re-picks it", async () => {
    const models = [
      model({ name: "a:q8", family: "flux" }),
      model({ name: "b:q8", family: "flux" }),
    ];
    const wrapper = mountMenu({ models, selected: models[1] });

    await walk(wrapper, "Enter");
    expect(wrapper.emitted("pick")?.[0]?.[0]).toMatchObject({ name: "b:q8" });
  });

  it("leaves Escape to the host", async () => {
    const wrapper = mountMenu();
    const event = new KeyboardEvent("keydown", {
      key: "Escape",
      cancelable: true,
    });
    wrapper.get("[data-test='model-picker-menu']").element.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });
});

describe("StyleMenu phantom row", () => {
  it("stands at index 0 for a style no machine has, and picking it asks for the pull", async () => {
    const wrapper = mountMenu({
      models: [model()],
      missingModel: "wan22-ti2v-5b:fp16",
    });

    const phantom = wrapper.get("[data-test='model-option-missing']");
    expect(phantom.text()).toContain("wan22-ti2v-5b:fp16");
    expect(phantom.text()).toContain("Not on this machine — get it");
    // Row 0 is the phantom: Enter without moving takes it.
    await wrapper
      .get("[data-test='model-picker-menu']")
      .trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("pick-missing")?.[0]).toEqual([
      "wan22-ti2v-5b:fp16",
    ]);
  });

  it("lines its name up with every other row's, which now indents for a source glyph", () => {
    const wrapper = mountMenu({
      models: [model()],
      missingModel: "wan22-ti2v-5b:fp16",
    });
    const phantom = wrapper.get("[data-test='model-option-missing']");
    // A blank, same-width stand-in for the glyph every real row draws by
    // default — never a visible mark, since nothing is known to point it at.
    const spacer = phantom.get(".ms-model__glyph--spacer");
    expect(spacer.attributes("aria-hidden")).toBe("true");
    expect(spacer.find("svg").exists()).toBe(false);
  });

  it("is suppressed while filtering and when a real style is selected", async () => {
    const selected = mountMenu({
      models: [model()],
      selected: model(),
      missingModel: "wan22-ti2v-5b:fp16",
    });
    expect(selected.find("[data-test='model-option-missing']").exists()).toBe(
      false,
    );

    const wrapper = mountMenu({
      models: many(9),
      missingModel: "wan22-ti2v-5b:fp16",
    });
    expect(wrapper.find("[data-test='model-option-missing']").exists()).toBe(
      true,
    );
    await wrapper.get("[data-test='model-filter']").setValue("sdxl");
    expect(wrapper.find("[data-test='model-option-missing']").exists()).toBe(
      false,
    );
  });
});

describe("StyleMenu refusals and tags", () => {
  it("says why a row refuses and does not pick it", async () => {
    const wrapper = mountMenu({
      models: [model()],
      disabledReason: (m: StyleMenuModel) =>
        m.name === "flux-dev:q8" ? "Download only" : null,
    });

    expect(wrapper.get("[data-test='model-disabled-reason']").text()).toBe(
      "Download only",
    );
    expect(
      wrapper.get(".ms-model__option").attributes("disabled"),
    ).toBeDefined();
    await wrapper.get(".ms-model__option").trigger("click");
    expect(wrapper.emitted("pick")).toBeUndefined();
  });

  it("renders an availability tag when the host supplies one and nothing when it is null", () => {
    expect(
      mountMenu({ availabilityTag: () => "2 machines" })
        .get("[data-test='model-availability']")
        .text(),
    ).toBe("2 machines");
    expect(
      mountMenu({ availabilityTag: () => null })
        .find("[data-test='model-availability']")
        .exists(),
    ).toBe(false);
    expect(mountMenu().find("[data-test='model-availability']").exists()).toBe(
      false,
    );
  });

  it("names what the menu holds and ends in Browse more", async () => {
    const wrapper = mountMenu({ kicker: "still picture styles" });
    expect(wrapper.get("[data-test='model-picker-kicker']").text()).toBe(
      "still picture styles",
    );

    await wrapper.get("[data-test='browse-catalog']").trigger("click");
    expect(wrapper.emitted("browse")).toHaveLength(1);
    expect(wrapper.get("[data-test='browse-catalog']").text()).toBe(
      "Browse more →",
    );
  });
});

describe("StyleMenu touch sizing", () => {
  const block = (selector: string) => {
    const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return (
      styleMenuSource.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`))?.[1] ??
      ""
    );
  };

  it("marks the menu so a touch host gets finger-sized rows", () => {
    expect(
      mountMenu({ touch: true })
        .get("[data-test='model-picker-menu']")
        .classes(),
    ).toContain("ms-model__menu--touch");
    expect(
      mountMenu().get("[data-test='model-picker-menu']").classes(),
    ).not.toContain("ms-model__menu--touch");
  });

  it("gives those rows the 44px target and 16px text iOS asks for", () => {
    const row = block(".ms-model__menu--touch .ms-model__option");
    expect(row).toMatch(/min-height:\s*44px\s*;/);
    expect(row).toMatch(/font-size:\s*var\(--mold-fs-md[^)]*\)\s*;/);
  });

  it("keeps every clickable row on the hand cursor", () => {
    expect(block(".ms-model__option")).toContain("cursor: pointer");
    expect(block(".ms-model__browse")).toContain("cursor: pointer");
  });
});
