import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ShapePicker from "./ShapePicker.vue";
import { ASPECTS, swatchDims } from "../lib/resolution";

const source = readFileSync(resolve(__dirname, "./ShapePicker.vue"), "utf8");

function make(modelValue = "square", extra: Record<string, unknown> = {}) {
  return mount(ShapePicker, {
    props: { modelValue, label: "Shape", ...extra },
  });
}

describe("ShapePicker", () => {
  it("renders one radio per canonical aspect with the active one checked", () => {
    const wrapper = make("wide");
    const radios = wrapper.findAll("[role=radio]");
    expect(radios).toHaveLength(ASPECTS.length);
    const wideIndex = ASPECTS.findIndex((a) => a.id === "wide");
    expect(radios[wideIndex]!.attributes("aria-checked")).toBe("true");
    expect(radios[0]!.attributes("aria-checked")).toBe("false");
    expect(radios[wideIndex]!.attributes("data-on")).toBe("true");
  });

  it("labels the group and shows the ratio labels", () => {
    const wrapper = make();
    expect(wrapper.find("[role=radiogroup]").attributes("aria-label")).toBe(
      "Shape",
    );
    for (const aspect of ASPECTS) {
      expect(wrapper.text()).toContain(aspect.label);
    }
  });

  it("emits update:modelValue with the option id on click", async () => {
    const wrapper = make("square");
    const tallIndex = ASPECTS.findIndex((a) => a.id === "tall");
    await wrapper.findAll("button")[tallIndex]!.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([["tall"]]);
  });

  it("sizes each swatch proportionally via swatchDims, never text", () => {
    const wrapper = make();
    const swatches = wrapper.findAll(".ms-shape__swatch");
    expect(swatches).toHaveLength(ASPECTS.length);
    ASPECTS.forEach((aspect, i) => {
      const { width, height } = swatchDims(aspect.ratio);
      const style = swatches[i]!.attributes("style") ?? "";
      expect(style).toContain(`width: ${width}px`);
      expect(style).toContain(`height: ${height}px`);
      expect(swatches[i]!.text()).toBe("");
    });
  });

  it("moves selection with arrow keys and wraps", async () => {
    const last = ASPECTS[ASPECTS.length - 1]!.id;
    const wrapper = make(last);
    await wrapper.find("[role=radiogroup]").trigger("keydown", {
      key: "ArrowRight",
    });
    expect(wrapper.emitted("update:modelValue")).toEqual([[ASPECTS[0]!.id]]);
  });

  it("moves selection backwards with ArrowLeft", async () => {
    const wrapper = make(ASPECTS[0]!.id);
    await wrapper.find("[role=radiogroup]").trigger("keydown", {
      key: "ArrowLeft",
    });
    expect(wrapper.emitted("update:modelValue")).toEqual([
      [ASPECTS[ASPECTS.length - 1]!.id],
    ]);
  });

  it("only the active option is tabbable (roving tabindex)", () => {
    const wrapper = make("portrait");
    const tabs = wrapper.findAll("button").map((b) => b.attributes("tabindex"));
    const expected = ASPECTS.map((a) => (a.id === "portrait" ? "0" : "-1"));
    expect(tabs).toEqual(expected);
  });

  it("accepts custom options", () => {
    const options = [
      { id: "cine", label: "21:9", ratio: 21 / 9 },
      { id: "square", label: "1:1", ratio: 1 },
    ] as const;
    const wrapper = make("cine", { options });
    expect(wrapper.findAll("[role=radio]")).toHaveLength(2);
    expect(wrapper.text()).toContain("21:9");
  });

  it("marks the active chip approximate for a custom size from Advanced", () => {
    const wrapper = make("square", { approximate: true });
    const active = wrapper.get("[data-on='true']");
    expect(active.attributes("data-approximate")).toBe("true");
    expect(active.text()).toContain("≈");
    expect(active.attributes("title")).toContain("set in Advanced");
    // Inactive chips carry no approximation marks.
    const others = wrapper
      .findAll("[role=radio]")
      .filter((chip) => chip.attributes("data-on") !== "true");
    for (const chip of others) {
      expect(chip.attributes("data-approximate")).toBeUndefined();
      expect(chip.text()).not.toContain("≈");
    }
  });

  it("ignores interaction when disabled", async () => {
    const wrapper = make("square", { disabled: true });
    await wrapper.find("[role=radiogroup]").trigger("keydown", {
      key: "ArrowRight",
    });
    await wrapper.findAll("button")[1]!.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    expect(wrapper.find("[role=radiogroup]").attributes("aria-disabled")).toBe(
      "true",
    );
  });
});

describe("ShapePicker tile sizing (CSS pin)", () => {
  /*
   * Five 52px-wide tiles plus 7px gaps need 288px, but a 320px rail (web's
   * ControlsAside `.controls`) leaves only 282px inside its 18px padding and
   * 1px border either side — so a fixed-width flex row wrapped the 9:16 tile
   * onto a second line. A grid with `auto-fit`/`minmax` lets every tile
   * shrink to fit instead.
   */
  it("lays out .ms-shape as a grid of auto-fit, minmax(48px, 1fr) columns", () => {
    const block = source.slice(source.indexOf(".ms-shape {"), source.indexOf(".ms-shape__btn {"));
    expect(block).toMatch(/display:\s*grid/);
    expect(block).toMatch(
      /grid-template-columns:\s*repeat\(auto-fit,\s*minmax\(48px,\s*1fr\)\)/,
    );
  });

  it("gives .ms-shape__btn no fixed width, so a grid column can shrink it", () => {
    const start = source.indexOf(".ms-shape__btn {");
    const end = source.indexOf("}", start);
    const block = source.slice(start, end);
    // Excludes `max-width`, which is a legitimate cap on a grid column that
    // grows wider than the tile should ever be.
    expect(block).not.toMatch(/(?<!-)width:\s*\d/);
    expect(block).toMatch(/height:\s*60px/);
  });
});
