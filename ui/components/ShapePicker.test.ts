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
   * onto a second line. A grid lets every tile shrink to fit instead.
   *
   * `auto-fill`, not `auto-fit`: `auto-fit` COLLAPSES empty tracks and hands
   * their width to the occupied ones, so at a wider container (e.g. the
   * desktop inspector) five tiles grow well past the mock's 52px with an
   * uneven gap. `auto-fill` keeps every track — occupied or not — at the
   * `minmax` minimum plus its even share of the `1fr` remainder, so the
   * tiles stay close to their authored size and the leftover space sits in
   * the trailing empty tracks instead of stretching the tiles. That also
   * means the tile needs no `max-width` cap.
   */
  it("lays out .ms-shape as a grid of auto-fill, minmax(48px, 1fr) columns", () => {
    const start = source.indexOf(".ms-shape {");
    const end = source.indexOf(".ms-shape__btn {");
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    const block = source.slice(start, end);
    expect(block).toMatch(/display:\s*grid/);
    expect(block).toMatch(
      /grid-template-columns:\s*repeat\(auto-fill,\s*minmax\(48px,\s*1fr\)\)/,
    );
  });

  it("gives .ms-shape__btn no fixed width or max-width, so the grid track sizes it", () => {
    const start = source.indexOf(".ms-shape__btn {");
    expect(start).toBeGreaterThanOrEqual(0);
    const end = source.indexOf("}", start);
    expect(end).toBeGreaterThan(start);
    const block = source.slice(start, end);
    expect(block).not.toMatch(/(?<!-)width:\s*\d/);
    expect(block).toMatch(/height:\s*60px/);
  });

  /*
   * The actual contract, not just the declared text: five tiles plus four
   * gaps must fit the 282px rail interior. A reorder of the CSS or a change
   * to ASPECTS would not be caught by the text pins above, so pin the
   * arithmetic too. `minmax(48px, 1fr)`'s floor is 48px and the row's own
   * `gap` is 7px — both re-read from the source so this fails the moment
   * either number drifts without the rail check being redone.
   */
  it("fits five tiles inside the 282px web rail interior at the minmax floor", () => {
    const trackMin = Number(
      source.match(
        /grid-template-columns:\s*repeat\(auto-fill,\s*minmax\((\d+)px/,
      )?.[1],
    );
    const gap = Number(
      source
        .slice(
          source.indexOf(".ms-shape {"),
          source.indexOf(".ms-shape__btn {"),
        )
        .match(/gap:\s*(\d+)px/)?.[1],
    );
    expect(trackMin).toBeGreaterThan(0);
    expect(gap).toBeGreaterThan(0);
    const tiles = ASPECTS.length;
    const required = tiles * trackMin + (tiles - 1) * gap;
    const RAIL_INTERIOR = 282;
    expect(required).toBeLessThanOrEqual(RAIL_INTERIOR);
  });
});
