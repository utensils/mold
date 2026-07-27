import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import type { VueWrapper } from "@vue/test-utils";
import ThumbnailSizeSlider from "./ThumbnailSizeSlider.vue";

/**
 * Reads the ramp glyph's geometry straight out of the rendered SVG so the
 * small-to-large silhouette stays a tested contract rather than eyeballed CSS.
 */
function rampGeometry(wrapper: VueWrapper) {
  const svg = wrapper.get('[data-test="thumbnail-size-ramp"]');
  const viewBox = (svg.attributes("viewBox") ?? "").split(/\s+/).map(Number);
  const viewWidth = viewBox[2] ?? Number.NaN;
  const viewHeight = viewBox[3] ?? Number.NaN;
  const points = [
    ...svg
      .get("path")
      .attributes("d")!
      .matchAll(/(-?[\d.]+)[ ,]+(-?[\d.]+)/g),
  ].map(([, x, y]) => ({ x: Number(x), y: Number(y) }));

  const xs = points.map((point) => point.x);
  const edgeAt = (x: number) =>
    points.filter((point) => point.x === x).map((point) => point.y);
  const thicknessAt = (x: number) =>
    Math.max(...edgeAt(x)) - Math.min(...edgeAt(x));

  return {
    viewWidth,
    viewHeight,
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    leftThickness: thicknessAt(Math.min(...xs)),
    rightThickness: thicknessAt(Math.max(...xs)),
  };
}

describe("ThumbnailSizeSlider", () => {
  it("draws the small-to-large ramp as one unbroken glyph", () => {
    const wrapper = mount(ThumbnailSizeSlider, {
      props: { modelValue: 220, min: 120, max: 360, step: 10 },
    });

    // Two separately positioned wedges left a dead gap mid-track; one path
    // spanning the whole viewBox is what makes the ramp continuous.
    expect(wrapper.findAll("svg")).toHaveLength(1);
    expect(wrapper.findAll("path")).toHaveLength(1);

    const ramp = rampGeometry(wrapper);
    expect(ramp.minX).toBe(0);
    expect(ramp.maxX).toBe(ramp.viewWidth);
  });

  it("thickens the ramp from left to right so it reads dense-to-large", () => {
    const wrapper = mount(ThumbnailSizeSlider, {
      props: { modelValue: 220, min: 120, max: 360, step: 10 },
    });

    const ramp = rampGeometry(wrapper);
    expect(ramp.leftThickness).toBeGreaterThan(0);
    expect(ramp.rightThickness).toBeGreaterThan(ramp.leftThickness);
    expect(ramp.rightThickness).toBeLessThanOrEqual(ramp.viewHeight);
  });

  it("exposes an accessible pixel-valued range and emits numeric input", async () => {
    const wrapper = mount(ThumbnailSizeSlider, {
      props: { modelValue: 220, min: 120, max: 360, step: 10 },
    });
    const input = wrapper.get('input[aria-label="Thumbnail size"]');

    expect(input.attributes()).toMatchObject({
      min: "120",
      max: "360",
      step: "10",
      "aria-valuetext": "220 px",
    });

    (input.element as HTMLInputElement).value = "280";
    await input.trigger("input");
    expect(wrapper.emitted("update:modelValue")).toEqual([[280]]);
  });
});
