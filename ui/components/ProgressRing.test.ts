import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ProgressRing from "./ProgressRing.vue";

const CIRCUMFERENCE = 2 * Math.PI * 34;

function make(value: number, extra: Record<string, unknown> = {}) {
  return mount(ProgressRing, { props: { value, ...extra } });
}

function dashOffset(wrapper: ReturnType<typeof make>): number {
  const attr = wrapper.find(".ms-ring__value").attributes("stroke-dashoffset");
  return Number.parseFloat(attr!);
}

describe("ProgressRing", () => {
  it("exposes progressbar semantics with the current value", () => {
    const wrapper = make(42);
    const bar = wrapper.find("[role=progressbar]");
    expect(bar.exists()).toBe(true);
    expect(bar.attributes("aria-valuenow")).toBe("42");
    expect(bar.attributes("aria-valuemin")).toBe("0");
    expect(bar.attributes("aria-valuemax")).toBe("100");
  });

  it("offsets the full circumference at 0", () => {
    expect(dashOffset(make(0))).toBeCloseTo(CIRCUMFERENCE, 4);
  });

  it("offsets half the circumference at 50", () => {
    expect(dashOffset(make(50))).toBeCloseTo(CIRCUMFERENCE / 2, 4);
  });

  it("offsets zero at 100", () => {
    expect(dashOffset(make(100))).toBeCloseTo(0, 4);
  });

  it("sets the dasharray to the circumference", () => {
    const attr = make(50)
      .find(".ms-ring__value")
      .attributes("stroke-dasharray");
    expect(Number.parseFloat(attr!)).toBeCloseTo(CIRCUMFERENCE, 4);
  });

  it("clamps out-of-range values", () => {
    const over = make(150);
    expect(dashOffset(over)).toBeCloseTo(0, 4);
    expect(over.attributes("aria-valuenow")).toBe("100");
    const under = make(-20);
    expect(dashOffset(under)).toBeCloseTo(CIRCUMFERENCE, 4);
    expect(under.attributes("aria-valuenow")).toBe("0");
  });

  it("defaults to 104px and honors the size prop", () => {
    const dflt = make(10);
    expect(dflt.attributes("style")).toContain("width: 104px");
    expect(dflt.find("svg").attributes("width")).toBe("104");
    const sized = make(10, { size: 76 });
    expect(sized.find("svg").attributes("width")).toBe("76");
    expect(sized.find("svg").attributes("viewBox")).toBe("0 0 80 80");
  });

  it("shows the percent label only when showLabel is set", () => {
    expect(make(62).find(".ms-ring__label").exists()).toBe(false);
    const labeled = make(62, { showLabel: true });
    expect(labeled.find(".ms-ring__label").text()).toBe("62%");
  });
});
