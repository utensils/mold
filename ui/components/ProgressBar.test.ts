import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ProgressBar from "./ProgressBar.vue";

function make(value: number, extra: Record<string, unknown> = {}) {
  return mount(ProgressBar, { props: { value, ...extra } });
}

describe("ProgressBar", () => {
  it("exposes progressbar semantics with the current value", () => {
    const wrapper = make(62, { label: "GPU load" });
    const bar = wrapper.find("[role=progressbar]");
    expect(bar.exists()).toBe(true);
    expect(bar.attributes("aria-valuenow")).toBe("62");
    expect(bar.attributes("aria-valuemin")).toBe("0");
    expect(bar.attributes("aria-valuemax")).toBe("100");
    expect(bar.attributes("aria-label")).toBe("GPU load");
  });

  it("sets the fill width from value", () => {
    const fill = make(41).find(".ms-bar__fill");
    expect(fill.attributes("style")).toContain("width: 41%");
  });

  it("clamps values above 100", () => {
    const wrapper = make(140);
    expect(wrapper.find(".ms-bar__fill").attributes("style")).toContain(
      "width: 100%",
    );
    expect(wrapper.attributes("aria-valuenow")).toBe("100");
  });

  it("clamps values below 0", () => {
    const wrapper = make(-5);
    expect(wrapper.find(".ms-bar__fill").attributes("style")).toContain(
      "width: 0%",
    );
    expect(wrapper.attributes("aria-valuenow")).toBe("0");
  });

  it("defaults to the accent tone", () => {
    expect(make(50).find(".ms-bar__fill").classes()).toContain(
      "ms-bar__fill--accent",
    );
  });

  it.each(["accent", "info", "success", "warning", "danger"] as const)(
    "applies the %s tone class",
    (tone) => {
      const fill = make(50, { tone }).find(".ms-bar__fill");
      expect(fill.classes()).toContain(`ms-bar__fill--${tone}`);
    },
  );

  it("defaults to a 6px track and honors the height prop", () => {
    expect(make(50).attributes("style")).toContain("height: 6px");
    expect(make(50, { height: 4 }).attributes("style")).toContain(
      "height: 4px",
    );
  });
});
