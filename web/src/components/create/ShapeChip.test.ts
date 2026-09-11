import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ShapeChip from "./ShapeChip.vue";

describe("ShapeChip", () => {
  it("reads the shape it is handed, family first and size in mono", () => {
    const wrapper = mount(ShapeChip, {
      props: { label: "Square", sublabel: "1024" },
    });
    const chip = wrapper.get("[data-test='shape-chip']");
    expect(chip.text()).toContain("Square");
    expect(wrapper.get("[data-test='shape-chip-size']").text()).toBe("1024");
  });

  /*
   * The chip PRESENTS the one `resolveOutputShape` answer the page computes;
   * it never reads the form's pixels itself. A wide size therefore arrives
   * already written as `1216×704`, and an approximate one already marked.
   */
  it("states a non-square size exactly as the resolver wrote it", () => {
    const wrapper = mount(ShapeChip, {
      props: { label: "Landscape", sublabel: "≈1216×704" },
    });
    expect(wrapper.get("[data-test='shape-chip-size']").text()).toBe(
      "≈1216×704",
    );
  });

  it("says only the family when the resolver has no size to name", () => {
    const wrapper = mount(ShapeChip, {
      props: { label: "Source", sublabel: null },
    });
    expect(wrapper.find("[data-test='shape-chip-size']").exists()).toBe(false);
    expect(wrapper.get("[data-test='shape-chip']").text()).toContain("Source");
  });

  it("asks the page for the Shape group on click, claiming no popup of its own", async () => {
    // At 900px and above the page only scrolls the rail's Shape group into
    // view, so the chip cannot honestly promise a dialog.
    const wrapper = mount(ShapeChip, {
      props: { label: "Square", sublabel: "1024" },
    });
    const chip = wrapper.get("[data-test='shape-chip']");
    expect(chip.attributes("aria-haspopup")).toBeUndefined();
    await chip.trigger("click");
    expect(wrapper.emitted("open")).toHaveLength(1);
  });

  it("opens nothing while disabled", async () => {
    const wrapper = mount(ShapeChip, {
      props: { label: "Square", sublabel: "1024", disabled: true },
    });
    const chip = wrapper.get("[data-test='shape-chip']");
    expect(chip.attributes("disabled")).toBeDefined();
    await chip.trigger("click");
    expect(wrapper.emitted("open")).toBeUndefined();
  });
});
