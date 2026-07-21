import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobileAdvancedSheet from "./MobileAdvancedSheet.vue";

describe("MobileAdvancedSheet", () => {
  it("keeps hosted advanced controls mounted whether the sheet is open or closed", async () => {
    const wrapper = mount(MobileAdvancedSheet, {
      props: { open: false, count: 0 },
      slots: { default: '<button data-test="hosted-control">Scheduler</button>' },
    });

    // Controls stay in the DOM while collapsed so the Generate flow can still
    // read every advanced field and validity binding.
    expect(wrapper.find("[data-test='hosted-control']").exists()).toBe(true);
    expect(wrapper.get("[data-test='mobile-advanced-sheet']").classes()).not.toContain("is-open");

    await wrapper.setProps({ open: true });
    expect(wrapper.get("[data-test='mobile-advanced-sheet']").classes()).toContain("is-open");
    expect(wrapper.find("[data-test='hosted-control']").exists()).toBe(true);
  });

  it("shows the active-count badge only when at least one advanced setting is set", async () => {
    const wrapper = mount(MobileAdvancedSheet, {
      props: { open: true, count: 0 },
    });

    expect(wrapper.find("[data-test='mobile-advanced-count']").exists()).toBe(false);

    await wrapper.setProps({ count: 3 });
    expect(wrapper.get("[data-test='mobile-advanced-count']").text()).toBe("3");
  });

  it("emits close and reset for the header control and the reset action", async () => {
    const wrapper = mount(MobileAdvancedSheet, {
      props: { open: true, count: 2 },
    });

    await wrapper.get("[data-test='mobile-advanced-close']").trigger("click");
    await wrapper.get("[data-test='mobile-advanced-reset']").trigger("click");

    expect(wrapper.emitted("close")).toHaveLength(1);
    expect(wrapper.emitted("reset")).toHaveLength(1);
  });
});
