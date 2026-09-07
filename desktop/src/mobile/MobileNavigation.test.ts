import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobileNavigation from "./MobileNavigation.vue";
import { MOBILE_TABS } from "./navigation";

describe("mobile Studio navigation", () => {
  it("keeps all five destinations available with the deliberate short phone labels", async () => {
    const wrapper = mount(MobileNavigation, { props: { modelValue: "generate", queueCount: 0 } });
    expect(wrapper.findAll("button").map((button) => button.text())).toEqual([
      "Make",
      "Queue",
      "Images",
      "Styles",
      "Machines",
    ]);
    for (const item of MOBILE_TABS) {
      await wrapper.get(`[data-test='mobile-tab-${item.id}']`).trigger("click");
    }
    expect(wrapper.emitted("update:modelValue")).toEqual(MOBILE_TABS.map((item) => [item.id]));
  });

  it("announces queue work without crowding the tab label or hiding an empty destination", async () => {
    const wrapper = mount(MobileNavigation, { props: { modelValue: "queue", queueCount: 3 } });
    const queue = wrapper.get("[data-test='mobile-tab-queue']");
    expect(queue.attributes("aria-current")).toBe("page");
    expect(queue.text()).toContain("3 items");
    await wrapper.setProps({ queueCount: 0 });
    expect(queue.text()).toBe("Queue");
    expect(wrapper.findAll("button")).toHaveLength(5);
  });
});
