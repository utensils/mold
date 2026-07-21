import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MachinesPage from "./MachinesPage.vue";

describe("MachinesPage", () => {
  it("renders the placeholder empty state", () => {
    const wrapper = mount(MachinesPage);
    expect(wrapper.get('[data-test="machines-title"]').text()).toBe("Machines");
    expect(wrapper.text()).toContain("No machines yet — add one.");
    expect(wrapper.text()).toContain("Machines arrive in the next milestone.");
  });
});
