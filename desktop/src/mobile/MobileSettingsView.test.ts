import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobileSettingsView from "./MobileSettingsView.vue";

describe("MobileSettingsView", () => {
  it("offers accessible theme choices and emits immediate updates", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: { theme: "system", themeFamily: "mold" },
        hostCount: 2,
        appVersion: "0.18.0",
      },
    });

    expect(wrapper.findAll("fieldset")).toHaveLength(2);
    expect(wrapper.text()).toContain("Change the chrome without changing the color of your prints");
    expect(wrapper.text()).toContain("2 hosts saved");
    expect(wrapper.text()).toContain("0.18.0");
    expect(wrapper.text()).toContain("James Brink");
    expect(wrapper.text()).toContain("Jeffrey Dilley");

    await wrapper.get('input[name="mobile-theme-family"][value="safelight"]').setValue(true);
    await wrapper.get('input[name="mobile-theme-appearance"][value="light"]').setValue(true);

    expect(wrapper.emitted("update")).toEqual([
      [{ themeFamily: "safelight" }],
      [{ theme: "light" }],
    ]);
  });

  it("routes host management through an explicit action", async () => {
    const wrapper = mount(MobileSettingsView, {
      props: {
        settings: { theme: "dark", themeFamily: "mold" },
        hostCount: 0,
        appVersion: "Development build",
      },
    });

    expect(wrapper.text()).toContain("No hosts saved");
    await wrapper.get(".mobile-settings-manage").trigger("click");
    expect(wrapper.emitted("manage-hosts")).toHaveLength(1);
  });
});
