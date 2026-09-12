import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SettingRow from "./SettingRow.vue";
import settingRowSource from "./SettingRow.vue?raw";

describe("SettingRow", () => {
  it("reads its height from the table-row token, never a literal", () => {
    // The phone has no `ui/mold-desktop.css`, so the token carries its own
    // fallback; a literal here would make web and desktop only accidentally
    // equal and would drift the moment the token moved.
    expect(settingRowSource).toMatch(
      /min-height:\s*var\(--mold-row-h-table,\s*52px\)/,
    );
  });

  it("shows the label, the help sentence, and nothing else it was not given", () => {
    const wrapper = mount(SettingRow, {
      props: {
        label: "Detail",
        help: "How many passes a new picture starts with.",
      },
    });
    expect(wrapper.text()).toContain("Detail");
    expect(wrapper.text()).toContain(
      "How many passes a new picture starts with.",
    );
    expect(wrapper.find("[data-test='setting-reset']").exists()).toBe(false);
  });

  it("tags where the value came from, and says when a restart is needed", () => {
    const wrapper = mount(SettingRow, {
      props: { label: "Server port", source: "file", needsEngineRestart: true },
    });
    expect(wrapper.text()).toContain("FILE");
    expect(wrapper.text()).toContain("RESTART ENGINE");
  });

  it("offers ↺ only on a resettable row, and never on a locked one", async () => {
    const resettable = mount(SettingRow, {
      props: { label: "Top-p", resettable: true },
    });
    const button = resettable.get("[data-test='setting-reset']");
    expect(button.attributes("aria-label")).toBe("Reset to default");
    await button.trigger("click");
    expect(resettable.emitted("reset")).toEqual([[]]);

    const locked = mount(SettingRow, {
      props: {
        label: "Top-p",
        resettable: true,
        lockedReason: "Locked by MOLD_EXPAND_TOP_P — unset it to edit here.",
      },
    });
    expect(locked.find("[data-test='setting-reset']").exists()).toBe(false);
    expect(locked.text()).toContain("unset it to edit here");
  });
});
