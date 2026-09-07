import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import NamedViewsPanel from "./NamedViewsPanel.vue";

const profile = {
  mode: "adjustable" as const,
  roles: ["front", "left", "back", "right"] as Array<
    "front" | "left" | "back" | "right"
  >,
  min_count: 1,
  max_count: 4,
};

describe("NamedViewsPanel", () => {
  it("renders every server-advertised semantic slot", () => {
    const wrapper = mount(NamedViewsPanel, { props: { profile } });
    for (const role of profile.roles) {
      expect(
        wrapper.find(`[data-test='named-view-${role}-well']`).exists(),
      ).toBe(true);
    }
    expect(wrapper.text()).toContain("1–4 required");
  });

  it("shows the selected view in its original slot", () => {
    const wrapper = mount(NamedViewsPanel, {
      props: {
        profile,
        modelValue: {
          back: {
            base64: "AAAA",
            filename: "rear.png",
            mimeType: "image/png",
            width: 4,
            height: 5,
          },
        },
      },
    });
    expect(
      wrapper.get("[data-test='named-view-back-preview']").attributes("alt"),
    ).toBe("Back object view");
    expect(wrapper.text()).toContain("rear.png");
  });
});
