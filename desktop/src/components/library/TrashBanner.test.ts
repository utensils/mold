import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import TrashBanner from "./TrashBanner.vue";

describe("TrashBanner", () => {
  it("renders the shared retention sentence with numbers in the mono face", () => {
    const wrapper = mount(TrashBanner, {
      props: {
        hosts: [
          { label: "This device", retentionDays: 30 },
          { label: "plato", retentionDays: 7 },
        ],
        count: 3,
        bytes: 41_600_000,
      },
    });
    const summary = wrapper.find("[data-test='trash-banner-summary']");
    expect(summary.text()).toBe("Prints stay in the trash 30 d before purge · plato keeps 7 d");
    expect(summary.findAll("b").map((b) => b.text())).toEqual(["30 d", "7 d"]);
    expect(summary.find("b").classes()).toContain("font-mono");
    expect(wrapper.find("[data-test='trash-banner-count']").text()).toBe(
      "3 pictures in trash · 41.6 MB",
    );
    expect(wrapper.find("[data-test='trash-banner']").attributes("role")).toBe("status");
  });

  it("emits changeRetention from the mono retention control", async () => {
    const wrapper = mount(TrashBanner, {
      props: { hosts: [{ label: "This device", retentionDays: 0 }], count: 1 },
    });
    expect(wrapper.find("[data-test='trash-banner-summary']").text()).toBe(
      "Prints stay in the trash until you empty it",
    );
    expect(wrapper.find("[data-test='trash-banner-count']").text()).toBe("1 picture in trash");
    const link = wrapper.find("[data-test='trash-banner-link']");
    expect(link.text()).toBe("Keep forever ▼");
    expect(link.attributes("title")).toBe("Change retention · Machines");
    await link.trigger("click");
    expect(wrapper.emitted("changeRetention")).toHaveLength(1);
  });

  it("names this device's retention and carries Empty now at the right edge", async () => {
    const wrapper = mount(TrashBanner, {
      props: {
        hosts: [{ label: "This device", retentionDays: 30 }],
        count: 3,
      },
    });
    expect(wrapper.get("[data-test='trash-banner-link']").text()).toBe("Keep for 30 days ▼");
    const empty = wrapper.get("[data-test='empty-trash']");
    expect(empty.text()).toBe("Empty now");
    expect(empty.classes()).toContain("ms-toolbar-button--danger");
    await empty.trigger("click");
    expect(wrapper.emitted("emptyNow")).toHaveLength(1);
  });

  it("disables Empty now at zero", () => {
    const wrapper = mount(TrashBanner, {
      props: { hosts: [{ label: "This device", retentionDays: 30 }], count: 0 },
    });
    expect(wrapper.get("[data-test='empty-trash']").attributes("disabled")).toBeDefined();
  });

  it("explains when no connected machine keeps a trash, and hides the link", () => {
    const wrapper = mount(TrashBanner, { props: { hosts: [], count: 0 } });
    expect(wrapper.find("[data-test='trash-banner-summary']").text()).toBe(
      "No connected machine keeps a trash.",
    );
    expect(wrapper.find("[data-test='trash-banner-link']").exists()).toBe(false);
    expect(wrapper.find("[data-test='trash-banner-count']").text()).toBe("0 pictures in trash");
  });
});
