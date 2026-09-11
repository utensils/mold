import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import DisclosureRow from "./DisclosureRow.vue";
import DisclosureList from "./DisclosureList.vue";

function factory(
  props: Partial<InstanceType<typeof DisclosureRow>["$props"]> = {},
) {
  return mount(DisclosureRow, {
    props: { label: "Start from a photo", ...props },
  });
}

describe("DisclosureRow", () => {
  it("says what it is, what it does, and what it currently holds", () => {
    const wrapper = factory({
      note: "Crop, mask or use a face — opens over the page",
      value: "None",
    });
    expect(wrapper.get("[data-test='disclosure-label']").text()).toBe(
      "Start from a photo",
    );
    expect(wrapper.get("[data-test='disclosure-note']").text()).toBe(
      "Crop, mask or use a face — opens over the page",
    );
    expect(wrapper.get("[data-test='disclosure-value']").text()).toBe("None");
  });

  it("leaves out the note and the value it was not given", () => {
    const wrapper = factory();
    expect(wrapper.find("[data-test='disclosure-note']").exists()).toBe(false);
    expect(wrapper.find("[data-test='disclosure-value']").exists()).toBe(false);
  });

  it("opens on click", async () => {
    const wrapper = factory();
    await wrapper.get("[role='button']").trigger("click");
    expect(wrapper.emitted("open")).toHaveLength(1);
  });

  /*
   * The row is a door, so it answers to the keys a door answers to. It is not
   * a native <button> because its own children include drop wells and mono
   * readouts a button would flatten.
   */
  it("opens on Enter and Space, and on nothing else", async () => {
    const wrapper = factory();
    const row = wrapper.get("[role='button']");
    expect(row.attributes("tabindex")).toBe("0");
    await row.trigger("keydown", { key: "Enter" });
    await row.trigger("keydown", { key: " " });
    expect(wrapper.emitted("open")).toHaveLength(2);
    await row.trigger("keydown", { key: "a" });
    expect(wrapper.emitted("open")).toHaveLength(2);
  });

  it("carries the probe its owner names it by", () => {
    expect(
      factory({ testId: "create-source-row" })
        .find("[data-test='create-source-row']")
        .exists(),
    ).toBe(true);
  });
});

describe("DisclosureList", () => {
  it("lays its rows out in one bordered card", () => {
    const wrapper = mount(DisclosureList, {
      slots: {
        default: [
          "<div data-test='row-a'>a</div>",
          "<div data-test='row-b'>b</div>",
        ].join(""),
      },
    });
    expect(wrapper.get("[data-test='disclosure-list']").exists()).toBe(true);
    expect(wrapper.find("[data-test='row-a']").exists()).toBe(true);
    expect(wrapper.find("[data-test='row-b']").exists()).toBe(true);
  });
});
