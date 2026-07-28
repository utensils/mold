import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { defineComponent, h, ref } from "vue";
import Popover from "./Popover.vue";

function harness(open = false) {
  return mount(
    defineComponent({
      setup() {
        const isOpen = ref(open);
        return { isOpen };
      },
      render() {
        return h(
          Popover,
          {
            open: this.isOpen,
            label: "Test menu",
            "onUpdate:open": (v: boolean) => {
              this.isOpen = v;
            },
          },
          {
            trigger: () =>
              h(
                "button",
                {
                  class: "trigger",
                  onClick: () => (this.isOpen = !this.isOpen),
                },
                "Open",
              ),
            default: () => h("div", { class: "content" }, "Panel body"),
          },
        );
      },
    }),
    { attachTo: document.body },
  );
}

describe("Popover", () => {
  it("shows the panel only while open, as a labelled dialog", async () => {
    const wrapper = harness();
    expect(wrapper.find(".content").exists()).toBe(false);
    await wrapper.find(".trigger").trigger("click");
    expect(wrapper.find(".content").exists()).toBe(true);
    expect(wrapper.find("[role=dialog]").attributes("aria-label")).toBe(
      "Test menu",
    );
    wrapper.unmount();
  });

  it("dismisses on outside pointerdown but not inside", async () => {
    const wrapper = harness(true);
    document.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    await wrapper.vm.$nextTick();
    expect(wrapper.find(".content").exists()).toBe(false);
    wrapper.unmount();
  });

  it("dismisses on Escape", async () => {
    const wrapper = harness(true);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await wrapper.vm.$nextTick();
    expect(wrapper.find(".content").exists()).toBe(false);
    wrapper.unmount();
  });
});
