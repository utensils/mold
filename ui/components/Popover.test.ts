import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { defineComponent, h, ref } from "vue";
import Popover from "./Popover.vue";
import popoverSource from "./Popover.vue?raw";
import drawerSource from "./DrawerPanel.vue?raw";
import modalSource from "./ModalPanel.vue?raw";
import desktopTokens from "../mold-desktop.css?raw";

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

// The panel teleports to <body>, so it is queried on the document,
// not through the wrapper's own subtree.
const panel = () => document.querySelector<HTMLElement>(".ms-popover__panel");

afterEach(() => {
  document.body.innerHTML = "";
});

describe("Popover", () => {
  it("keeps body-teleported menus above the drawer and modal that own their trigger", () => {
    const layer = (source: string, selector: string) => {
      const block = source
        .slice(source.indexOf(`${selector} {`))
        .split("}")[0]!;
      const value = block
        .match(/z-index:\s*([^;]+);/)![1]!
        .replace(
          /var\((--[\w-]+),\s*(\d+)\)/g,
          (_, token: string, fallback: string) =>
            desktopTokens.match(new RegExp(`${token}:\\s*(\\d+);`))?.[1] ??
            fallback,
        )
        .replace(/^calc\((.*)\)$/, "$1");
      const sum = value.split("+").map(Number);
      expect(sum.every(Number.isFinite)).toBe(true);
      return sum.reduce((total, next) => total + next, 0);
    };
    const menuLayer = layer(popoverSource, ".ms-popover__panel");
    expect(menuLayer).toBeGreaterThan(layer(drawerSource, ".ms-drawer"));
    expect(menuLayer).toBeGreaterThan(layer(modalSource, ".ms-modal"));
  });
  it("shows the panel only while open, as a labelled dialog", async () => {
    const wrapper = harness();
    expect(panel()).toBeNull();
    await wrapper.find(".trigger").trigger("click");
    expect(panel()?.querySelector(".content")).not.toBeNull();
    expect(panel()?.getAttribute("role")).toBe("dialog");
    expect(panel()?.getAttribute("aria-label")).toBe("Test menu");
    wrapper.unmount();
  });

  it("teleports the panel to <body> so a scrollable ancestor never grows", async () => {
    // Rendered in place, an open panel extended the Create bench's
    // scrollable overflow and grew a scrollbar; fixed under <body> it
    // contributes overflow to nothing.
    const wrapper = harness(true);
    await wrapper.vm.$nextTick();
    expect(panel()?.parentElement).toBe(document.body);
    expect(wrapper.element.querySelector(".ms-popover__panel")).toBeNull();
    wrapper.unmount();
  });

  it("dismisses on outside pointerdown but not inside the panel", async () => {
    const wrapper = harness(true);
    await wrapper.vm.$nextTick();
    panel()?.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    await wrapper.vm.$nextTick();
    expect(panel()).not.toBeNull();

    document.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    await wrapper.vm.$nextTick();
    expect(panel()).toBeNull();
    wrapper.unmount();
  });

  it("dismisses on Escape", async () => {
    const wrapper = harness(true);
    await wrapper.vm.$nextTick();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await wrapper.vm.$nextTick();
    expect(panel()).toBeNull();
    wrapper.unmount();
  });
});
