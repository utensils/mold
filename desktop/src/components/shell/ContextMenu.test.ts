import { beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import ContextMenu from "./ContextMenu.vue";
import { useContextMenuStore } from "../../stores/contextMenu";

beforeEach(() => {
  setActivePinia(createPinia());
});

function openMenu() {
  const menu = useContextMenuStore();
  menu.$patch({
    visible: true,
    x: 10,
    y: 10,
    entries: [
      { label: "Cancel", danger: true, action: () => {} },
      { separator: true },
      { label: "Show in Gallery", disabled: true, action: () => {} },
    ],
  });
}

describe("ContextMenu a11y roles", () => {
  it("exposes a vertical menu with menuitem roles and a separator", async () => {
    // Teleports to <body>; attachTo keeps the DOM around for querying.
    const wrapper = mount(ContextMenu, { attachTo: document.body });
    openMenu();
    await wrapper.vm.$nextTick();

    const menu = document.body.querySelector("[role='menu']");
    expect(menu).not.toBeNull();
    expect(menu?.getAttribute("aria-orientation")).toBe("vertical");
    expect(menu?.getAttribute("aria-label")).toBe("Context menu");

    const items = document.body.querySelectorAll("[role='menuitem']");
    expect(items.length).toBe(2);
    expect(document.body.querySelectorAll("[role='separator']").length).toBe(1);
    wrapper.unmount();
  });

  it("marks disabled items with aria-disabled", async () => {
    const wrapper = mount(ContextMenu, { attachTo: document.body });
    openMenu();
    await wrapper.vm.$nextTick();

    const items = Array.from(document.body.querySelectorAll("[role='menuitem']"));
    const disabled = items.find((el) => el.textContent?.includes("Show in Gallery"));
    expect(disabled?.getAttribute("aria-disabled")).toBe("true");
    const enabled = items.find((el) => el.textContent?.includes("Cancel"));
    expect(enabled?.getAttribute("aria-disabled")).toBeNull();
    wrapper.unmount();
  });
});
