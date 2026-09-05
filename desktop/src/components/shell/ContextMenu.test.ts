import { beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import ContextMenu from "./ContextMenu.vue";
import { overlayDepth, resetOverlayStackForTests } from "@ui/lib/overlayStack";
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

describe("ContextMenu — an overlay like any other", () => {
  /** A menu over a dialog is the topmost overlay: registered, one Escape
   *  closes the menu and leaves the dialog beneath it standing. */
  it("registers on the overlay stack while it is open", async () => {
    resetOverlayStackForTests();
    const wrapper = mount(ContextMenu, { attachTo: document.body });
    expect(overlayDepth()).toBe(0);
    openMenu();
    await wrapper.vm.$nextTick();
    expect(overlayDepth()).toBe(1);
    useContextMenuStore().close();
    await wrapper.vm.$nextTick();
    expect(overlayDepth()).toBe(0);
    wrapper.unmount();
  });
});

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

  it("constrains the teleported menu to the scaled viewport", async () => {
    const wrapper = mount(ContextMenu, { attachTo: document.body });
    openMenu();
    await wrapper.vm.$nextTick();
    const menu = document.body.querySelector<HTMLElement>("[role='menu']");
    expect(menu?.className).toContain("max-h-[calc(100vh-12px)]");
    expect(menu?.className).toContain("max-w-[calc(100vw-12px)]");
    wrapper.unmount();
  });
});

describe("ContextMenu submenus + checkable items", () => {
  function openTree() {
    const menu = useContextMenuStore();
    menu.$patch({
      visible: true,
      x: 10,
      y: 10,
      entries: [
        { label: "Reuse settings", action: () => {} },
        {
          label: "Tags",
          children: [
            { label: "blue", checked: true, action: () => {} },
            { label: "keep", checked: false, action: () => {} },
          ],
        },
        { label: "Delete", danger: true, action: () => {} },
      ],
    });
    return menu;
  }

  it("marks a parent item with aria-haspopup and a chevron, no check column without checkable roots", async () => {
    const wrapper = mount(ContextMenu, { attachTo: document.body });
    openTree();
    await wrapper.vm.$nextTick();
    const items = Array.from(document.body.querySelectorAll("[role='menuitem']"));
    const tags = items.find((el) => el.textContent?.includes("Tags"))!;
    expect(tags.getAttribute("aria-haspopup")).toBe("menu");
    expect(tags.getAttribute("aria-expanded")).toBe("false");
    expect(tags.querySelector("[data-test='menu-chevron']")).not.toBeNull();
    expect(document.body.querySelector("[data-test='menu-check']")).toBeNull();
    expect(document.body.querySelector("[data-test='submenu']")).toBeNull();
    wrapper.unmount();
  });

  it("hovering a parent opens its submenu with checkbox roles and a ✓ column", async () => {
    const wrapper = mount(ContextMenu, { attachTo: document.body });
    const menu = openTree();
    await wrapper.vm.$nextTick();
    const tags = Array.from(document.body.querySelectorAll<HTMLElement>("[role='menuitem']")).find(
      (el) => el.textContent?.includes("Tags"),
    )!;
    tags.dispatchEvent(new MouseEvent("mouseenter"));
    await wrapper.vm.$nextTick();

    expect(menu.submenu?.parentIndex).toBe(1);
    expect(tags.getAttribute("aria-expanded")).toBe("true");
    const submenu = document.body.querySelector("[data-test='submenu']")!;
    expect(submenu.getAttribute("role")).toBe("menu");
    expect(submenu.getAttribute("aria-label")).toBe("Tags");
    const boxes = Array.from(submenu.querySelectorAll("[role='menuitemcheckbox']"));
    expect(boxes.map((b) => b.getAttribute("aria-checked"))).toEqual(["true", "false"]);
    const checks = Array.from(submenu.querySelectorAll("[data-test='menu-check']"));
    expect(checks.map((c) => c.textContent?.trim())).toEqual(["✓", ""]);

    // Hovering a leaf closes it again.
    const reuse = Array.from(document.body.querySelectorAll<HTMLElement>("[role='menuitem']")).find(
      (el) => el.textContent?.includes("Reuse"),
    )!;
    reuse.dispatchEvent(new MouseEvent("mouseenter"));
    await wrapper.vm.$nextTick();
    expect(menu.submenu).toBeNull();
    wrapper.unmount();
  });

  it("keyboard: ArrowRight opens, ArrowLeft closes, Escape closes the submenu before the menu", async () => {
    const wrapper = mount(ContextMenu, { attachTo: document.body });
    const menu = openTree();
    await wrapper.vm.$nextTick();
    const key = (k: string) =>
      window.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true }));

    key("ArrowDown");
    key("ArrowDown");
    expect(menu.highlighted).toBe(1);
    key("ArrowRight");
    expect(menu.submenu?.parentIndex).toBe(1);
    expect(menu.submenu?.highlighted).toBe(0);
    key("ArrowLeft");
    expect(menu.submenu).toBeNull();
    expect(menu.highlighted).toBe(1);

    key("ArrowRight");
    expect(menu.submenu).not.toBeNull();
    key("Escape");
    expect(menu.submenu).toBeNull();
    expect(menu.visible).toBe(true);
    key("Escape");
    expect(menu.visible).toBe(false);
    wrapper.unmount();
  });
});
