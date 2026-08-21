import { afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import SequenceContextMenu from "./SequenceContextMenu.vue";
import type { SequenceMenuEntry } from "@studio/lib/sequenceContextMenu";

function entries(overrides: SequenceMenuEntry[] = []): SequenceMenuEntry[] {
  return overrides.length
    ? overrides
    : [
        { label: "Add clip", action: vi.fn(), disabled: false },
        { separator: true },
        { label: "Validate plan", action: vi.fn(), disabled: true },
        { separator: true },
        { label: "Clear sequence", action: vi.fn(), danger: true },
      ];
}

function mountMenu(list: SequenceMenuEntry[] = entries()) {
  return mount(SequenceContextMenu, {
    props: { entries: list, x: 40, y: 60 },
    attachTo: document.body,
  });
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("SequenceContextMenu", () => {
  it("renders items, separators, disabled state, and danger styling", () => {
    const wrapper = mountMenu();
    expect(wrapper.find("[data-test='sequence-context-menu']").exists()).toBe(
      true,
    );
    const items = wrapper.findAll("[data-test='sequence-context-item']");
    expect(items.map((item) => item.text())).toEqual([
      "Add clip",
      "Validate plan",
      "Clear sequence",
    ]);
    expect(items[0]!.attributes("disabled")).toBeUndefined();
    expect(items[1]!.attributes("disabled")).toBeDefined();
    expect(items[2]!.classes()).toContain("sq-context__danger");
    expect(wrapper.findAll(".sq-context__separator")).toHaveLength(2);
    expect(
      wrapper.get("[data-test='sequence-context-menu']").attributes("role"),
    ).toBe("menu");
  });

  it("runs an enabled item's action and closes", async () => {
    const action = vi.fn();
    const wrapper = mountMenu([{ label: "Add clip", action }]);
    await wrapper.get("[data-test='sequence-context-item']").trigger("click");
    expect(action).toHaveBeenCalledTimes(1);
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("never runs a disabled item", async () => {
    const action = vi.fn();
    const wrapper = mountMenu([{ label: "Add clip", action, disabled: true }]);
    await wrapper.get("[data-test='sequence-context-item']").trigger("click");
    expect(action).not.toHaveBeenCalled();
  });

  it("closes on Escape and on an outside pointerdown", async () => {
    const wrapper = mountMenu();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(wrapper.emitted("close")).toHaveLength(1);

    document.body.dispatchEvent(
      new PointerEvent("pointerdown", { bubbles: true }),
    );
    expect(wrapper.emitted("close")).toHaveLength(2);

    // A pointerdown inside the menu must not close it.
    wrapper
      .get("[data-test='sequence-context-menu']")
      .element.dispatchEvent(
        new PointerEvent("pointerdown", { bubbles: true }),
      );
    expect(wrapper.emitted("close")).toHaveLength(2);
  });

  it("clamps the panel inside the viewport", () => {
    const wrapper = mount(SequenceContextMenu, {
      props: { entries: entries(), x: 99_999, y: 99_999 },
      attachTo: document.body,
    });
    const style = wrapper
      .get("[data-test='sequence-context-menu']")
      .attributes("style");
    const left = Number(/left: (\d+(?:\.\d+)?)px/.exec(style ?? "")?.[1]);
    const top = Number(/top: (\d+(?:\.\d+)?)px/.exec(style ?? "")?.[1]);
    expect(left).toBeLessThan(window.innerWidth);
    expect(top).toBeLessThan(window.innerHeight);
    expect(left).toBeGreaterThanOrEqual(0);
    expect(top).toBeGreaterThanOrEqual(0);
  });
});
