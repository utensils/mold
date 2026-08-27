import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import SwipeActionRow from "./SwipeActionRow.vue";

function mountRow(props: Record<string, unknown> = {}) {
  return mount(SwipeActionRow, {
    props: {
      label: "FLUX print",
      actions: [
        {
          id: "cancel",
          label: "Cancel",
          tone: "danger",
          commitOnFullSwipe: true,
        },
      ],
      ...props,
    },
    slots: { default: "<span>FLUX print</span>" },
    attachTo: document.body,
  });
}

function pointer(
  wrapper: ReturnType<typeof mountRow>,
  name: string,
  x: number,
  y = 0,
) {
  return wrapper
    .get('[data-test="swipe-action-row"] > div:last-child')
    .trigger(name, {
      pointerId: 1,
      pointerType: "touch",
      clientX: x,
      clientY: y,
    });
}

describe("SwipeActionRow", () => {
  it("reveals the tray on a right-to-left swipe without committing", async () => {
    const wrapper = mountRow();
    vi.spyOn(
      wrapper.get('[data-test="swipe-action-row"]').element,
      "getBoundingClientRect",
    ).mockReturnValue({ width: 390 } as DOMRect);

    await pointer(wrapper, "pointerdown", 300);
    await pointer(wrapper, "pointermove", 240);
    await pointer(wrapper, "pointerup", 240);

    expect(wrapper.emitted("act")).toBeUndefined();
    const surface = wrapper.get(
      '[data-test="swipe-action-row"] > div:last-child',
    );
    expect(surface.attributes("style")).toContain("translateX(-88px)");
  });

  it("a single full swipe from a closed row reveals but never commits", async () => {
    const wrapper = mountRow();
    vi.spyOn(
      wrapper.get('[data-test="swipe-action-row"]').element,
      "getBoundingClientRect",
    ).mockReturnValue({ width: 390 } as DOMRect);

    await pointer(wrapper, "pointerdown", 380);
    await pointer(wrapper, "pointermove", 80);
    await pointer(wrapper, "pointerup", 80);

    expect(wrapper.emitted("act")).toBeUndefined();
    expect(
      wrapper
        .get('[data-test="swipe-action-row"] > div:last-child')
        .attributes("style"),
    ).toContain("translateX(-88px)");
  });

  it("commits the opted-in action on a second full swipe from the revealed tray", async () => {
    const wrapper = mountRow();
    vi.spyOn(
      wrapper.get('[data-test="swipe-action-row"]').element,
      "getBoundingClientRect",
    ).mockReturnValue({ width: 390 } as DOMRect);

    await pointer(wrapper, "pointerdown", 300);
    await pointer(wrapper, "pointermove", 240);
    await pointer(wrapper, "pointerup", 240);
    expect(wrapper.emitted("act")).toBeUndefined();

    await pointer(wrapper, "pointerdown", 380);
    await pointer(wrapper, "pointermove", 80);
    await pointer(wrapper, "pointerup", 80);

    expect(wrapper.emitted("act")).toEqual([["cancel"]]);
  });

  it("closes a revealed tray when the pointer lands outside the row", async () => {
    const wrapper = mountRow();
    vi.spyOn(
      wrapper.get('[data-test="swipe-action-row"]').element,
      "getBoundingClientRect",
    ).mockReturnValue({ width: 390 } as DOMRect);

    await pointer(wrapper, "pointerdown", 300);
    await pointer(wrapper, "pointermove", 240);
    await pointer(wrapper, "pointerup", 240);
    document.body.dispatchEvent(
      new PointerEvent("pointerdown", { bubbles: true, pointerType: "touch" }),
    );
    await wrapper.vm.$nextTick();

    expect(
      wrapper
        .get('[data-test="swipe-action-row"] > div:last-child')
        .attributes("style"),
    ).toContain("translateX(0px)");
  });

  it("does not steal a vertical scroll", async () => {
    const wrapper = mountRow();
    await pointer(wrapper, "pointerdown", 300, 300);
    await pointer(wrapper, "pointermove", 288, 220);
    await pointer(wrapper, "pointerup", 288, 220);

    expect(wrapper.emitted("act")).toBeUndefined();
    expect(
      wrapper
        .get('[data-test="swipe-action-row"] > div:last-child')
        .attributes("style"),
    ).toContain("translateX(0px)");
  });

  it("restores rather than acting when the gesture is cancelled", async () => {
    const wrapper = mountRow();
    vi.spyOn(
      wrapper.get('[data-test="swipe-action-row"]').element,
      "getBoundingClientRect",
    ).mockReturnValue({ width: 390 } as DOMRect);

    await pointer(wrapper, "pointerdown", 380);
    await pointer(wrapper, "pointermove", 80);
    await pointer(wrapper, "pointercancel", 80);

    expect(wrapper.emitted("act")).toBeUndefined();
  });

  it("is inert while one of its actions is in flight", async () => {
    const wrapper = mountRow({ disabled: true });
    await pointer(wrapper, "pointerdown", 380);
    await pointer(wrapper, "pointermove", 80);
    await pointer(wrapper, "pointerup", 80);

    expect(wrapper.emitted("act")).toBeUndefined();
  });

  it("reaches every action without the gesture", async () => {
    const wrapper = mountRow();
    const more = wrapper.get('[data-test="swipe-row-actions"]');
    expect(more.attributes("aria-label")).toBe("Actions for FLUX print");
    expect(
      wrapper.get('[data-test="swipe-action-cancel"]').attributes("tabindex"),
    ).toBe("-1");

    await more.trigger("click");
    expect(more.attributes("aria-expanded")).toBe("true");
    expect(
      wrapper.get('[data-test="swipe-action-cancel"]').attributes("tabindex"),
    ).toBe("0");

    await wrapper.get('[data-test="swipe-action-cancel"]').trigger("click");
    expect(wrapper.emitted("act")).toEqual([["cancel"]]);
  });

  it("scopes the horizontal pan to the row so the list still scrolls", () => {
    const wrapper = mountRow();
    expect(
      getComputedStyle(wrapper.get('[data-test="swipe-action-row"]').element)
        .touchAction,
    ).not.toBe("none");
  });
});
