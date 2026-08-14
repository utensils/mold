import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount, type VueWrapper } from "@vue/test-utils";
import Tooltip from "./Tooltip.vue";

let wrapper: VueWrapper | null = null;

function mountTip(text = "Relative on-disk footprint") {
  return mount(Tooltip, {
    props: { text },
    slots: { default: '<button class="anchor">chip</button>' },
    attachTo: document.body,
  });
}

function tipEl(): HTMLElement | null {
  return document.body.querySelector('[role="tooltip"]');
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  document.body.innerHTML = "";
  vi.useRealTimers();
});

describe("Tooltip", () => {
  it("shows the tip after the hover delay and hides on leave", async () => {
    wrapper = mountTip();
    await wrapper.get(".ms-tooltip").trigger("mouseenter");
    expect(tipEl()).toBeNull();

    vi.advanceTimersByTime(400);
    await wrapper.vm.$nextTick();
    expect(tipEl()?.textContent).toContain("Relative on-disk footprint");

    await wrapper.get(".ms-tooltip").trigger("mouseleave");
    expect(tipEl()).toBeNull();
  });

  it("shows immediately on keyboard focus and hides on blur and Escape", async () => {
    wrapper = mountTip();
    await wrapper.get(".ms-tooltip").trigger("focusin");
    await wrapper.vm.$nextTick();
    expect(tipEl()).not.toBeNull();

    await wrapper.get(".ms-tooltip").trigger("keydown", { key: "Escape" });
    expect(tipEl()).toBeNull();

    await wrapper.get(".ms-tooltip").trigger("focusin");
    await wrapper.vm.$nextTick();
    expect(tipEl()).not.toBeNull();
    await wrapper.get(".ms-tooltip").trigger("focusout");
    expect(tipEl()).toBeNull();
  });

  it("names the trigger's description accessibly while open", async () => {
    wrapper = mountTip();
    const root = wrapper.get(".ms-tooltip");
    expect(root.attributes("aria-describedby")).toBeUndefined();

    await root.trigger("focusin");
    await wrapper.vm.$nextTick();
    const id = root.attributes("aria-describedby");
    expect(id).toBeTruthy();
    expect(document.getElementById(id!)?.getAttribute("role")).toBe("tooltip");
  });

  it("flips below the trigger when the preferred top placement leaves the viewport", async () => {
    const anchorRect = {
      top: 4,
      bottom: 12,
      left: 100,
      right: 140,
      width: 40,
      height: 8,
    } as DOMRect;
    const clippedTipRect = {
      top: -10,
      bottom: 10,
      left: 100,
      right: 160,
      width: 60,
      height: 20,
    } as DOMRect;
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        return this.classList.contains("ms-tooltip__tip")
          ? clippedTipRect
          : anchorRect;
      },
    );

    wrapper = mountTip();
    await wrapper.get(".ms-tooltip").trigger("focusin");
    await wrapper.vm.$nextTick();
    await wrapper.vm.$nextTick();

    // Anchor bottom (12) + gap — below the trigger instead of off-screen.
    expect(tipEl()?.style.top).toBe("18px");
  });

  it("cancels a pending show when the pointer leaves before the delay", async () => {
    wrapper = mountTip();
    await wrapper.get(".ms-tooltip").trigger("mouseenter");
    await wrapper.get(".ms-tooltip").trigger("mouseleave");
    vi.advanceTimersByTime(1000);
    await wrapper.vm.$nextTick();
    expect(tipEl()).toBeNull();
  });
});
