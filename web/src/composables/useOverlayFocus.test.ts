import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { defineComponent, h, ref } from "vue";
import { focusableWithin, useOverlayFocus } from "./useOverlayFocus";

/**
 * Minimal overlay host: a wrapper div bound to the trap plus three buttons,
 * one of which carries `tabindex="-1"` (the shape a roving-tabindex kit
 * control or a visually-hidden file input produces).
 */
const Overlay = defineComponent({
  props: { open: { type: Boolean, default: false } },
  setup(props) {
    const host = ref<HTMLElement | null>(null);
    const open = ref(props.open);
    const closed = ref(0);
    const { onKeydown } = useOverlayFocus(open, host, () => {
      closed.value += 1;
    });
    return { host, onKeydown, open, closed };
  },
  render() {
    if (!this.open) return null;
    return h("div", { ref: "host", onKeydown: this.onKeydown }, [
      h("button", { id: "a" }, "a"),
      h("button", { id: "skip", tabindex: "-1" }, "skip"),
      h("button", { id: "b" }, "b"),
    ]);
  },
});

function press(el: HTMLElement, shiftKey = false) {
  el.dispatchEvent(
    new KeyboardEvent("keydown", { key: "Tab", bubbles: true, shiftKey }),
  );
}

describe("focusableWithin", () => {
  it("skips disabled and tabindex=-1 nodes", () => {
    const root = document.createElement("div");
    root.innerHTML = `
      <button id="one"></button>
      <button id="two" disabled></button>
      <button id="three" tabindex="-1"></button>
      <input id="four" />
    `;
    expect(focusableWithin(root).map((el) => el.id)).toEqual(["one", "four"]);
  });
});

describe("useOverlayFocus", () => {
  it("locks body scrolling while open and restores the prior value", async () => {
    document.body.style.overflow = "clip";
    const w = mount(Overlay, { props: { open: false } });

    w.vm.open = true;
    await w.vm.$nextTick();
    expect(document.body.style.overflow).toBe("hidden");

    w.vm.open = false;
    await w.vm.$nextTick();
    expect(document.body.style.overflow).toBe("clip");
    document.body.style.overflow = "";
    w.unmount();
  });

  it("closes on Escape even when focus escaped the overlay", async () => {
    const outside = document.createElement("button");
    document.body.appendChild(outside);
    const w = mount(Overlay, {
      props: { open: true },
      attachTo: document.body,
    });
    outside.focus();

    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );

    expect(w.vm.closed).toBe(1);
    outside.remove();
    w.unmount();
  });

  it("cycles Tab and Shift+Tab across the panel's edges", () => {
    const w = mount(Overlay, {
      props: { open: true },
      attachTo: document.body,
    });
    const first = document.querySelector("#a") as HTMLButtonElement;
    const last = document.querySelector("#b") as HTMLButtonElement;

    last.focus();
    press(last);
    expect(document.activeElement).toBe(first);

    press(first, true);
    expect(document.activeElement).toBe(last);
    w.unmount();
  });

  it("pulls focus back inside when it sits outside the panel", () => {
    const outside = document.createElement("button");
    document.body.appendChild(outside);
    const w = mount(Overlay, {
      props: { open: true },
      attachTo: document.body,
    });
    const host = w.element as HTMLElement;

    outside.focus();
    press(host);
    expect(document.activeElement).toBe(document.querySelector("#a"));

    outside.focus();
    press(host, true);
    expect(document.activeElement).toBe(document.querySelector("#b"));

    outside.remove();
    w.unmount();
  });

  it("returns focus to the opener when the overlay closes", async () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();

    const w = mount(Overlay, {
      props: { open: false },
      attachTo: document.body,
    });
    w.vm.open = true;
    await w.vm.$nextTick();
    (document.querySelector("#a") as HTMLButtonElement).focus();

    w.vm.open = false;
    await w.vm.$nextTick();
    expect(document.activeElement).toBe(opener);

    opener.remove();
    w.unmount();
  });

  it("returns focus to the opener when an open overlay unmounts", async () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();

    const w = mount(Overlay, {
      props: { open: false },
      attachTo: document.body,
    });
    w.vm.open = true;
    await w.vm.$nextTick();
    (document.querySelector("#a") as HTMLButtonElement).focus();

    w.unmount();
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });

  it("does not throw when the opener left the document", async () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();

    const w = mount(Overlay, {
      props: { open: false },
      attachTo: document.body,
    });
    w.vm.open = true;
    await w.vm.$nextTick();
    opener.remove();

    expect(() => {
      w.vm.open = false;
    }).not.toThrow();
    w.unmount();
  });
});
