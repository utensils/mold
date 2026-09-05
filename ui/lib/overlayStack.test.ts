import { afterEach, describe, expect, it } from "vitest";
import { defineComponent, h, ref, type Ref } from "vue";
import { mount } from "@vue/test-utils";
import {
  createOverlayToken,
  isTopOverlay,
  overlayDepth,
  popOverlay,
  pushOverlay,
  resetOverlayStackForTests,
  useOverlayStack,
} from "./overlayStack";

afterEach(() => {
  // The register is module state every overlay shares; a failed assertion
  // must not leak a phantom overlay into the next test.
  resetOverlayStackForTests();
});

describe("overlay register", () => {
  it("counts one entry per overlay and names the newest as the top", () => {
    const first = createOverlayToken("first");
    const second = createOverlayToken("second");
    expect(overlayDepth()).toBe(0);
    expect(isTopOverlay(first)).toBe(false);

    pushOverlay(first);
    expect(overlayDepth()).toBe(1);
    expect(isTopOverlay(first)).toBe(true);

    pushOverlay(second);
    expect(overlayDepth()).toBe(2);
    expect(isTopOverlay(second)).toBe(true);
    expect(isTopOverlay(first)).toBe(false);

    popOverlay(second);
    expect(isTopOverlay(first)).toBe(true);
    popOverlay(first);
    expect(overlayDepth()).toBe(0);
  });

  it("pushing the same overlay twice registers it once", () => {
    const token = createOverlayToken("once");
    pushOverlay(token);
    pushOverlay(token);
    expect(overlayDepth()).toBe(1);
    popOverlay(token);
    expect(overlayDepth()).toBe(0);
  });

  it("popping an overlay that never registered is a no-op", () => {
    const token = createOverlayToken("absent");
    popOverlay(token);
    popOverlay(token);
    expect(overlayDepth()).toBe(0);
  });

  it("an overlay below the top leaves without disturbing the one above it", () => {
    const below = createOverlayToken("below");
    const above = createOverlayToken("above");
    pushOverlay(below);
    pushOverlay(above);
    popOverlay(below);
    expect(overlayDepth()).toBe(1);
    expect(isTopOverlay(above)).toBe(true);
  });
});

describe("useOverlayStack", () => {
  function host(open: Ref<boolean>) {
    return defineComponent({
      setup(_props, { expose }) {
        const stack = useOverlayStack(open);
        expose({ isTop: stack.isTop });
        return () => h("div");
      },
    });
  }
  const isTop = (wrapper: ReturnType<typeof mount>) =>
    (wrapper.vm as unknown as { isTop: () => boolean }).isTop();

  it("registers while open, releases on close, and survives a double toggle", async () => {
    const open = ref(false);
    const wrapper = mount(host(open));
    expect(overlayDepth()).toBe(0);
    expect(isTop(wrapper)).toBe(false);

    open.value = true;
    await wrapper.vm.$nextTick();
    expect(overlayDepth()).toBe(1);
    expect(isTop(wrapper)).toBe(true);

    open.value = false;
    await wrapper.vm.$nextTick();
    expect(overlayDepth()).toBe(0);

    open.value = true;
    await wrapper.vm.$nextTick();
    open.value = true;
    await wrapper.vm.$nextTick();
    expect(overlayDepth()).toBe(1);
    wrapper.unmount();
    expect(overlayDepth()).toBe(0);
  });

  it("releases the register when an open overlay is unmounted", () => {
    const open = ref(true);
    const wrapper = mount(host(open));
    expect(overlayDepth()).toBe(1);
    wrapper.unmount();
    expect(overlayDepth()).toBe(0);
  });

  it("an overlay opened over another one is the top, and hands it back on close", async () => {
    const lower = ref(true);
    const upper = ref(false);
    const under = mount(host(lower));
    const over = mount(host(upper));
    expect(isTop(under)).toBe(true);

    upper.value = true;
    await over.vm.$nextTick();
    expect(isTop(over)).toBe(true);
    expect(isTop(under)).toBe(false);

    upper.value = false;
    await over.vm.$nextTick();
    expect(isTop(under)).toBe(true);
    over.unmount();
    under.unmount();
  });
});
