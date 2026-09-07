import { mount } from "@vue/test-utils";
import { defineComponent, ref, nextTick } from "vue";
import { describe, expect, it, vi } from "vitest";
import { useMobileBack } from "./useMobileBack";
vi.mock("./platform", () => ({ isNativeAndroidRuntime: () => true }));

describe("mobile Android Back", () => {
  it("dismisses only the top temporary surface and consumes Done without closing the one beneath", async () => {
    const first = ref(false);
    const second = ref(false);
    const closeFirst = vi.fn(() => {
      first.value = false;
    });
    const closeSecond = vi.fn(() => {
      second.value = false;
    });
    const back = vi.spyOn(window.history, "back").mockImplementation(() => {});
    const harness = mount(
      defineComponent({
        setup() {
          useMobileBack(first, closeFirst);
          useMobileBack(second, closeSecond);
          return () => null;
        },
      }),
    );
    first.value = true;
    second.value = true;
    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(closeSecond).toHaveBeenCalledOnce();
    expect(closeFirst).not.toHaveBeenCalled();
    expect(first.value).toBe(true);
    second.value = true;
    second.value = false;
    expect(back).toHaveBeenCalledOnce();
    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(closeFirst).not.toHaveBeenCalled();
    window.dispatchEvent(new PopStateEvent("popstate"));
    expect(closeFirst).toHaveBeenCalledOnce();
    await nextTick();
    harness.unmount();
    back.mockRestore();
    window.history.replaceState(null, "");
  });
});
