import { createHeldQueueTransfer } from "@studio/composables/useHeldQueueTransfer";
import { mount } from "@vue/test-utils";
import { computed, defineComponent, ref, nextTick } from "vue";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useMobileBack } from "./useMobileBack";
vi.mock("./platform", () => ({ isNativeAndroidRuntime: () => true }));

async function settle() {
  for (let i = 0; i < 6; i++) await nextTick();
}

function setup() {
  const states: unknown[] = [null];
  let cursor = 0;
  vi.spyOn(window.history, "state", "get").mockImplementation(() => states[cursor]);
  vi.spyOn(window.history, "pushState").mockImplementation((state) => {
    states.splice(cursor + 1);
    states.push(state);
    cursor++;
  });
  const go = vi.spyOn(window.history, "go").mockImplementation((delta = 0) => {
    queueMicrotask(() => {
      cursor += delta;
      if (cursor < 0 || cursor >= states.length) throw new Error("History escaped the app");
      window.dispatchEvent(new PopStateEvent("popstate", { state: states[cursor] }));
    });
  });
  const first = ref(false);
  const second = ref(false);
  const closeFirst = vi.fn(() => {
    first.value = false;
  });
  const closeSecond = vi.fn(() => {
    second.value = false;
  });
  const harness = mount(
    defineComponent({
      setup() {
        useMobileBack(first, closeFirst);
        useMobileBack(second, closeSecond);
        return () => null;
      },
    }),
  );
  return { first, second, closeFirst, closeSecond, go, harness, cursor: () => cursor };
}

afterEach(() => vi.restoreAllMocks());

describe("mobile Android Back", () => {
  it("dismisses a held-transfer picker without losing an active send or its parent", async () => {
    const h = setup();
    const transfer = createHeldQueueTransfer(
      computed(() => [
        {
          id: "source",
          label: "Source",
          instanceId: "source",
          ready: true,
          target: { baseUrl: "http://source", apiKey: null },
        },
        {
          id: "dest",
          label: "Destination",
          instanceId: "dest",
          ready: true,
          target: { baseUrl: "http://dest", apiKey: null },
        },
      ]),
    );
    const picker = mount(
      defineComponent({
        setup() {
          useMobileBack(
            computed(() => transfer.selection.value !== null),
            transfer.close,
          );
          return () => null;
        },
      }),
    );
    h.first.value = true;
    transfer.open("source", "held-job");
    transfer.busy.value = true;
    window.dispatchEvent(new Event("mold:native-back", { cancelable: true }));
    expect(transfer.selection.value?.jobId).toBe("held-job");
    expect(h.first.value).toBe(true);
    transfer.busy.value = false;
    window.dispatchEvent(new Event("mold:native-back", { cancelable: true }));
    expect(transfer.selection.value).toBeNull();
    expect(h.first.value).toBe(true);
    await settle();
    h.first.value = false;
    await settle();
    picker.unmount();
    h.harness.unmount();
    await settle();
  });

  it("consumes immediate native Back before history reconciliation", async () => {
    const h = setup();
    h.first.value = true;
    h.second.value = true;
    const back = new Event("mold:native-back", { cancelable: true });
    window.dispatchEvent(back);
    expect(back.defaultPrevented).toBe(true);
    expect(h.closeSecond).toHaveBeenCalledOnce();
    expect(h.closeFirst).not.toHaveBeenCalled();
    await settle();
    h.first.value = false;
    await settle();
    const rootBack = new Event("mold:native-back", { cancelable: true });
    window.dispatchEvent(rootBack);
    expect(rootBack.defaultPrevented).toBe(false);
    h.harness.unmount();
    await settle();
  });

  it("dismisses details before the viewer and Done leaves the viewer open", async () => {
    const h = setup();
    h.first.value = true;
    h.second.value = true;
    await settle();
    h.go(-1);
    await settle();
    expect(h.closeSecond).toHaveBeenCalledOnce();
    expect(h.closeFirst).not.toHaveBeenCalled();
    h.second.value = true;
    await settle();
    h.second.value = false;
    await settle();
    expect(h.cursor()).toBe(1);
    expect(h.closeFirst).not.toHaveBeenCalled();
    h.go(-1);
    await settle();
    expect(h.closeFirst).toHaveBeenCalledOnce();
    expect(h.cursor()).toBe(0);
    h.harness.unmount();
    await settle();
  });

  it("consumes a viewer and its details together when the viewer closes", async () => {
    const h = setup();
    h.first.value = true;
    h.second.value = true;
    await settle();
    h.first.value = false;
    h.second.value = false;
    await settle();
    expect(h.go).toHaveBeenCalledExactlyOnceWith(-2);
    expect(h.cursor()).toBe(0);
    expect(h.closeFirst).not.toHaveBeenCalled();
    expect(h.closeSecond).not.toHaveBeenCalled();
    h.harness.unmount();
    await settle();
  });

  it("can replace a surface and skip its dismissed history slot", async () => {
    const h = setup();
    h.first.value = true;
    await settle();
    h.first.value = false;
    h.second.value = true;
    await settle();
    h.go(-1);
    await settle();
    expect(h.closeSecond).toHaveBeenCalledOnce();
    expect(h.closeFirst).not.toHaveBeenCalled();
    expect(h.cursor()).toBe(0);
    h.harness.unmount();
    await settle();
  });

  it("preserves a reopened surface while its previous close is navigating", async () => {
    const h = setup();
    h.first.value = true;
    await settle();
    h.first.value = false;
    await nextTick();
    h.first.value = true;
    await settle();
    expect(h.first.value).toBe(true);
    h.go(-1);
    await settle();
    expect(h.closeFirst).toHaveBeenCalledOnce();
    expect(h.cursor()).toBe(0);
    h.harness.unmount();
    await settle();
  });
});
