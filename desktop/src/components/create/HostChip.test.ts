/**
 * The generation-host chip. It used to live in the Create header; the
 * redesign moved it into the inspector's "Where it runs" row, so these
 * routing invariants now mount the component directly — the chip is the
 * authority for the persisted `generateTargetHost` contract wherever it is
 * rendered.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import HostChip from "./HostChip.vue";
import chipSource from "./HostChip.vue?raw";

import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";

/*
 * The routing menu is the shared Popover, whose panel teleports to <body>:
 * rendered in place it extended `.ms-inspector__scroll`'s scrollable area
 * instead of overlaying it, so merely opening it pushed the options below
 * the fold. These read the DOCUMENT rather than the wrapper's subtree for
 * exactly that reason.
 */
function menu(): HTMLElement | null {
  return document.querySelector("[data-test='host-menu']");
}
function option(id: string): HTMLElement {
  const el = document.querySelector<HTMLElement>(`[data-test='host-option-${id}']`);
  if (!el) throw new Error(`no host option ${id}`);
  return el;
}

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    appSettingsGet: vi.fn().mockResolvedValue({}),
  },
}));

beforeEach(() => setActivePinia(createPinia()));

/*
 * Every wrapper is UNMOUNTED, never wiped with `innerHTML = ""`. The menu
 * teleports to <body> and the Popover keeps document-level Escape and
 * pointerdown listeners while it is open, so a wrapper whose DOM was deleted
 * out from under it still answers the next Escape and patches a detached
 * tree.
 */
const mounted: { unmount: () => void }[] = [];
afterEach(() => {
  while (mounted.length) mounted.pop()!.unmount();
  document.body.innerHTML = "";
});

function mountChip() {
  const wrapper = mount(HostChip, { attachTo: document.body });
  mounted.push(wrapper);
  return wrapper;
}

function readyLocal() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
}

function addRemote(id = "hal9000-7680", label = "hal9000") {
  useHostsStore().extras.push({
    id,
    label,
    url: `http://${label}:7680`,
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: null,
  });
}

describe("HostChip", () => {
  it("does not open a routing menu with a single host", async () => {
    readyLocal();
    const wrapper = mountChip();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(menu()).toBeNull();
  });

  it("toggles the routing menu open and closed from the chip", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mountChip();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(menu()).not.toBeNull();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(menu()).toBeNull();
  });

  it("lists Auto, Most capable, and every host; picking one persists and closes", async () => {
    readyLocal();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: null } as never;
    const update = vi.spyOn(prefs, "update").mockResolvedValue(undefined as never);
    addRemote();
    const wrapper = mountChip();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(document.querySelector("[data-test='host-option-auto']")).not.toBeNull();
    expect(document.querySelector("[data-test='host-option-capable']")).not.toBeNull();
    option("hal9000-7680").click();
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ generateTargetHost: "hal9000-7680" });
    expect(menu()).toBeNull();
  });

  it("maps Auto back to null in the persisted setting", async () => {
    readyLocal();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: "hal9000-7680" } as never;
    const update = vi.spyOn(prefs, "update").mockResolvedValue(undefined as never);
    addRemote();
    const wrapper = mountChip();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    option("auto").click();
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ generateTargetHost: null });
  });

  it("shows a stale persisted pick as Auto when the host is gone", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: "ghost-7680" } as never;
    addRemote();
    const wrapper = mountChip();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(option("auto").getAttribute("aria-checked")).toBe("true");
  });

  it("names the sticky pick on the chip", () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    addRemote();
    const wrapper = mountChip();
    expect(wrapper.get("[data-test='host-chip']").text()).toContain("hal9000");
  });

  /*
   * "Where it runs" sits near the bottom of the inspector's Settings list,
   * inside `.ms-inspector__scroll` (`overflow-y: auto`). A hand-rolled
   * `position: absolute` panel there contributes to that ancestor's
   * scrollable area instead of overlaying it, so opening the menu grew the
   * inspector's scroll height and pushed the options below the fold — which
   * is the exact failure the shared Popover was written to fix.
   */
  it("uses the shared Popover instead of an in-flow absolute panel", () => {
    expect(chipSource).toContain('import Popover from "@ui/components/Popover.vue"');
    expect(chipSource).toMatch(/<Popover[\s\S]*v-model:open="popoverOpen"/);
    expect(chipSource).not.toContain("position: absolute");
    // Dismissal comes with the Popover; the chip no longer owns document
    // listeners of its own.
    expect(chipSource).not.toContain("document.addEventListener");
  });

  it("teleports the panel out of the scrolling inspector", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mountChip();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    const panel = menu();
    expect(panel).not.toBeNull();
    expect(wrapper.element.contains(panel)).toBe(false);
    expect(panel!.closest(".ms-popover__panel")).not.toBeNull();
  });

  it("closes the menu on Escape", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mountChip();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(menu()).toBeNull();
  });
});
