/**
 * The shell itself: what it starts on launch, which keys it swallows, and the
 * dialogs it owns on behalf of every surface.
 *
 * The redesign promoted gallery-derived counts to shell chrome — the sidebar's
 * picture count, the Queue view's Done today, Create's Recent strip — without
 * moving the load that fills them. Every `fetchAll` caller was a view or an
 * overlay, and both refresh paths refuse a bucket nobody opened, so those
 * counts read an empty store until My images was visited.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

vi.mock("./lib/ipc", () => ({
  ipc: {
    setDockBadge: vi.fn().mockResolvedValue(undefined),
    takeNotificationAction: vi.fn().mockResolvedValue(null),
  },
  inTauri: () => false,
}));

import App from "./App.vue";
import { __resetQueueCommandState, useQueueCommands } from "./composables/useQueueCommands";
import { useAppPrefsStore } from "./stores/appPrefs";
import { useConnectionStore } from "./stores/connection";
import { useEventsStore } from "./stores/events";
import { useGalleryStore } from "./stores/gallery";
import { useGenerationStore } from "./stores/generation";
import { useHostsStore } from "./stores/hosts";
import { useHostStatusStore } from "./stores/hostStatus";
import { useJobsStore } from "./stores/jobs";
import { useLandedPrintsStore } from "./stores/landedPrints";
import { useLibraryPrefsStore } from "./stores/libraryPrefs";
import { useUpdaterStore } from "./stores/updater";

const stub = { template: "<div />" };
let router: Router;

beforeEach(() => {
  setActivePinia(createPinia());
  __resetQueueCommandState();
});

/** Mount the shell with every child stubbed but the dialogs it owns itself,
 *  and every launch-time store action silenced. */
async function mountApp() {
  router = createRouter({
    history: createMemoryHistory(),
    routes: ["/create", "/queue", "/library", "/models", "/machines", "/settings"].map((path) => ({
      path,
      component: stub,
    })),
  });
  await router.push("/create");
  await router.isReady();

  vi.spyOn(useLibraryPrefsStore(), "init").mockImplementation(() => undefined);
  vi.spyOn(useJobsStore(), "startPolling").mockImplementation(() => undefined);
  vi.spyOn(useJobsStore(), "stopPolling").mockImplementation(() => undefined);
  vi.spyOn(useAppPrefsStore(), "init").mockResolvedValue(null as never);
  vi.spyOn(useUpdaterStore(), "init").mockResolvedValue(undefined as never);
  vi.spyOn(useHostsStore(), "init").mockResolvedValue(undefined as never);
  vi.spyOn(useConnectionStore(), "init").mockResolvedValue(undefined as never);
  vi.spyOn(useGenerationStore(), "resumeDurableGenerations").mockImplementation(() => undefined);
  vi.spyOn(useGenerationStore(), "reconcileDurableAll").mockResolvedValue(undefined as never);
  vi.spyOn(useHostStatusStore(), "stop").mockImplementation(() => undefined);
  vi.spyOn(useEventsStore(), "unsubscribe").mockImplementation(() => undefined);

  const wrapper = mount(App, {
    global: {
      plugins: [router],
      stubs: {
        TitleBar: stub,
        UpdateBanner: stub,
        Sidebar: stub,
        StatusBar: stub,
        Toasts: stub,
        CommandPalette: stub,
        ContextMenu: stub,
        LicenseAcceptanceDialog: stub,
      },
    },
  });
  await flushPromises();
  return wrapper;
}

describe("App — launch", () => {
  it("loads the gallery once, so the shell's counts are not reading an empty store", async () => {
    const gallery = useGalleryStore();
    const fetchAll = vi.spyOn(gallery, "fetchAll").mockResolvedValue(undefined);

    await mountApp();

    expect(fetchAll).toHaveBeenCalledTimes(1);
  });

  it("leaves an already-loaded gallery alone", async () => {
    const gallery = useGalleryStore();
    const fetchAll = vi.spyOn(gallery, "fetchAll").mockResolvedValue(undefined);
    // `loaded` is a getter over the current sources' buckets; a shell that
    // relaunches into a warm store must not refetch every host.
    vi.spyOn(gallery, "loaded", "get").mockReturnValue(true);

    await mountApp();

    expect(fetchAll).not.toHaveBeenCalled();
  });
});

/*
 * The subscription follows the FLEET, not this device. Remote-only is a
 * supported configuration — the built-in engine off, or failed to start — and
 * gating the whole subscription on `connection.ready` meant no machine got a
 * stream and the badge counted nothing, on machines that were perfectly
 * reachable. The capability probe and the old-server poller stay the
 * primary's; only the subscription's lifecycle moves.
 */
describe("App — fleet event subscription", () => {
  /** One ready remote, and this device's engine not running. */
  function remoteOnlyFleet() {
    const conn = useConnectionStore();
    conn.info = null;
    conn.status = "error";
    useHostsStore().extras = [
      {
        id: "plato",
        label: "plato",
        url: "http://plato:7680",
        apiKey: "plato-key",
        status: "ready",
        error: null,
        instanceId: "i-plato",
      },
    ];
  }

  it("subscribes when a machine is ready even though this device's engine is not", async () => {
    const events = useEventsStore();
    const resubscribe = vi.spyOn(events, "resubscribe").mockResolvedValue(undefined);
    await mountApp();
    expect(useConnectionStore().ready).toBe(false);

    remoteOnlyFleet();
    await nextTick();

    expect(resubscribe).toHaveBeenCalled();
  });

  it("does not tear the fleet's streams down when this device's engine drops", async () => {
    const events = useEventsStore();
    vi.spyOn(events, "resubscribe").mockResolvedValue(undefined);
    await mountApp();
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    useHostsStore().extras = [
      {
        id: "plato",
        label: "plato",
        url: "http://plato:7680",
        apiKey: "plato-key",
        status: "ready",
        error: null,
        instanceId: "i-plato",
      },
    ];
    await nextTick();
    vi.mocked(events.unsubscribe).mockClear();

    conn.status = "error";
    await nextTick();

    expect(events.unsubscribe).not.toHaveBeenCalled();
  });

  it("unsubscribes once no machine is ready", async () => {
    vi.spyOn(useEventsStore(), "resubscribe").mockResolvedValue(undefined);
    await mountApp();
    const events = useEventsStore();
    remoteOnlyFleet();
    await nextTick();
    vi.mocked(events.unsubscribe).mockClear();

    useHostsStore().extras[0]!.status = "error";
    await nextTick();

    expect(events.unsubscribe).toHaveBeenCalled();
  });
});

/*
 * The Dock badge answers "what landed while you were away", fleet-wide, and
 * clears the moment the window comes back — the way a messages badge does.
 * It deliberately no longer counts this app's own pending jobs: a long clip
 * left a standing number nobody could clear, and work another machine did
 * never showed at all.
 */
describe("App — Dock badge", () => {
  it("badges prints that landed while the app was in the background", async () => {
    const { ipc } = await import("./lib/ipc");
    await mountApp();
    const landed = useLandedPrintsStore();

    landed.unseen = new Map([
      ["a.png", "local"],
      ["b.png", "plato"],
    ]);
    await nextTick();

    expect(ipc.setDockBadge).toHaveBeenLastCalledWith(2);
  });

  it("clears the badge when the window comes back", async () => {
    const { ipc } = await import("./lib/ipc");
    await mountApp();
    const landed = useLandedPrintsStore();
    landed.unseen = new Map([["a.png", "local"]]);
    await nextTick();
    expect(ipc.setDockBadge).toHaveBeenLastCalledWith(1);

    window.dispatchEvent(new Event("focus"));
    await nextTick();

    expect(landed.count).toBe(0);
    expect(ipc.setDockBadge).toHaveBeenLastCalledWith(null);
  });

  it("never badges this app's own pending queue", async () => {
    const { ipc } = await import("./lib/ipc");
    await mountApp();
    vi.mocked(ipc.setDockBadge).mockClear();
    const generation = useGenerationStore();

    generation.jobs.push({ status: "developing" } as never);
    await nextTick();

    expect(ipc.setDockBadge).not.toHaveBeenCalled();
  });
});

/*
 * The shell root is not a scroll container. `overflow: hidden` still lets a
 * programmatic scroll (focus() on a field, scrollIntoView) move the box, and
 * every `.sr-only` span in a view is `position: absolute` against the root
 * because nothing nearer is positioned — the Machines page's styles list gave
 * the root a 4900px scroll height, and focusing the Connect dialog's address
 * field shifted the whole chrome up by 8px. `overflow: clip` clips the same
 * way and can never scroll.
 */
describe("App — the shell root never scrolls", () => {
  it("clips its overflow rather than hiding it", async () => {
    const wrapper = await mountApp();
    const rootEl = wrapper.element as HTMLElement;
    expect(rootEl.className).toContain("overflow-clip");
    expect(rootEl.className).not.toContain("overflow-hidden");
    expect(wrapper.get("main").element.className).toContain("overflow-clip");
  });
});

describe("App — keys the shell swallows", () => {
  function pressBackspace(): KeyboardEvent {
    const event = new KeyboardEvent("keydown", {
      key: "Backspace",
      bubbles: true,
      cancelable: true,
    });
    window.dispatchEvent(event);
    return event;
  }

  it("never lets a bare Backspace navigate the webview away from the app", async () => {
    // The webview reads Backspace outside a field as history Back, which in a
    // single-page app unmounts the window — the user's Delete on a two-scene
    // clip left the key unconsumed and the app went blank.
    await mountApp();
    (document.activeElement as HTMLElement | null)?.blur();
    expect(pressBackspace().defaultPrevented).toBe(true);
  });

  it("leaves Backspace to the field being typed in", async () => {
    await mountApp();
    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();

    expect(pressBackspace().defaultPrevented).toBe(false);
    input.remove();
  });
});

describe("App — the shared Stop everything confirm", () => {
  it("renders one dialog for the whole app, naming what it will stop", async () => {
    const wrapper = await mountApp();
    expect(wrapper.find("[data-test='confirm-dialog']").exists()).toBe(false);

    useQueueCommands().askStopEverything();
    await nextTick();

    const dialog = wrapper.get("[data-test='confirm-dialog']");
    expect(dialog.text()).toContain("Stop everything?");
    expect(dialog.text()).toContain("Anything part-finished is lost.");
    expect(wrapper.get("[data-test='confirm-accept']").text()).toBe("Stop everything");
  });

  it("closes on Cancel without stopping anything", async () => {
    const wrapper = await mountApp();
    const cancelAll = vi.spyOn(useJobsStore(), "cancelAll").mockResolvedValue(undefined as never);

    useQueueCommands().askStopEverything();
    await nextTick();
    await wrapper.get("[data-test='confirm-cancel']").trigger("click");
    await flushPromises();

    expect(wrapper.find("[data-test='confirm-dialog']").exists()).toBe(false);
    expect(cancelAll).not.toHaveBeenCalled();
  });
});
