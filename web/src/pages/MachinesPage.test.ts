import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick, ref } from "vue";
import type { HostStatus } from "../components/machines/hostClient";
import type { ResourceSnapshot } from "../types";
import MachinesPage from "./MachinesPage.vue";
import ConnectModal from "../components/machines/ConnectModal.vue";

const { toastMock } = vi.hoisted(() => ({ toastMock: vi.fn() }));

vi.mock("../lib/toasts", () => ({
  toast: toastMock,
  requestConfirm: vi.fn().mockResolvedValue(false),
}));

// Shared, mutable poll stub returned by the mocked useHostPoll (every HostCard
// shares it — the page renders one card here, the primary origin).
let poll: {
  status: ReturnType<typeof ref<HostStatus | null>>;
  resources: ReturnType<typeof ref<ResourceSnapshot | null>>;
  online: ReturnType<typeof ref<boolean>>;
  stale: ReturnType<typeof ref<boolean>>;
  authorityRejected: ReturnType<typeof ref<boolean>>;
  lastSeen: ReturnType<typeof ref<number | null>>;
  error: ReturnType<typeof ref<string | null>>;
  loading: ReturnType<typeof ref<boolean>>;
  refresh: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
};

vi.mock("../components/machines/hostClient", () => ({
  useHostPoll: () => poll,
}));

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));

function makeStatus(over: Partial<HostStatus> = {}): HostStatus {
  return {
    version: "0.16.0",
    models_loaded: [],
    busy: false,
    uptime_secs: 120,
    queue_depth: 1,
    gpus: [
      {
        ordinal: 0,
        name: "Apple M3 Max",
        vram_total_bytes: 48_000_000_000,
        vram_used_bytes: 30_000_000_000,
        state: "idle",
      },
    ],
    ...over,
  };
}

const mountPage = () =>
  mount(MachinesPage, {
    global: { stubs: { ConnectModal: true } },
  });

beforeEach(() => {
  localStorage.clear();
  toastMock.mockClear();
  poll = {
    status: ref<HostStatus | null>(null),
    resources: ref<ResourceSnapshot | null>(null),
    online: ref(false),
    stale: ref(false),
    authorityRejected: ref(false),
    lastSeen: ref<number | null>(null),
    error: ref<string | null>(null),
    loading: ref(true),
    refresh: vi.fn(),
    stop: vi.fn(),
  };
});

describe("MachinesPage", () => {
  it("keeps the workspace title", () => {
    const w = mountPage();
    expect(w.get('[data-test="machines-title"]').text()).toBe("Machines");
  });

  it("shows a shimmer skeleton while the first poll is in flight", () => {
    const w = mountPage();
    expect(w.find('[data-test="host-card-skeleton"]').exists()).toBe(true);
    expect(w.find('[data-test="host-card"]').exists()).toBe(false);
  });

  it("renders the primary origin card once online", async () => {
    poll.loading.value = false;
    poll.online.value = true;
    poll.lastSeen.value = Date.now();
    poll.status.value = makeStatus();
    const w = mountPage();
    await nextTick();
    expect(w.find('[data-test="host-card"]').exists()).toBe(true);
    expect(w.get('[data-test="host-name"]').text()).toBe("this server");
    expect(w.get('[data-test="host-queue"]').text()).toBe("queue 1");
    expect(w.get('[data-test="host-mem"]').text()).toContain("/ 48.0 GB");
  });

  it("opens a common-actions context menu for a connected host", async () => {
    poll.loading.value = false;
    poll.online.value = true;
    poll.status.value = makeStatus();
    const w = mountPage();
    await nextTick();

    await w.get('[data-test="host-card"]').trigger("contextmenu", {
      clientX: 24,
      clientY: 32,
    });
    const menu = w.get('[data-test="machine-context-menu"]');
    expect(menu.text()).toContain("Open details");
    // The default target is model-aware Auto, so even the origin offers the
    // explicit pin rather than claiming to already be the target.
    expect(menu.text()).toContain("Set as generation target");
    expect(menu.text()).toContain("Copy address");
    expect(menu.text()).not.toContain("Disconnect");
    expect(menu.text()).not.toContain("Forget");

    document.body.dispatchEvent(
      new PointerEvent("pointerdown", { bubbles: true }),
    );
    await nextTick();
    expect(w.find('[data-test="machine-context-menu"]').exists()).toBe(false);
  });

  it("shows a reconnecting card without claiming a never-reached host is offline", async () => {
    poll.loading.value = false;
    poll.online.value = false;
    poll.lastSeen.value = Date.now();
    const w = mountPage();
    await nextTick();
    expect(w.find('[data-test="host-offline"]').exists()).toBe(false);
    expect(w.find('[data-test="host-reconnecting-state"]').exists()).toBe(true);
    // The poll keeps retrying on its own — the card says so.
    expect(w.get('[data-test="host-reconnecting"]').text()).toBe(
      "reconnecting…",
    );
    await w.get('[data-test="host-retry"]').trigger("click");
    expect(poll.refresh).toHaveBeenCalled();
  });

  it("keeps last-good telemetry visible while a verified host reconnects", async () => {
    poll.loading.value = false;
    poll.online.value = true;
    poll.stale.value = true;
    poll.lastSeen.value = Date.now();
    poll.status.value = makeStatus({ queue_depth: 7 });

    const w = mountPage();
    await nextTick();

    expect(w.find('[data-test="host-offline"]').exists()).toBe(false);
    expect(w.get('[data-test="host-reconnecting-state"]').text()).toContain(
      "reconnecting",
    );
    expect(w.get('[data-test="host-queue"]').text()).toBe("queue 7");
  });

  it("shows a remembered disconnected host and reconnects it explicitly", async () => {
    localStorage.setItem(
      "mold.web.hosts.v1",
      JSON.stringify([
        {
          id: "studio-7680",
          name: "Studio",
          url: "http://studio:7680",
          apiKey: "secret",
          connected: false,
        },
      ]),
    );
    poll.loading.value = false;
    const w = mountPage();
    const card = w.findAll('[data-test="host-card"]')[1]!;
    expect(w.get('[data-test="host-disconnected"]').text()).toBe(
      "disconnected",
    );
    expect(card.attributes("role")).toBeUndefined();
    expect(card.attributes("tabindex")).toBeUndefined();
    expect(card.attributes("aria-label")).toBeUndefined();
    await w.get('[data-test="host-reconnect"]').trigger("click");
    expect(
      JSON.parse(localStorage.getItem("mold.web.hosts.v1") ?? "[]")[0],
    ).toMatchObject({
      connected: true,
      apiKey: "secret",
    });
  });

  it("offers reconnect and forget for a remembered host context menu", async () => {
    localStorage.setItem(
      "mold.web.hosts.v1",
      JSON.stringify([
        {
          id: "studio-7680",
          name: "Studio",
          url: "http://studio:7680",
          apiKey: "secret",
          connected: false,
        },
      ]),
    );
    poll.loading.value = false;
    const w = mountPage();
    const cards = w.findAll('[data-test="host-card"]');
    expect(cards).toHaveLength(2);
    await cards[1]!.trigger("contextmenu", { clientX: 24, clientY: 32 });

    const menu = w.get('[data-test="machine-context-menu"]');
    expect(menu.text()).toContain("Connect");
    expect(menu.text()).toContain("Copy address");
    expect(menu.text()).toContain("Forget…");
    expect(menu.text()).not.toContain("Disconnect");

    await menu
      .findAll("button")
      .find((button) => button.text() === "Connect")!
      .trigger("click");
    expect(
      JSON.parse(localStorage.getItem("mold.web.hosts.v1") ?? "[]")[0],
    ).toMatchObject({ connected: true });
    expect(toastMock).toHaveBeenCalledWith("success", "Studio connected.");
  });

  it("opens the connect modal from the header button", async () => {
    const w = mountPage();
    expect(w.findComponent(ConnectModal).props("open")).toBe(false);
    await w.get('[data-test="add-machine"]').trigger("click");
    expect(w.findComponent(ConnectModal).props("open")).toBe(true);
  });

  it("also opens the connect modal from the dashed add card", async () => {
    const w = mountPage();
    await w.get('[data-test="add-machine-card"]').trigger("click");
    expect(w.findComponent(ConnectModal).props("open")).toBe(true);
  });
});
