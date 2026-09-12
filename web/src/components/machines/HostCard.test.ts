import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import HostCard from "./HostCard.vue";
import type { HostStatus } from "./hostClient";
import type { HostEntry } from "../../lib/hostRegistry";
import type { ResourceSnapshot } from "../../types";

/*
 * The whole card is the door to a machine, the way the whole print tile is the
 * door to a print (`GalleryCard.vue`). The name button survives as the card's
 * accessible name, but it is no longer the only place that opens anything — a
 * person who clicks the GPU line, the meter, or the empty space beside them
 * meant to open the machine.
 *
 * A disconnected card is not a control at all: there is nothing to open, and
 * Connect is the only door.
 */

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

vi.mock("./hostClient", () => ({
  useHostPoll: () => poll,
  hostStatus: vi.fn(),
}));

function makeStatus(): HostStatus {
  return {
    version: "0.16.0",
    models_loaded: [],
    busy: false,
    uptime_secs: 120,
    queue_depth: 1,
    hostname: "plato",
    gpus: [
      {
        ordinal: 0,
        name: "NVIDIA L40S",
        vram_total_bytes: 48_000_000_000,
        vram_used_bytes: 12_000_000_000,
        state: "idle" as const,
      },
    ],
  } as unknown as HostStatus;
}

const host: HostEntry = {
  id: "plato",
  name: "plato",
  url: "http://plato:7680",
  connected: true,
};

beforeEach(() => {
  poll = {
    status: ref<HostStatus | null>(makeStatus()),
    resources: ref<ResourceSnapshot | null>(null),
    online: ref(true),
    stale: ref(false),
    authorityRejected: ref(false),
    lastSeen: ref<number | null>(Date.now()),
    error: ref<string | null>(null),
    loading: ref(false),
    refresh: vi.fn(),
    stop: vi.fn(),
  };
});

function mountCard(over: Partial<HostEntry> = {}) {
  return mount(HostCard, { props: { host: { ...host, ...over } } });
}

describe("HostCard as a door", () => {
  it("announces itself as one control that opens the machine", () => {
    const card = mountCard().get("[data-test=host-card]");
    expect(card.attributes("role")).toBe("button");
    expect(card.attributes("tabindex")).toBe("0");
    expect(card.attributes("aria-label")).toBe("Open plato");
  });

  it("opens on a click anywhere on the card", async () => {
    const w = mountCard();
    await w.get("[data-test=host-card]").trigger("click");
    expect(w.emitted("open")).toEqual([["plato"]]);
  });

  it.each(["enter", "space"])("opens on %s", async (key) => {
    const w = mountCard();
    await w.get("[data-test=host-card]").trigger(`keydown.${key}`);
    expect(w.emitted("open")).toEqual([["plato"]]);
  });

  // Two tab stops onto the same destination is one too many; the name keeps
  // its label for the tests and for a screen reader reading the head row.
  it("keeps the name as a label but takes it out of the tab order", () => {
    const name = mountCard().get("[data-test=host-open]");
    expect(name.attributes("tabindex")).toBe("-1");
    expect(name.attributes("aria-label")).toBe("Open plato");
  });

  it("opens exactly once when the name itself is clicked", async () => {
    const w = mountCard();
    await w.get("[data-test=host-open]").trigger("click");
    expect(w.emitted("open")).toEqual([["plato"]]);
  });
});

describe("HostCard nested controls", () => {
  it.each([
    ["host-actions", "click"],
    ["host-actions", "keydown.enter"],
    ["host-actions", "keydown.space"],
  ])("%s on %s never opens the machine", async (target, event) => {
    const w = mountCard();
    await w.get(`[data-test=${target}]`).trigger(event);
    expect(w.emitted("open")).toBeUndefined();
  });

  it("keeps Retry to itself", async () => {
    poll.online.value = false;
    const w = mountCard();
    const retry = w.get("[data-test=host-retry]");
    await retry.trigger("keydown.enter");
    await retry.trigger("keydown.space");
    await retry.trigger("click");
    expect(w.emitted("open")).toBeUndefined();
    expect(poll.refresh).toHaveBeenCalledTimes(1);
  });

  it("keeps Connect to itself", async () => {
    const w = mountCard({ connected: false });
    const connect = w.get("[data-test=host-reconnect]");
    await connect.trigger("keydown.enter");
    await connect.trigger("click");
    expect(w.emitted("open")).toBeUndefined();
    expect(w.emitted("reconnect")).toEqual([["plato"]]);
  });
});

describe("HostCard when the machine is disconnected", () => {
  it("is not announced as a control", () => {
    const card = mountCard({ connected: false }).get("[data-test=host-card]");
    expect(card.attributes("role")).toBeUndefined();
    expect(card.attributes("tabindex")).toBeUndefined();
  });

  it("opens nothing on click or Enter — Connect is the only door", async () => {
    const w = mountCard({ connected: false });
    await w.get("[data-test=host-card]").trigger("click");
    await w.get("[data-test=host-card]").trigger("keydown.enter");
    expect(w.emitted("open")).toBeUndefined();
    expect(w.find("[data-test=host-reconnect]").exists()).toBe(true);
  });
});
