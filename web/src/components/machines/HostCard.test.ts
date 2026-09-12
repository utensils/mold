import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import HostCard from "./HostCard.vue";
import type { HostStatus } from "./hostClient";
import type { HostEntry } from "../../lib/hostRegistry";
import type { ResourceSnapshot } from "../../types";

/*
 * The whole card opens the machine, but the card itself is NOT the control.
 * `role="button"` prunes every descendant from the accessibility tree, and
 * this card's whole payload is text — the GPU sentence, the memory and queue
 * readouts, the meter's label, the reconnecting warning. A VoiceOver user
 * would have heard "Open plato, button" and nothing else.
 *
 * So the card stays a plain box and a real `<button>` is stretched over it.
 * One tab stop, one click target across the card, Enter and Space for free,
 * every readout still announced, and Retry / Connect / "…" no longer nested
 * inside a control.
 *
 * A disconnected card has no door at all: there is nothing to open, and
 * Connect is the only way back.
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

function mountCard(
  over: Partial<HostEntry> = {},
  props: Record<string, unknown> = {},
) {
  return mount(HostCard, { props: { host: { ...host, ...over }, ...props } });
}

describe("HostCard door", () => {
  it("is a real button covering the card, named for the machine", () => {
    const w = mountCard();
    const door = w.get("[data-test=host-open]");
    expect(door.element.tagName).toBe("BUTTON");
    expect(door.attributes("aria-label")).toBe("Open plato");
    // The card itself is not a control, so its readouts stay announced.
    const card = w.get("[data-test=host-card]");
    expect(card.attributes("role")).toBeUndefined();
    expect(card.attributes("tabindex")).toBeUndefined();
  });

  it("opens on click", async () => {
    const w = mountCard();
    await w.get("[data-test=host-open]").trigger("click");
    expect(w.emitted("open")).toEqual([["plato"]]);
  });

  it("leaves the card's readouts outside any control", () => {
    const w = mountCard();
    for (const hook of ["host-gpu", "host-mem", "host-queue"]) {
      const el = w.get(`[data-test=${hook}]`).element;
      expect(el.closest("button")).toBeNull();
    }
  });

  it("is the card's only tab stop besides its own controls", () => {
    const w = mountCard();
    const stops = w
      .findAll("button")
      .filter((b) => b.attributes("tabindex") !== "-1");
    expect(stops.map((b) => b.attributes("data-test"))).toEqual([
      "host-open",
      "host-actions",
    ]);
  });
});

describe("HostCard nested controls", () => {
  it.each(["click", "keydown.enter", "keydown.space"])(
    "the actions button never opens the machine on %s",
    async (event) => {
      const w = mountCard();
      await w.get("[data-test=host-actions]").trigger(event);
      expect(w.emitted("open")).toBeUndefined();
    },
  );

  it("keeps Retry to itself, and outside any control", async () => {
    poll.online.value = false;
    const w = mountCard();
    const retry = w.get("[data-test=host-retry]");
    expect(retry.element.parentElement?.closest("button")).toBeNull();
    await retry.trigger("keydown.enter");
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

/*
 * The Machines page closes an open context menu on `pointerdown` anywhere
 * outside it. The `click` that completes that same gesture then lands on the
 * card — so dismissing a menu by clicking the card must not also navigate.
 */
describe("HostCard dismissing the actions menu", () => {
  it("does not open the machine on the click that dismissed the menu", async () => {
    const w = mountCard({}, { actionsOpen: true });
    const door = w.get("[data-test=host-open]");
    await door.trigger("pointerdown");
    // The page's document handler has closed the menu by now.
    await w.setProps({ actionsOpen: false });
    await door.trigger("click");
    expect(w.emitted("open")).toBeUndefined();
  });

  it("opens on the next click, once the menu is gone", async () => {
    const w = mountCard({}, { actionsOpen: true });
    const door = w.get("[data-test=host-open]");
    await door.trigger("pointerdown");
    await w.setProps({ actionsOpen: false });
    await door.trigger("click");

    await door.trigger("pointerdown");
    await door.trigger("click");
    expect(w.emitted("open")).toEqual([["plato"]]);
  });

  it("opens normally when no menu was open", async () => {
    const w = mountCard();
    const door = w.get("[data-test=host-open]");
    await door.trigger("pointerdown");
    await door.trigger("click");
    expect(w.emitted("open")).toEqual([["plato"]]);
  });
});

describe("HostCard when the machine is disconnected", () => {
  it("has no door at all", () => {
    const w = mountCard({ connected: false });
    expect(w.find("[data-test=host-open]").exists()).toBe(false);
    expect(w.find("[data-test=host-reconnect]").exists()).toBe(true);
  });

  it("still names the machine, as plain text", () => {
    const w = mountCard({ connected: false });
    const name = w.get("[data-test=host-name]");
    expect(name.text()).toBe("plato");
    expect(name.element.closest("button")).toBeNull();
  });
});

/*
 * The door's geometry cannot be observed in a DOM with no layout engine, so
 * the rules that make it a door are pinned at the source, the way
 * `desktop/src/styles/kitLayer.test.ts` pins the kit's cascade layer.
 */
describe("HostCard door CSS", () => {
  const source = readFileSync(
    resolve(__dirname, "../../../../web/src/components/machines/HostCard.vue"),
    "utf8",
  );

  function ruleBody(selector: string): string {
    const marker = `\n${selector} {`;
    const start = source.indexOf(marker);
    expect(start, `no \`${selector} {\` rule`).toBeGreaterThan(0);
    const open = start + marker.length;
    return source.slice(open, source.indexOf("\n}", open));
  }

  it("stretches the door over the whole card", () => {
    expect(ruleBody(".hc-card--open")).toMatch(/position:\s*relative\s*;/);
    const door = ruleBody(".hc__door");
    expect(door).toMatch(/position:\s*absolute\s*;/);
    expect(door).toMatch(/inset:\s*0\s*;/);
  });

  it("lets a press on a readout reach the door underneath it", () => {
    const readouts = ruleBody(".hc__head,\n.hc__gpu,\n.hc__row,\n.hc__offline");
    expect(readouts).toMatch(/pointer-events:\s*none\s*;/);
    expect(readouts).toMatch(/z-index:\s*1\s*;/);
    expect(ruleBody(".hc__actions,\n.hc__retry")).toMatch(
      /pointer-events:\s*auto\s*;/,
    );
  });

  it("tints the hover INTO the card's own fill, never over the page", () => {
    // Mixing against `transparent` washes the page's deeper crust over the
    // card, so a hovered card sank instead of lifting, in every theme.
    expect(ruleBody(".hc-card--open:hover")).toMatch(
      /background:\s*color-mix\(in srgb, var\(--mold-text\) 4%, var\(--mold-bg\)\)\s*;/,
    );
  });

  it("traces the card with the focus ring, not the inset content box", () => {
    expect(source).toMatch(/\.hc__door:focus-visible \{/);
    expect(source).not.toMatch(/\.hc:focus-visible \{/);
  });
});
