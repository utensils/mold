import { mount, flushPromises } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SourceMediaBrowser from "./SourceMediaBrowser.vue";
import type { SourceMediaOwner } from "../lib/sourceMediaBrowser";
const inventory = vi.hoisted(() => vi.fn());
vi.mock("../api/sourceMediaBrowser", () => ({
  sourceMediaInventories: inventory,
}));
const owner = (n: number): SourceMediaOwner => ({
  key: String(n),
  filename: `${n}.png`,
  title: `Print ${n}`,
  hostLabel: "Machine",
  target: { baseUrl: "http://machine", apiKey: null },
});
const result = (names: string[]) => ({
  inventories: Object.fromEntries(
    names.map((filename) => [
      filename,
      {
        availability: "available",
        members: [
          {
            member_id: filename,
            role: "source_image",
            display_name: "original.png",
            size_bytes: 100,
          },
        ],
      },
    ]),
  ),
});
beforeEach(() => {
  inventory.mockReset();
  inventory.mockImplementation(async (names: string[]) => result(names));
});
const options = {
  global: {
    stubs: {
      SourceMediaCard: {
        props: ["selection"],
        template:
          '<div data-test="source-card">{{selection.owner.title}}</div>',
      },
    },
  },
};
describe("source media ownership browsing", () => {
  it("reaches owners beyond a batch without silently truncating", async () => {
    const view = mount(SourceMediaBrowser, {
      props: { owners: Array.from({ length: 101 }, (_, n) => owner(n)) },
      ...options,
    });
    await flushPromises();
    expect(view.text()).toContain("100 of 101");
    expect(view.findAll('[data-test="source-card"]')).toHaveLength(100);
    await view.get(".source-browser__more").trigger("click");
    await flushPromises();
    expect(view.text()).toContain("101 of 101");
    expect(view.text()).toContain("Print 100");
    expect(inventory.mock.calls.map((c) => c[0].length)).toEqual([100, 1]);
    view.unmount();
  });
  it("removes private owners immediately and ignores their delayed response", async () => {
    let finish!: (value: ReturnType<typeof result>) => void;
    inventory.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const view = mount(SourceMediaBrowser, {
      props: { owners: [owner(1)] },
      ...options,
    });
    await flushPromises();
    const oldSignal = inventory.mock.calls[0]![2] as AbortSignal;
    await view.setProps({ owners: [] });
    expect(oldSignal.aborted).toBe(true);
    finish(result(["1.png"]));
    await flushPromises();
    expect(view.text()).not.toContain("Print 1");
    expect(view.findAll('[data-test="source-card"]')).toHaveLength(0);
    view.unmount();
  });
  it("keeps partial failure visible and offers retry", async () => {
    inventory.mockRejectedValueOnce(new Error("Machine offline"));
    const view = mount(SourceMediaBrowser, {
      props: { owners: [owner(1)] },
      ...options,
    });
    await flushPromises();
    expect(view.text()).toContain("Machine offline");
    await view.get(".source-browser__error button").trigger("click");
    await flushPromises();
    expect(view.findAll('[data-test="source-card"]')).toHaveLength(1);
    view.unmount();
  });
});
