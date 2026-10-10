import { mount, flushPromises } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  sourceMediaGalleryEvent,
  clearSourceResources,
} from "../lib/sourceMediaResources";
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
  clearSourceResources();
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
    expect(view.text()).toContain("25 of 101");
    expect(view.findAll('[data-test="source-card"]')).toHaveLength(25);
    await view.get(".source-browser__more").trigger("click");
    await flushPromises();
    expect(view.text()).toContain("101 of 101");
    expect(view.findAll('[data-test="source-card"]')).toHaveLength(48);
    const moreSources = () =>
      view
        .findAll("button")
        .find((button) => button.text().startsWith("Next sources"))!;
    await moreSources().trigger("click");
    await moreSources().trigger("click");
    expect(view.text()).toContain("Print 100");
    expect(inventory.mock.calls.map((c) => c[0].length)).toEqual([
      25, 25, 25, 25, 1,
    ]);
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

it("retains unchanged inventories on append and reorder", async () => {
  const view = mount(SourceMediaBrowser, {
    props: { owners: [owner(1), owner(2)] },
    ...options,
  });
  await flushPromises();
  await view.setProps({ owners: [owner(3), owner(1), owner(2)] });
  await flushPromises();
  expect(inventory.mock.calls.map((c) => c[0])).toEqual([
    ["1.png", "2.png"],
    ["3.png"],
  ]);
  await view.setProps({ owners: [owner(2), owner(1), owner(3)] });
  await flushPromises();
  expect(inventory).toHaveBeenCalledTimes(2);
  view.unmount();
});
it("bounds rendered members independently of owner count", async () => {
  inventory.mockImplementation(async (names: string[]) => ({
    inventories: Object.fromEntries(
      names.map((filename) => [
        filename,
        {
          availability: "available",
          members: Array.from({ length: 100 }, (_, n) => ({
            member_id: String(n),
            role: "source_image",
            display_name: "ref.png",
            size_bytes: 1,
          })),
        },
      ]),
    ),
  }));
  const view = mount(SourceMediaBrowser, {
    props: { owners: [owner(1)] },
    ...options,
  });
  await flushPromises();
  expect(view.findAll('[data-test="source-card"]')).toHaveLength(48);
  view.unmount();
});
it("deduplicates verified bytes across roles and leaves unknown identity separate", async () => {
  inventory.mockImplementation(async (names: string[]) => ({
    inventories: Object.fromEntries(
      names.map((filename) => [
        filename,
        {
          availability: "available",
          members: [
            {
              member_id: "same",
              role: filename === "1.png" ? "source_image" : "keyframes",
              display_name: "ref",
              size_bytes: 1,
              content_sha256: "b".repeat(64),
            },
            {
              member_id: "unknown",
              role: "source_image",
              display_name: "ref",
              size_bytes: 1,
            },
          ],
        },
      ]),
    ),
  }));
  const view = mount(SourceMediaBrowser, {
    props: { owners: [owner(1), owner(2)] },
    ...options,
  });
  await flushPromises();
  expect(view.findAll('[data-test="source-card"]')).toHaveLength(3);
  await view.setProps({ owners: [owner(2)] });
  await flushPromises();
  expect(view.findAll('[data-test="source-card"]')).toHaveLength(2);
  expect(view.text()).not.toContain("Print 1");
  view.unmount();
});
it("refreshes after freshness expires and preserves changed display labels without refetch", async () => {
  const view = mount(SourceMediaBrowser, {
    props: { owners: [owner(1)] },
    ...options,
  });
  await flushPromises();
  await view.setProps({ owners: [{ ...owner(1), title: "Renamed" }] });
  await flushPromises();
  expect(inventory).toHaveBeenCalledTimes(1);
  expect(view.text()).toContain("Renamed");
  view.unmount();
  vi.spyOn(Date, "now").mockReturnValue(Date.now() + 60_001);
  const reopened = mount(SourceMediaBrowser, {
    props: { owners: [owner(1)] },
    ...options,
  });
  await flushPromises();
  expect(inventory).toHaveBeenCalledTimes(2);
  reopened.unmount();
  vi.restoreAllMocks();
});
it("clamps source pages when refreshed results shrink", async () => {
  const many = (count: number) => ({
    inventories: {
      "1.png": {
        availability: "available",
        members: Array.from({ length: count }, (_, n) => ({
          member_id: String(n),
          role: "source_image",
          display_name: "reference",
          size_bytes: 1,
        })),
      },
    },
  });
  inventory.mockResolvedValue(many(100));
  const view = mount(SourceMediaBrowser, {
    props: { owners: [owner(1)] },
    ...options,
  });
  await flushPromises();
  await view
    .findAll("button")
    .find((b) => b.text().startsWith("Next sources"))!
    .trigger("click");
  inventory.mockResolvedValue(many(1));
  await view
    .findAll("button")
    .find((b) => b.text() === "Refresh sources")!
    .trigger("click");
  await flushPromises();
  expect(view.findAll('[data-test="source-card"]')).toHaveLength(1);
  view.unmount();
});

it("refreshes only the owner named by an authenticated gallery event", async () => {
  const view = mount(SourceMediaBrowser, {
    props: { owners: [owner(1), owner(2)] },
    ...options,
  });
  await flushPromises();
  inventory.mockClear();
  sourceMediaGalleryEvent(
    { baseUrl: "http://machine", apiKey: "wrong" },
    { type: "gallery_added", filename: "1.png" },
  );
  await flushPromises();
  expect(inventory).not.toHaveBeenCalled();
  sourceMediaGalleryEvent(owner(1).target, {
    type: "gallery_added",
    filename: "1.png",
  });
  await flushPromises();
  expect(inventory).toHaveBeenCalledTimes(1);
  expect(inventory.mock.calls[0]![0]).toEqual(["1.png"]);
  view.unmount();
});
