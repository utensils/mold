import { mount, flushPromises } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import SourceMediaViewer from "./SourceMediaViewer.vue";
import { clearSourceResources } from "../lib/sourceMediaResources";
const content = vi.hoisted(() => vi.fn());
vi.mock("../lib/imageDimensions", () => ({
  imageDimensionsFromBytes: () => ({ width: 100, height: 100 }),
}));
vi.mock("../api/sourceMediaBrowser", () => ({
  retainedSourceMediaContent: content,
}));
const selection = {
  owner: {
    key: "a",
    filename: "print.png",
    title: "Print A",
    hostLabel: "Host A",
    target: { baseUrl: "https://a", apiKey: null },
  },
  member: {
    member_id: "source",
    role: "source_image",
    display_name: "Original.png",
    size_bytes: 123,
    media_kind: "image",
  },
};
beforeEach(() => {
  clearSourceResources();
  content.mockReset();
  URL.createObjectURL = vi.fn(() => "blob:original");
  URL.revokeObjectURL = vi.fn();
});
it("loads the original once on repeat open and emits exact usage", async () => {
  content.mockResolvedValue(new Blob(["original"], { type: "image/png" }));
  let view = mount(SourceMediaViewer, {
    props: { selection, usages: [selection] },
    global: { stubs: { Teleport: true } },
  });
  await flushPromises();
  expect(view.get("img").attributes("src")).toBe("blob:original");
  expect(view.text()).toContain("Print A");
  await view.get('[data-test="source-viewer-reuse"]').trigger("click");
  expect(view.emitted("use")?.[0]).toEqual([selection]);
  view.unmount();
  view = mount(SourceMediaViewer, {
    props: { selection, usages: [selection] },
    global: { stubs: { Teleport: true } },
  });
  await flushPromises();
  expect(content).toHaveBeenCalledTimes(1);
  view.unmount();
});
it("cancels pending originals when its owner is invalidated", async () => {
  let signal: AbortSignal | undefined;
  content.mockImplementation((_a, _b, _c, s) => {
    signal = s;
    return new Promise(() => {});
  });
  const view = mount(SourceMediaViewer, {
    props: { selection, usages: [selection] },
    global: { stubs: { Teleport: true } },
  });
  await flushPromises();
  clearSourceResources();
  await flushPromises();
  expect(signal?.aborted).toBe(true);
  expect(view.emitted("close")).toHaveLength(1);
  view.unmount();
});
it("does not publish a late original after switching exact owners", async () => {
  let resolve!: (blob: Blob) => void;
  content
    .mockImplementationOnce(
      () =>
        new Promise<Blob>((r) => {
          resolve = r;
        }),
    )
    .mockResolvedValue(new Blob(["second"]));
  const view = mount(SourceMediaViewer, {
    props: { selection, usages: [selection] },
    global: { stubs: { Teleport: true } },
  });
  await flushPromises();
  const next = {
    ...selection,
    owner: {
      ...selection.owner,
      key: "b",
      target: { baseUrl: "https://b", apiKey: null },
    },
  };
  await view.setProps({ selection: next, usages: [next] });
  await flushPromises();
  resolve(new Blob(["late"]));
  await flushPromises();
  expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
  view.unmount();
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:original");
});
