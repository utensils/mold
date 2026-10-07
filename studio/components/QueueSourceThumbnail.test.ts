import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import QueueSourceThumbnail from "./QueueSourceThumbnail.vue";
const mocks = vi.hoisted(() => ({
  fetch: vi.fn(),
  inputs: vi.fn(async () => [{ label: "Source", preview: true }]),
}));
vi.mock("../api/queueSourceThumbnail", () => ({
  queueSourceThumbnail: mocks.fetch,
  queueInputs: mocks.inputs,
}));
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
describe("retained source thumbnail lifetime", () => {
  it("keeps the image and pending request across equivalent polling targets", async () => {
    mocks.fetch.mockClear();
    let finish!: (blob: Blob) => void;
    mocks.fetch.mockImplementationOnce(
      () =>
        new Promise<Blob>((resolve) => {
          finish = resolve;
        }),
    );
    const revoke = vi.fn();
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL = vi.fn(() => "blob:stable");
        static revokeObjectURL = revoke;
      },
    );
    const view = mount(QueueSourceThumbnail, {
      props: {
        target: { baseUrl: "http://machine", apiKey: "key" },
        jobId: "job",
        instanceId: "instance",
      },
    });
    await flushPromises();
    const signal = mocks.fetch.mock.calls[0]![2] as AbortSignal;
    await view.setProps({
      target: { baseUrl: "http://machine", apiKey: "key" },
    });
    expect(signal.aborted).toBe(false);
    finish(new Blob(["source"]));
    await flushPromises();
    const img = view.get("img").element;
    await view.setProps({
      target: { baseUrl: "http://machine", apiKey: "key" },
    });
    expect(view.get("img").element).toBe(img);
    expect(view.get("img").attributes("src")).toBe("blob:stable");
    expect(mocks.fetch).toHaveBeenCalledTimes(1);
    expect(revoke).not.toHaveBeenCalled();
    view.unmount();
  });
  it("ignores obsolete host responses and releases blob URLs", async () => {
    let finish!: (blob: Blob) => void;
    mocks.fetch.mockImplementationOnce(
      () =>
        new Promise<Blob>((resolve) => {
          finish = resolve;
        }),
    );
    mocks.fetch.mockResolvedValueOnce(new Blob(["new"]));
    const create = vi.fn(() => "blob:new");
    const revoke = vi.fn();
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL = create;
        static revokeObjectURL = revoke;
      },
    );
    const view = mount(QueueSourceThumbnail, {
      props: {
        target: { baseUrl: "http://old", apiKey: "a" },
        jobId: "same",
        instanceId: "old",
      },
    });
    await flushPromises();
    const oldSignal = mocks.fetch.mock.calls[0]![2] as AbortSignal;
    await view.setProps({
      target: { baseUrl: "http://new", apiKey: "b" },
      instanceId: "new",
    });
    await flushPromises();
    finish(new Blob(["old"]));
    await flushPromises();
    expect(oldSignal.aborted).toBe(true);
    expect(create).toHaveBeenCalledTimes(1);
    expect(view.find("img").attributes("src")).toBe("blob:new");
    expect(view.text()).toBe("Source");
    view.unmount();
    expect(revoke).toHaveBeenCalledWith("blob:new");
  });
  it("does not request offline sources and retries on reconnect", async () => {
    mocks.fetch.mockClear();
    mocks.fetch.mockResolvedValue(new Blob(["source"]));
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL = vi.fn(() => "blob:source");
        static revokeObjectURL = vi.fn();
      },
    );
    const view = mount(QueueSourceThumbnail, {
      props: {
        target: { baseUrl: "http://machine", apiKey: null },
        jobId: "job",
        online: false,
      },
    });
    expect(mocks.fetch).not.toHaveBeenCalled();
    await view.setProps({ online: true });
    await flushPromises();
    expect(view.find("img").attributes("alt")).toBe(
      "Source image for this render",
    );
    view.unmount();
  });
});

it("details show every ordered role and keep later images after one preview fails", async () => {
  mocks.inputs.mockResolvedValueOnce([
    { index: 2, label: "Reference image 1", preview: true },
    { index: 3, label: "Reference image 2", preview: true },
    { index: 4, label: "Reference 3 · audio", preview: false },
  ] as never);
  mocks.fetch.mockReset();
  mocks.fetch.mockRejectedValueOnce(new Error("missing"));
  mocks.fetch.mockResolvedValueOnce(new Blob(["second"]));
  vi.stubGlobal(
    "URL",
    class extends URL {
      static createObjectURL = vi.fn(() => "blob:second");
      static revokeObjectURL = vi.fn();
    },
  );
  const view = mount(QueueSourceThumbnail, {
    props: {
      target: { baseUrl: "http://box", apiKey: null },
      jobId: "edit",
      detailed: true,
    },
  });
  await flushPromises();
  expect(view.findAll("figure")).toHaveLength(3);
  expect(view.text()).toContain("Reference image 1");
  expect(view.text()).toContain("Preview unavailable");
  expect(view.text()).toContain("Reference 3 · audio");
  expect(view.findAll("img")).toHaveLength(1);
  expect(mocks.fetch.mock.calls.map((call) => call[3])).toEqual([2, 3]);
  view.unmount();
});

it("compact rows keep their fallback when no image can be loaded", async () => {
  mocks.inputs.mockResolvedValueOnce([
    { index: 0, label: "Source audio", preview: false },
  ] as never);
  const view = mount(QueueSourceThumbnail, {
    props: { target: { baseUrl: "http://box", apiKey: null }, jobId: "audio" },
    slots: { default: "Running preview" },
  });
  await flushPromises();
  expect(view.text()).toBe("Running preview");
  view.unmount();
});
