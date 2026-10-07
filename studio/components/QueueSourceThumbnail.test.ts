import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import QueueSourceThumbnail from "./QueueSourceThumbnail.vue";
const mocks = vi.hoisted(() => ({ fetch: vi.fn() }));
vi.mock("../api/queueSourceThumbnail", () => ({
  queueSourceThumbnail: mocks.fetch,
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
