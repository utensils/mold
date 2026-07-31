import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import MobileImagePickerSheet from "./MobileImagePickerSheet.vue";

const { apiJsonTo, apiFetchTo } = vi.hoisted(() => ({
  apiJsonTo: vi.fn(),
  apiFetchTo: vi.fn(),
}));

vi.mock("../lib/api/client", () => ({ apiJsonTo, apiFetchTo }));

const target = { baseUrl: "http://remote-host:7680", apiKey: "secret" };
const metadata = {
  prompt: "p",
  model: "m",
  seed: 1,
  steps: 1,
  guidance: 1,
  width: 1,
  height: 1,
};

describe("MobileImagePickerSheet", () => {
  beforeEach(() => {
    apiJsonTo.mockReset().mockResolvedValue([
      { filename: "still.png", metadata, timestamp: 3 },
      { filename: "photo.jpg", metadata, timestamp: 2 },
      { filename: "movie.mp4", metadata, timestamp: 1 },
    ]);
    apiFetchTo.mockReset().mockResolvedValue({
      headers: new Headers({ "content-length": "5" }),
      blob: () => Promise.resolve(new Blob(["bytes"], { type: "image/png" })),
    });
  });

  it("offers local PNG/JPEG input and filters the remote gallery", async () => {
    const wrapper = mount(MobileImagePickerSheet, {
      props: { open: true, target },
      global: { stubs: { AuthedMedia: true } },
    });
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-image-picker-input']").attributes("accept")).toBe(
      "image/png,image/jpeg",
    );
    await wrapper.get("[data-test='mobile-image-picker-gallery-tab']").trigger("click");
    expect(wrapper.findAll("[data-test='mobile-image-picker-gallery-item']")).toHaveLength(2);
  });

  it("rejects unsupported local image formats with an associated alert", async () => {
    const wrapper = mount(MobileImagePickerSheet, {
      props: { open: true, target },
      global: { stubs: { AuthedMedia: true } },
    });
    const input = wrapper.get<HTMLInputElement>("[data-test='mobile-image-picker-input']");
    Object.defineProperty(input.element, "files", {
      configurable: true,
      value: [new File(["photo"], "photo.webp", { type: "image/webp" })],
    });

    await input.trigger("change");
    await flushPromises();

    const alert = wrapper.get("[role='alert']");
    expect(alert.text()).toContain("PNG or JPEG");
    expect(wrapper.emitted("pick")).toBeUndefined();
  });

  it("fetches a gallery choice from the exact authenticated host and emits wire bytes", async () => {
    const wrapper = mount(MobileImagePickerSheet, {
      props: { open: true, target },
      global: { stubs: { AuthedMedia: true } },
    });
    await flushPromises();
    await wrapper.get("[data-test='mobile-image-picker-gallery-tab']").trigger("click");
    await wrapper.findAll("[data-test='mobile-image-picker-gallery-item']")[0]!.trigger("click");
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/still.png");
    expect(wrapper.emitted("pick")?.[0]?.[0]).toEqual({
      filename: "still.png",
      base64: "Ynl0ZXM=",
    });
  });

  it("honors a caller's combined-media budget before downloading gallery bytes", async () => {
    const wrapper = mount(MobileImagePickerSheet, {
      props: {
        open: true,
        target,
        maxBytes: 1,
        oversizeMessage: "Combined media is too large.",
      },
      global: { stubs: { AuthedMedia: true } },
    });
    await flushPromises();
    await wrapper.get("[data-test='mobile-image-picker-gallery-tab']").trigger("click");
    await wrapper.findAll("[data-test='mobile-image-picker-gallery-item']")[0]!.trigger("click");
    await flushPromises();

    expect(wrapper.get("[role='alert']").text()).toBe("Combined media is too large.");
    expect(wrapper.emitted("pick")).toBeUndefined();
  });

  it("ignores a stale gallery response after the target host changes", async () => {
    let resolveFirst!: (
      entries: Array<{ filename: string; metadata: typeof metadata; timestamp: number }>,
    ) => void;
    let resolveSecond!: (
      entries: Array<{ filename: string; metadata: typeof metadata; timestamp: number }>,
    ) => void;
    apiJsonTo
      .mockReset()
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSecond = resolve;
          }),
      );
    const wrapper = mount(MobileImagePickerSheet, {
      props: { open: true, target },
      global: { stubs: { AuthedMedia: true } },
    });
    const nextTarget = { baseUrl: "http://next-host:7680", apiKey: "next-secret" };
    await wrapper.setProps({ target: nextTarget });
    resolveSecond([{ filename: "next.png", metadata, timestamp: 2 }]);
    await flushPromises();
    resolveFirst([{ filename: "stale.png", metadata, timestamp: 1 }]);
    await flushPromises();

    await wrapper.get("[data-test='mobile-image-picker-gallery-tab']").trigger("click");
    const items = wrapper.findAll("[data-test='mobile-image-picker-gallery-item']");
    expect(items).toHaveLength(1);
    expect(items[0]!.attributes("aria-label")).toBe("Use next.png");
    expect(apiJsonTo).toHaveBeenNthCalledWith(2, nextTarget, "/api/gallery");
  });
});
