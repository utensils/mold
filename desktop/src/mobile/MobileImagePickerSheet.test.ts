import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import MobileImagePickerSheet from "./MobileImagePickerSheet.vue";

const { apiJsonTo, apiFetchTo } = vi.hoisted(() => ({
  apiJsonTo: vi.fn(),
  apiFetchTo: vi.fn(),
}));

vi.mock("../lib/api/client", () => ({ apiJsonTo, apiFetchTo }));

const target = { baseUrl: "http://remote-host:7680", apiKey: "secret" };
const peerTarget = { baseUrl: "http://peer-host:7680", apiKey: "peer-secret" };
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

  it("merges every available host and fetches a peer print from its own authenticated origin", async () => {
    apiJsonTo.mockImplementation((route: typeof target) =>
      Promise.resolve(
        route.baseUrl === peerTarget.baseUrl
          ? [{ filename: "peer.png", metadata, timestamp: 4 }]
          : [{ filename: "selected.png", metadata, timestamp: 3 }],
      ),
    );
    const gallerySources = [
      { id: "selected", label: "Studio", target },
      { id: "peer", label: "Render", target: peerTarget },
    ];
    const wrapper = mount(MobileImagePickerSheet, {
      props: { open: true, target, gallerySources },
      global: { stubs: { AuthedMedia: true } },
    });
    await flushPromises();
    await wrapper.get("[data-test='mobile-image-picker-gallery-tab']").trigger("click");

    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/gallery", {
      signal: expect.any(AbortSignal),
    });
    expect(apiJsonTo).toHaveBeenCalledWith(peerTarget, "/api/gallery", {
      signal: expect.any(AbortSignal),
    });
    const items = wrapper.findAll("[data-test='mobile-image-picker-gallery-item']");
    expect(items.map((item) => item.attributes("aria-label"))).toEqual([
      "Use peer.png from Render",
      "Use selected.png from Studio",
    ]);
    expect(
      wrapper.findAll("[data-test='mobile-image-picker-host']").map((chip) => chip.text()),
    ).toEqual(["Render", "Studio"]);

    await items[0]!.trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(peerTarget, "/api/gallery/image/peer.png");
  });

  it("renders a healthy host without waiting for a peer that never settles", async () => {
    apiJsonTo.mockImplementation((route: typeof target) =>
      route.baseUrl === peerTarget.baseUrl
        ? new Promise(() => {})
        : Promise.resolve([{ filename: "healthy.png", metadata, timestamp: 3 }]),
    );
    const wrapper = mount(MobileImagePickerSheet, {
      props: {
        open: true,
        target,
        gallerySources: [
          { id: "healthy", label: "Studio", target },
          { id: "hanging", label: "Render", target: peerTarget },
        ],
      },
      global: { stubs: { AuthedMedia: true } },
    });
    await flushPromises();
    await wrapper.get("[data-test='mobile-image-picker-gallery-tab']").trigger("click");

    expect(wrapper.findAll("[data-test='mobile-image-picker-gallery-item']")).toHaveLength(1);
    expect(wrapper.text()).not.toContain("Loading gallery");
    wrapper.unmount();
  });

  it("keeps healthy gallery results when a peer fails", async () => {
    apiJsonTo.mockImplementation((route: typeof target) =>
      route.baseUrl === peerTarget.baseUrl
        ? Promise.reject(new Error("peer offline"))
        : Promise.resolve([{ filename: "healthy.png", metadata, timestamp: 3 }]),
    );
    const wrapper = mount(MobileImagePickerSheet, {
      props: {
        open: true,
        target,
        gallerySources: [
          { id: "healthy", label: "Studio", target },
          { id: "failed", label: "Render", target: peerTarget },
        ],
      },
      global: { stubs: { AuthedMedia: true } },
    });
    await flushPromises();
    await wrapper.get("[data-test='mobile-image-picker-gallery-tab']").trigger("click");

    expect(wrapper.findAll("[data-test='mobile-image-picker-gallery-item']")).toHaveLength(1);
    expect(wrapper.find("[role='alert']").exists()).toBe(false);
  });

  it("removes stale tiles immediately when a host credential changes", async () => {
    const rotatedTarget = { ...target, apiKey: "rotated-secret" };
    let resolveRotated!: (
      entries: Array<{ filename: string; metadata: typeof metadata; timestamp: number }>,
    ) => void;
    apiJsonTo
      .mockReset()
      .mockResolvedValueOnce([{ filename: "old.png", metadata, timestamp: 1 }])
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRotated = resolve;
          }),
      );
    const wrapper = mount(MobileImagePickerSheet, {
      props: {
        open: true,
        target,
        gallerySources: [{ id: "studio", label: "Studio", target }],
      },
      global: { stubs: { AuthedMedia: true } },
    });
    await flushPromises();
    await wrapper.get("[data-test='mobile-image-picker-gallery-tab']").trigger("click");
    expect(wrapper.findAll("[data-test='mobile-image-picker-gallery-item']")).toHaveLength(1);

    await wrapper.setProps({
      gallerySources: [{ id: "studio", label: "Studio", target: rotatedTarget }],
    });
    expect(wrapper.findAll("[data-test='mobile-image-picker-gallery-item']")).toHaveLength(0);
    resolveRotated([{ filename: "new.png", metadata, timestamp: 2 }]);
    await flushPromises();
    expect(apiJsonTo).toHaveBeenLastCalledWith(rotatedTarget, "/api/gallery", {
      signal: expect.any(AbortSignal),
    });
    expect(
      wrapper.get("[data-test='mobile-image-picker-gallery-item']").attributes("aria-label"),
    ).toBe("Use new.png from Studio");
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
    expect(items[0]!.attributes("aria-label")).toBe("Use next.png from Selected machine");
    expect(apiJsonTo).toHaveBeenNthCalledWith(2, nextTarget, "/api/gallery", {
      signal: expect.any(AbortSignal),
    });
  });
});
