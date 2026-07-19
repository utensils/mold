import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DOMWrapper, flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia, type Pinia } from "pinia";
import { nextTick } from "vue";
import ImagePickerModal from "./ImagePickerModal.vue";
import { useConnectionStore } from "../../stores/connection";
import { useGalleryStore, type GalleryBucket } from "../../stores/gallery";
import { useHostsStore } from "../../stores/hosts";
import type { GalleryImage } from "../../lib/api/types";

const meta = { prompt: "", model: "m", seed: 1, steps: 4, guidance: 3, width: 8, height: 8 };
const img = (filename: string, timestamp: number): GalleryImage => ({
  filename,
  metadata: meta,
  timestamp,
});

vi.mock("../../lib/ipc", () => ({
  ipc: {
    localGalleryList: vi.fn(),
    localGalleryDelete: vi.fn(),
  },
}));

const apiFetch = vi.fn();
const apiFetchTo = vi.fn();
vi.mock("../../lib/api/client", () => ({
  apiJson: (...args: unknown[]) => apiJson(...args),
  apiJsonTo: vi.fn(),
  apiFetch: (...args: unknown[]) => apiFetch(...args),
  apiFetchTo: (...args: unknown[]) => apiFetchTo(...args),
}));
const apiJson = vi.fn();

// AuthedMedia resolves thumbnails through authedMediaUrl; stub it so mounting
// the gallery grid doesn't hit the network.
vi.mock("../../lib/gallery/media", async () => {
  const actual =
    await vi.importActual<typeof import("../../lib/gallery/media")>("../../lib/gallery/media");
  return {
    ...actual,
    authedMediaUrl: vi.fn(() => Promise.resolve("blob:thumb")),
    evictMedia: vi.fn(),
    evictHostMedia: vi.fn(),
  };
});

const loadedBucket = (items: GalleryImage[]): GalleryBucket => ({
  items,
  loading: false,
  error: null,
  loaded: true,
});

describe("ImagePickerModal", () => {
  let pinia: Pinia;

  function bodyGet<T extends Element>(selector: string): DOMWrapper<T> {
    const element = document.body.querySelector<T>(selector);
    expect(element).not.toBeNull();
    return new DOMWrapper(element as T);
  }

  /** Local primary ready + hal9000 as an extra ready host, one bucket each.
   *  The mp4/gif rows must be filtered out — the engine rejects them as
   *  source/keyframe input. */
  function seedTwoHostGallery() {
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    const gallery = useGalleryStore();
    vi.spyOn(gallery, "fetchAll").mockResolvedValue();
    gallery.buckets["local"] = loadedBucket([img("a.png", 4), img("clip.mp4", 2)]);
    gallery.buckets["hal9000-7680"] = loadedBucket([img("b.jpg", 3), img("loop.gif", 1)]);
    return gallery;
  }

  beforeEach(() => {
    pinia = createPinia();
    setActivePinia(pinia);
    apiJson.mockReset();
    apiFetch.mockReset().mockResolvedValue({
      blob: () => Promise.resolve(new Blob(["x"], { type: "image/png" })),
    });
    apiFetchTo.mockReset().mockResolvedValue({
      blob: () => Promise.resolve(new Blob(["y"], { type: "image/jpeg" })),
    });
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("lists still images from every host and picks a remote one via its own host", async () => {
    const gallery = seedTwoHostGallery();
    const wrapper = mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();
    await nextTick();
    expect(gallery.fetchAll).toHaveBeenCalled();

    await bodyGet<HTMLButtonElement>("[data-test='picker-tab-gallery']").trigger("click");
    await nextTick();

    // Both hosts' still images render, newest first; the mp4 and gif are excluded.
    const thumbs = document.body.querySelectorAll("[data-test='picker-gallery-item']");
    expect(thumbs.length).toBe(2);
    const labels = Array.from(thumbs).map((t) => t.getAttribute("aria-label"));
    expect(labels).toEqual(["a.png", "b.jpg"]);

    // The remote print's bytes come from ITS host, not the primary.
    await new DOMWrapper(thumbs[1] as HTMLElement).trigger("click");
    await vi.waitFor(() => expect(wrapper.emitted("pick")).toBeTruthy());
    expect(apiFetchTo).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://hal9000:7680" }),
      "/api/gallery/image/b.jpg",
    );

    const picked = wrapper.emitted("pick")?.[0]?.[0] as Array<{ filename: string; base64: string }>;
    expect(picked).toHaveLength(1);
    expect(picked[0]!.filename).toBe("b.jpg");
    // Blob(["y"]) base64-encodes to "eQ==".
    expect(picked[0]!.base64).toBe("eQ==");
    expect(wrapper.emitted("close")).toBeTruthy();
  });

  it("labels tiles with their host when more than one gallery source exists", async () => {
    seedTwoHostGallery();
    mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();
    await bodyGet<HTMLButtonElement>("[data-test='picker-tab-gallery']").trigger("click");
    await nextTick();

    const chips = Array.from(document.body.querySelectorAll("[data-test='picker-item-host']")).map(
      (c) => c.textContent?.trim(),
    );
    expect(chips).toHaveLength(2);
    expect(chips[1]).toBe("hal9000");
  });

  it("hides host labels with a single source", async () => {
    const gallery = useGalleryStore();
    vi.spyOn(gallery, "fetchAll").mockResolvedValue();
    gallery.buckets["local"] = loadedBucket([img("a.png", 4)]);
    mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();
    await bodyGet<HTMLButtonElement>("[data-test='picker-tab-gallery']").trigger("click");
    await nextTick();

    expect(document.body.querySelectorAll("[data-test='picker-gallery-item']").length).toBe(1);
    expect(document.body.querySelector("[data-test='picker-item-host']")).toBeNull();
  });

  it("picks a This-Mac print over the local protocol when the engine is down", async () => {
    // No connection info: the "local" bucket is IPC-backed, targetOf is null,
    // and bytes must come from mold-local:// — not the (dead) primary API.
    const gallery = useGalleryStore();
    vi.spyOn(gallery, "fetchAll").mockResolvedValue();
    gallery.buckets["local"] = loadedBucket([img("a.png", 4)]);
    const localFetch = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["z"], { type: "image/png" })),
    });
    vi.stubGlobal("fetch", localFetch);

    const wrapper = mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();
    await bodyGet<HTMLButtonElement>("[data-test='picker-tab-gallery']").trigger("click");
    await nextTick();
    await bodyGet<HTMLButtonElement>("[data-test='picker-gallery-item']").trigger("click");
    await vi.waitFor(() => expect(wrapper.emitted("pick")).toBeTruthy());

    expect(localFetch).toHaveBeenCalledWith("mold-local://localhost/a.png");
    expect(apiFetch).not.toHaveBeenCalled();
    const picked = wrapper.emitted("pick")?.[0]?.[0] as Array<{ base64: string }>;
    expect(picked[0]!.base64).toBe("eg=="); // Blob(["z"])
    vi.unstubAllGlobals();
  });

  it("treats a non-OK local-protocol response as a pick error, not image bytes", async () => {
    const gallery = useGalleryStore();
    vi.spyOn(gallery, "fetchAll").mockResolvedValue();
    gallery.buckets["local"] = loadedBucket([img("a.png", 4)]);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));

    const wrapper = mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();
    await bodyGet<HTMLButtonElement>("[data-test='picker-tab-gallery']").trigger("click");
    await nextTick();
    await bodyGet<HTMLButtonElement>("[data-test='picker-gallery-item']").trigger("click");
    await flushPromises();

    expect(wrapper.emitted("pick")).toBeFalsy();
    expect(document.body.textContent).toContain("HTTP 404");
    vi.unstubAllGlobals();
  });

  it("clears a stale pick error on the next open", async () => {
    const gallery = useGalleryStore();
    vi.spyOn(gallery, "fetchAll").mockResolvedValue();
    gallery.buckets["local"] = loadedBucket([img("a.png", 4)]);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 }));

    const wrapper = mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();
    await bodyGet<HTMLButtonElement>("[data-test='picker-tab-gallery']").trigger("click");
    await nextTick();
    await bodyGet<HTMLButtonElement>("[data-test='picker-gallery-item']").trigger("click");
    await flushPromises();
    expect(document.body.textContent).toContain("HTTP 500");

    await wrapper.setProps({ open: false });
    await wrapper.setProps({ open: true });
    await nextTick();
    expect(document.body.textContent).not.toContain("HTTP 500");
    vi.unstubAllGlobals();
  });

  it("surfaces bucket fetch errors instead of claiming an empty gallery", async () => {
    const gallery = seedTwoHostGallery();
    gallery.buckets["local"] = { items: [], loading: false, error: "HTTP 500", loaded: false };
    gallery.buckets["hal9000-7680"] = { items: [], loading: false, error: null, loaded: true };
    mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();
    await bodyGet<HTMLButtonElement>("[data-test='picker-tab-gallery']").trigger("click");
    await nextTick();

    const err = bodyGet<HTMLElement>("[data-test='picker-gallery-error']");
    expect(err.text()).toContain("HTTP 500");
  });

  it("refetches the gallery on every open so late-connecting hosts appear", async () => {
    const gallery = seedTwoHostGallery();
    const wrapper = mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();
    expect(gallery.fetchAll).toHaveBeenCalledTimes(1);

    await wrapper.setProps({ open: false });
    await wrapper.setProps({ open: true });
    await flushPromises();
    expect(gallery.fetchAll).toHaveBeenCalledTimes(2);
  });

  it("rejects non-PNG/JPEG uploads with an error instead of emitting", async () => {
    const gallery = useGalleryStore();
    vi.spyOn(gallery, "fetchAll").mockResolvedValue();
    const wrapper = mount(ImagePickerModal, {
      props: { open: true },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();

    const input = bodyGet<HTMLInputElement>("[data-test='picker-upload-input']");
    expect(input.attributes("accept")).toBe("image/png,image/jpeg");

    const webp = new File(["x"], "anim.webp", { type: "image/webp" });
    Object.defineProperty(input.element, "files", { value: [webp] });
    await input.trigger("change");
    await flushPromises();

    expect(wrapper.emitted("pick")).toBeFalsy();
    const err = bodyGet<HTMLElement>("[data-test='picker-upload-error']");
    expect(err.text()).toContain("Only PNG or JPEG");
  });

  it("does not render when closed", () => {
    mount(ImagePickerModal, {
      props: { open: false },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    expect(document.body.querySelector("[data-test='picker-tab-gallery']")).toBeNull();
  });

  it("exposes a labelled modal dialog and closes on Escape", async () => {
    const gallery = useGalleryStore();
    vi.spyOn(gallery, "fetchAll").mockResolvedValue();
    const wrapper = mount(ImagePickerModal, {
      props: { open: true, title: "Keyframe image" },
      global: { plugins: [pinia] },
      attachTo: document.body,
    });
    await flushPromises();

    const dialog = bodyGet<HTMLElement>("[role='dialog']");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Keyframe image");

    await bodyGet<HTMLElement>("[data-test='picker-tab-upload']").trigger("keydown", {
      key: "Escape",
    });
    expect(wrapper.emitted("close")).toBeTruthy();
  });
});
