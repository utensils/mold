import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Lightbox from "./Lightbox.vue";
import { __resetGalleryMediaForTests } from "../../lib/galleryMedia";
import type { GalleryImage } from "../../types";

const item: GalleryImage = {
  filename: "print.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "a lighthouse at dusk",
    model: "flux-dev:fp16",
    seed: 4242,
    steps: 20,
    guidance: 3.5,
    width: 1024,
    height: 768,
    version: "test",
  },
};

function setViewportWidth(px: number) {
  Object.defineProperty(window, "innerWidth", {
    value: px,
    configurable: true,
    writable: true,
  });
}

function mountWide(props: Record<string, unknown> = {}) {
  setViewportWidth(1200);
  return mount(Lightbox, {
    props: {
      item,
      index: 1,
      total: 5,
      hasPrev: true,
      hasNext: true,
      muted: true,
      ...props,
    },
    global: { stubs: { Transition: false } },
  });
}

describe("Lightbox (desktop two-pane)", () => {
  beforeEach(() => {
    setViewportWidth(1200);
    document.body.style.overflow = "";
  });

  it("renders the print details panel", () => {
    const wrapper = mountWide();
    const text = wrapper.text();
    expect(text).toContain("Print details");
    expect(text).toContain("2 / 5");
    expect(text).toContain("a lighthouse at dusk");
    expect(text).toContain("flux-dev");
    expect(text).toContain("4242");
    expect(text).toContain("1024");
    expect(text).toContain("Reuse these settings");
    expect(text).toContain("Use as source");
    expect(text).toContain("Download");
  });

  it("offers format export for MP4 videos without replacing download", () => {
    const wrapper = mountWide({
      item: { ...item, filename: "rain.mp4", format: "mp4" },
    });
    expect(wrapper.get("[data-test='export-video']").text()).toContain(
      "Export format",
    );
    expect(wrapper.text()).toContain("Download");
  });

  it("renders complete generation metadata and copies prompt and seed", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    const wrapper = mountWide({
      item: {
        ...item,
        size_bytes: 1_572_864,
        format: "webp",
        metadata: {
          ...item.metadata,
          original_prompt: "a lighthouse",
          negative_prompt: "fog",
          scheduler: "euler-ancestral",
          loras: [{ path: "cinematic.safetensors", scale: 0.75 }],
        },
      },
    });
    const text = wrapper.text();
    expect(text).toContain("20");
    expect(text).toContain("3.5");
    expect(text).toContain("euler-ancestral");
    expect(text).toContain("cinematic.safetensors · 0.75");
    expect(text).toContain("fog");
    expect(text).toContain("a lighthouse");
    expect(text).toContain("1.5 MiB");
    expect(text).toContain("WEBP");

    await wrapper.get('[data-test="copy-prompt"]').trigger("click");
    await wrapper.get('[data-test="copy-seed"]').trigger("click");
    expect(wrapper.get(".lb__prompt").attributes()).toHaveProperty(
      "data-selectable",
    );
    expect(writeText).toHaveBeenNthCalledWith(1, "a lighthouse at dusk");
    expect(writeText).toHaveBeenNthCalledWith(2, "4242");
  });

  it("shows the recorded runtime pipeline for an LTX video", () => {
    const wrapper = mountWide({
      item: {
        ...item,
        filename: "ltx-print.mp4",
        format: "mp4",
        metadata: {
          ...item.metadata,
          model: "ltx-2.3-22b-dev:fp8",
          frames: 97,
          fps: 24,
          pipeline: "two-stage",
        },
      },
    });

    expect(wrapper.get("[data-test='lightbox-pipeline']").text()).toContain(
      "two-stage",
    );
  });

  it("navigates with arrow keys and closes on Escape", async () => {
    const wrapper = mountWide();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("next")).toHaveLength(1);
    expect(wrapper.emitted("prev")).toHaveLength(1);
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits prev/next from the circle buttons", async () => {
    const wrapper = mountWide();
    await wrapper.find(".lb__nav--prev").trigger("click");
    await wrapper.find(".lb__nav--next").trigger("click");
    expect(wrapper.emitted("prev")).toHaveLength(1);
    expect(wrapper.emitted("next")).toHaveLength(1);
  });

  it("emits reuse and use-source with the item", async () => {
    const wrapper = mountWide();
    await wrapper.find(".lb__reuse").trigger("click");
    await wrapper.findAll(".lb__quiet")[0]!.trigger("click");
    expect((wrapper.emitted("reuse")![0]![0] as GalleryImage).filename).toBe(
      "print.png",
    );
    expect(
      (wrapper.emitted("use-source")![0]![0] as GalleryImage).filename,
    ).toBe("print.png");
  });

  it("opens the Library action menu when the media is right-clicked", async () => {
    const wrapper = mountWide();
    await wrapper.get(".lb__stage").trigger("contextmenu", {
      clientX: 120,
      clientY: 240,
    });

    expect(wrapper.emitted("context-menu")?.[0]?.[0]).toEqual({
      item,
      x: 120,
      y: 240,
    });
  });

  it("makes cached editing primary and keeps duplication explicit for sequence prints", async () => {
    const wrapper = mountWide({ isSequence: true, canEditSequence: true });
    expect(
      wrapper.get("[data-test='lightbox-primary-action']").text(),
    ).toContain("Edit sequence");
    expect(
      wrapper.get("[data-test='lightbox-duplicate-sequence']").text(),
    ).toBe("Duplicate as new");

    await wrapper.get("[data-test='lightbox-primary-action']").trigger("click");
    expect(wrapper.emitted("edit-sequence")).toHaveLength(1);
    expect(wrapper.emitted("reuse")).toBeUndefined();

    await wrapper
      .get("[data-test='lightbox-duplicate-sequence']")
      .trigger("click");
    expect(wrapper.emitted("reuse")).toHaveLength(1);
  });

  it("exposes Upscale and Delete via the overflow menu", async () => {
    const wrapper = mountWide();
    expect(wrapper.find(".lb__menu").exists()).toBe(false);
    await wrapper.find(".lb__morebtn").trigger("click");
    const items = wrapper.findAll(".lb__menu button");
    expect(items.map((b) => b.text())).toEqual(["Upscale…", "Delete"]);
    await items[1]!.trigger("click");
    expect((wrapper.emitted("delete")![0]![0] as GalleryImage).filename).toBe(
      "print.png",
    );
  });

  it("labels and emits Framewise upscale for an existing video", async () => {
    const video = { ...item, filename: "existing-clip.mp4", format: "mp4" };
    const wrapper = mountWide({ item: video });
    await wrapper.find(".lb__morebtn").trigger("click");
    const upscale = wrapper.findAll(".lb__menu button")[0]!;
    expect(upscale.text()).toBe("Framewise upscale…");
    await upscale.trigger("click");
    expect(wrapper.emitted("upscale")?.[0]?.[0]).toEqual(video);
  });

  it("hides video upscale when the origin host cannot execute it", async () => {
    const video = { ...item, filename: "clip.mp4", format: "mp4" };
    const wrapper = mountWide({ item: video, upscaleEnabled: false });
    await wrapper.find(".lb__morebtn").trigger("click");
    expect(
      wrapper.findAll(".lb__menu button").map((button) => button.text()),
    ).not.toContain("Framewise upscale…");
  });

  it("disables navigation at the ends", () => {
    const wrapper = mountWide({ hasPrev: false, hasNext: false });
    expect(
      (wrapper.find(".lb__nav--prev").element as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (wrapper.find(".lb__nav--next").element as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});

describe("Lightbox organization (title · ♥ · tags · collections · trash)", () => {
  beforeEach(() => setViewportWidth(1200));

  it("leads with the editable title and demotes the filename to a mono row", () => {
    const wrapper = mountWide({
      item: { ...item, title: "Lighthouse, v2" },
      canOrganize: true,
    });
    expect(wrapper.get("[data-test='title-text']").text()).toBe(
      "Lighthouse, v2",
    );
    expect(wrapper.get("[data-test='print-filename']").text()).toBe(
      "print.png",
    );
  });

  it("falls back to the prompt excerpt as the title placeholder", () => {
    const wrapper = mountWide({ canOrganize: true });
    expect(wrapper.get("[data-test='title-text']").text()).toBe(
      "a lighthouse at dusk",
    );
  });

  it("commits a title on Enter and reverts on Escape", async () => {
    const wrapper = mountWide({ canOrganize: true });
    await wrapper.get("[data-test='title-edit']").trigger("click");
    const input = wrapper.get<HTMLInputElement>("[data-test='title-input']");
    await input.setValue("  Harbor light  ");
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("rename")?.[0]).toEqual([item, "Harbor light"]);

    await wrapper.get("[data-test='title-edit']").trigger("click");
    const again = wrapper.get<HTMLInputElement>("[data-test='title-input']");
    await again.setValue("discarded");
    await again.trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("rename")).toHaveLength(1);
    expect(wrapper.find("[data-test='title-input']").exists()).toBe(false);
    // Escape inside the title never closes the viewer.
    expect(wrapper.emitted("close")).toBeUndefined();
  });

  it("refuses an over-long title inline instead of emitting it", async () => {
    const wrapper = mountWide({ canOrganize: true });
    await wrapper.get("[data-test='title-edit']").trigger("click");
    const input = wrapper.get<HTMLInputElement>("[data-test='title-input']");
    await input.setValue("x".repeat(121));
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("rename")).toBeUndefined();
    expect(wrapper.get("[data-test='title-error']").text()).toContain("120");
  });

  it("toggles the favorite and edits tags and collections through emits", async () => {
    const tagged = { ...item, favorite: false, tags: ["blue"] };
    const wrapper = mountWide({
      item: tagged,
      canOrganize: true,
      tagSuggestions: [
        { name: "outdoor", count: 3 },
        { name: "blue", count: 2 },
      ],
      collections: [
        { slug: "smurfs", name: "Smurfs", checked: true },
        { slug: "rivers", name: "Rivers", checked: false },
      ],
    });
    await wrapper.get("[data-test='favorite-toggle']").trigger("click");
    expect(wrapper.emitted("favorite")?.[0]).toEqual([tagged, true]);

    const tagInput = wrapper.get<HTMLInputElement>("[data-test='tag-input']");
    await tagInput.trigger("focus");
    // Suggestions skip tags the print already has.
    expect(
      wrapper.findAll("[data-test='tag-suggestion']").map((b) => b.text()),
    ).toEqual(["outdoor3"]);
    await tagInput.setValue("#Keep");
    await tagInput.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("add-tag")?.[0]).toEqual([tagged, "#Keep"]);
    await wrapper.get("[data-test='tag-remove']").trigger("click");
    expect(wrapper.emitted("remove-tag")?.[0]).toEqual([tagged, "blue"]);

    const toggles = wrapper.findAll("[data-test='collection-toggle']");
    await toggles[1]!.setValue(true);
    expect(wrapper.emitted("set-collection")?.[0]).toEqual([
      tagged,
      "rivers",
      true,
    ]);
    await wrapper.get("[data-test='collection-new']").trigger("click");
    expect(wrapper.emitted("new-collection")).toHaveLength(1);
  });

  it("hides every organization control on a host that cannot organize", () => {
    const wrapper = mountWide({ canOrganize: false });
    expect(wrapper.find("[data-test='title-edit']").exists()).toBe(false);
    expect(wrapper.find("[data-test='favorite-toggle']").exists()).toBe(false);
    expect(wrapper.find("[data-test='tag-input']").exists()).toBe(false);
    expect(wrapper.find("[data-test='collection-toggle']").exists()).toBe(
      false,
    );
  });

  it("names the download with the title, model, and seed grammar", () => {
    const wrapper = mountWide({
      item: { ...item, title: "Harbor Light!", hostId: "origin" },
      canOrganize: true,
    });
    expect(wrapper.get("a[download]").attributes("download")).toBe(
      "harbor-light__flux-dev-fp16__s4242.png",
    );
  });

  it("labels the overflow action Trash on a trash-capable host", async () => {
    const wrapper = mountWide({ canTrash: true });
    await wrapper.find(".lb__morebtn").trigger("click");
    expect(wrapper.findAll(".lb__menu button").map((b) => b.text())).toEqual([
      "Upscale…",
      "Trash",
    ]);
  });

  it("in the trash offers Restore and Delete forever with the purge countdown", async () => {
    const wrapper = mountWide({
      item: {
        ...item,
        trashed_at: 1_700_000_000,
        purge_at: Math.floor(Date.now() / 1000) + 5 * 86_400,
      },
      inTrash: true,
      canTrash: true,
      canOrganize: true,
    });
    expect(wrapper.text()).toContain("Purges in 5 d");
    expect(wrapper.find("[data-test='lightbox-primary-action']").exists()).toBe(
      false,
    );
    await wrapper.get("[data-test='lightbox-restore']").trigger("click");
    expect(wrapper.emitted("restore")?.[0]?.[0]).toMatchObject({
      filename: "print.png",
    });
    await wrapper.get("[data-test='lightbox-delete-forever']").trigger("click");
    expect(wrapper.emitted("delete-forever")?.[0]?.[0]).toMatchObject({
      filename: "print.png",
    });
    // Title / tags stay editable in the trash; nothing destructive hides.
    expect(wrapper.find("[data-test='title-edit']").exists()).toBe(true);
  });
});

describe("Lightbox (mobile full-screen)", () => {
  it("renders the full-screen viewer with metadata chips", () => {
    setViewportWidth(480);
    const wrapper = mount(Lightbox, {
      props: {
        item,
        index: 0,
        total: 3,
        hasPrev: false,
        hasNext: true,
        muted: true,
      },
      global: { stubs: { Transition: false } },
    });
    expect(wrapper.find(".lb__full").exists()).toBe(true);
    expect(wrapper.find(".lb__card").exists()).toBe(false);
    const text = wrapper.text();
    expect(text).toContain("a lighthouse at dusk");
    expect(text).toContain("seed 4242");
    expect(text).toContain("Reuse these settings");
  });
});

describe("Lightbox multi-host media", () => {
  const STUDIO = {
    id: "studio-7680",
    name: "studio",
    url: "http://studio:7680",
    apiKey: "studio-secret-key",
  };
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    setViewportWidth(1200);
    localStorage.clear();
    __resetGalleryMediaForTests();
    localStorage.setItem("mold.web.hosts.v1", JSON.stringify([STUDIO]));
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    localStorage.clear();
    vi.restoreAllMocks();
  });

  function mockFetch(handler: (url: string, init?: RequestInit) => unknown) {
    globalThis.fetch = vi.fn(async (url: string, init?: RequestInit) =>
      handler(url, init),
    ) as never;
  }

  it("keeps an origin print on the same-origin URL", async () => {
    mockFetch(() => {
      throw new Error("should not reach the network");
    });
    const wrapper = mountWide({
      item: { ...item, hostId: "origin", hostLabel: "this server" },
    });
    await flushPromises();
    expect(wrapper.find("img").attributes("src")).toBe(
      "/api/gallery/image/print.png",
    );
  });

  it("streams a keyed remote print through a media-token ticket", async () => {
    mockFetch((url) => {
      expect(url).toBe("http://studio:7680/api/gallery/media-token");
      return {
        ok: true,
        status: 200,
        json: async () => ({
          token: "tkt",
          expires_at: 1900,
          auth_required: true,
        }),
      };
    });
    const wrapper = mountWide({
      item: { ...item, hostId: "studio-7680", hostLabel: "studio" },
    });
    await flushPromises();

    const src = wrapper.find("img").attributes("src") ?? "";
    const url = new URL(src);
    expect(url.origin + url.pathname).toBe(
      "http://studio:7680/api/gallery/image/print.png",
    );
    expect(url.searchParams.get("media_token")).toBe("tkt");
    // The durable key never rides in a URL.
    expect(src).not.toContain("studio-secret-key");
  });

  it("names the owning host in the details panel", async () => {
    mockFetch(() => ({
      ok: true,
      status: 200,
      json: async () => ({
        token: "tkt",
        expires_at: 1900,
        auth_required: true,
      }),
    }));
    const wrapper = mountWide({
      item: { ...item, hostId: "studio-7680", hostLabel: "studio" },
    });
    await flushPromises();
    expect(wrapper.text()).toContain("studio");
  });

  it("does not fall back to the origin when the owning host is gone", async () => {
    mockFetch(() => {
      throw new Error("must not fetch the origin");
    });
    const wrapper = mountWide({
      item: { ...item, hostId: "forgotten", hostLabel: "old studio" },
    });
    await flushPromises();
    expect(wrapper.find("img").exists()).toBe(false);
    expect(wrapper.text()).toContain("old studio isn't connected anymore");
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("shows the can't-stream state for video on a ticket-less host", async () => {
    mockFetch((url) => {
      if (url.endsWith("/media-token")) return { ok: false, status: 404 };
      throw new Error("must not buffer a whole video");
    });
    const clip: GalleryImage = {
      ...item,
      filename: "clip.mp4",
      format: "mp4",
    };
    const wrapper = mountWide({
      item: { ...clip, hostId: "studio-7680", hostLabel: "studio" },
    });
    await flushPromises();

    expect(wrapper.find("video").exists()).toBe(false);
    expect(wrapper.text()).toContain("Can't stream this clip");
  });
});

// ── Face-identity provenance (PuLID, #1224) ─────────────────────────────
//
// Metadata records the photo's NAME and digest, never the face itself, so the
// aside is the only place a viewer can see which likeness a print was
// conditioned on and how strongly.
describe("Lightbox identity provenance", () => {
  const identityItem: GalleryImage = {
    ...item,
    metadata: {
      ...item.metadata,
      id_image_name: "ada.png",
      id_image_sha256: "9f".repeat(32),
      id_weight: 1.25,
      id_start_step: 2,
    },
  };

  it("shows the photo name, short digest, strength and start step", () => {
    const wrapper = mountWide({ item: identityItem });
    const row = wrapper.get("[data-test='lightbox-identity']").text();
    expect(row).toContain("ada.png");
    expect(row).toContain("9f9f9f9f9f9f");
    expect(row).toContain("1.25");
    expect(row).toContain("2");
  });

  it("says nothing for a print that carried no identity photo", () => {
    const wrapper = mountWide();
    expect(wrapper.find("[data-test='lightbox-identity']").exists()).toBe(
      false,
    );
  });

  it("carries the same provenance on the phone layout", () => {
    setViewportWidth(500);
    const wrapper = mount(Lightbox, {
      props: {
        item: identityItem,
        index: 0,
        total: 1,
        hasPrev: false,
        hasNext: false,
        muted: true,
      },
      global: { stubs: { Transition: false } },
    });
    expect(wrapper.get("[data-test='lightbox-identity']").text()).toContain(
      "ada.png",
    );
  });
});
