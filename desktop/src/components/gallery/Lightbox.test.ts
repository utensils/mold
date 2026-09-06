import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const { saveGalleryMedia, apiJsonTo } = vi.hoisted(() => ({
  saveGalleryMedia: vi.fn(),
  apiJsonTo: vi.fn(),
}));

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../../lib/ipc", () => ({
  inTauri: () => true,
  ipc: {
    getOutputDir: vi.fn().mockResolvedValue(null),
    revealOutputFile: vi.fn(),
    revealSavedMedia: vi.fn(),
  },
}));
vi.mock("../../lib/mediaSave", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/mediaSave")>()),
  saveGalleryMedia,
}));
vi.mock("../../lib/api/client", () => ({
  currentTarget: vi.fn(() => ({ baseUrl: "http://local", apiKey: "key" })),
  apiJson: vi.fn(),
  apiJsonTo,
}));

import Lightbox from "./Lightbox.vue";
import { overlayDepth } from "@ui/lib/overlayStack";
import { useContextMenuStore } from "../../stores/contextMenu";
import { useComposerStore } from "../../stores/composer";
import { useToastStore } from "../../stores/toasts";
import type { GalleryImage } from "../../lib/api/types";

const item: GalleryImage = {
  filename: "print-0001.png",
  timestamp: 1_700_000_000,
  format: "png",
  metadata: {
    prompt: "a lighthouse at dusk",
    model: "flux-dev:q8",
    seed: 42,
    steps: 4,
    guidance: 3.5,
    width: 1024,
    height: 1024,
  },
};

beforeEach(() => {
  setActivePinia(createPinia());
  saveGalleryMedia.mockReset();
  saveGalleryMedia.mockResolvedValue({
    filename: "print-0001.png",
    path: "/Users/test/Downloads/print-0001.png",
    directory: "/Users/test/Downloads",
  });
  apiJsonTo.mockResolvedValue({
    formats: ["gif", "apng"],
    gif_playback: ["loop", "bounce"],
    gif_repeat: ["forever", "once"],
  });
});

function mountLightbox(
  selectedItem: GalleryImage = item,
  video = false,
  props: Record<string, unknown> = {},
) {
  return mount(Lightbox, {
    props: { item: selectedItem, index: 0, count: 3, video, ...props },
    global: { stubs: { AuthedMedia: { template: "<div />" } } },
  });
}

describe("upscale capability", () => {
  it("hides Framewise upscale when the video host cannot execute it", () => {
    const video = { ...item, filename: "clip.mp4", format: "mp4" } as GalleryImage;
    const wrapper = mountLightbox(video, true, { upscaleEnabled: false });
    expect(wrapper.find("[data-test='lightbox-upscale']").exists()).toBe(false);
  });
});

describe("Lightbox reuse", () => {
  it("hands a one-shot back to the owner instead of prefilling it here", async () => {
    // Prefilling here called `composer.set`, which INVALIDATES retained-source
    // authority — so this button silently dropped a print's private source
    // media while the right-click item kept it. The owner runs both now.
    const metadata = {
      ...item.metadata,
      negative_prompt: "blurry",
      width: 4096,
      height: 4096,
      generation_width: 1024,
      generation_height: 1024,
      upscale_model: "real-esrgan-x4plus:fp16",
      scheduler: "ddim",
      strength: 0.6,
      loras: [{ path: "detail.safetensors", scale: 0.8 }],
    };
    const wrapper = mountLightbox({ ...item, metadata });

    await wrapper.get("[data-test='lightbox-primary-action']").trigger("click");

    expect(wrapper.emitted("reuse")).toHaveLength(1);
    expect(useComposerStore().prefill).toBeNull();
  });

  // A stitched print carries `metadata.chain` whether an author composed the
  // scenes or the server auto-chained one long clip, so the Lightbox reads it
  // as an ordinary print: one reuse door, no authoring re-entry.
  it("treats a chain-stitched print as an ordinary print", async () => {
    const stitched: GalleryImage = {
      ...item,
      filename: "clip-0001.mp4",
      format: "mp4",
      metadata: {
        ...item.metadata,
        chain: {
          stage_count: 2,
          motion_tail_frames: 8,
          stages: [
            { prompt: "a lighthouse at dusk", frames: 97, transition: "smooth", seed: "42" },
            { prompt: "the beam sweeps the bay", frames: 97, transition: "smooth", seed: "43" },
          ],
        },
      },
    };
    const wrapper = mountLightbox(stitched, true);

    const primary = wrapper.get("[data-test='lightbox-primary-action']");
    expect(primary.text()).toContain("Use these settings");
    expect(primary.text()).not.toContain("Edit clip");
    expect(wrapper.text()).not.toContain("Duplicate as new");
    expect(wrapper.find("[data-test='lightbox-duplicate-sequence']").exists()).toBe(false);

    await primary.trigger("click");
    expect(wrapper.emitted("reuse")).toHaveLength(1);

    // Save / Copy / Delete stay available for it.
    expect(wrapper.find("[data-test='save-media']").exists()).toBe(true);
    expect(wrapper.find("[data-test='lightbox-delete']").exists()).toBe(true);
    await wrapper.get("[data-test='lightbox-delete']").trigger("click");
    await wrapper.get("[data-test='lightbox-delete']").trigger("click");
    expect(wrapper.emitted("delete")).toHaveLength(1);
  });
});

describe("Lightbox metadata panel", () => {
  const richItem: GalleryImage = {
    filename: "print-0002.mp4",
    timestamp: 1_700_000_000,
    format: "mp4",
    size_bytes: 12_500_000,
    metadata: {
      generation_time_ms: 72_400,
      prompt: "a ship in a storm",
      negative_prompt: "calm seas",
      original_prompt: "a ship",
      batch_id: "batch-2026-07-20",
      batch_index: 2,
      batch_count: 3,
      model: "ltx2:q8",
      seed: 7,
      steps: 30,
      guidance: 3,
      width: 768,
      height: 512,
      strength: 0.6,
      scheduler: "ddim",
      cfg_plus: true,
      frames: 121,
      fps: 30,
      pipeline: "two-stage-hq",
      loras: [
        { path: "detail.safetensors", scale: 0.8 },
        { path: "grain.safetensors", scale: 0.5 },
      ],
      output_format: "mp4",
      version: "0.17.1",
    },
  };

  it("makes the prompt selectable and copies it from the visible control", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    const wrapper = mountLightbox();

    expect(wrapper.get("[data-test='lightbox-prompt']").attributes()).toHaveProperty(
      "data-selectable",
    );
    await wrapper.get("[data-test='copy-prompt']").trigger("click");

    expect(writeText).toHaveBeenCalledWith("a lighthouse at dusk");
    expect(useToastStore().items.at(-1)?.message).toBe("Copied");
  });

  it("renders the full embedded field set when present", () => {
    const wrapper = mountLightbox(richItem, true);
    const text = wrapper.get("aside").text();

    expect(wrapper.get("[data-test='lightbox-negative']").text()).toContain("calm seas");
    expect(wrapper.get("[data-test='lightbox-original']").text()).toContain("a ship");
    expect(wrapper.get("[data-test='lightbox-batch']").text()).toContain("2 of 3");
    expect(wrapper.get("[data-test='lightbox-batch']").text()).toContain("batch-2026-07-20");
    expect(wrapper.get("[data-test='lightbox-scheduler']").text()).toContain("ddim");
    expect(wrapper.get("[data-test='lightbox-cfg-plus']").text()).toContain("on");
    expect(wrapper.get("[data-test='lightbox-strength']").text()).toContain("0.60");
    expect(wrapper.get("[data-test='lightbox-video']").text()).toContain("121");
    expect(wrapper.get("[data-test='lightbox-video']").text()).toContain("30 fps");
    expect(wrapper.get("[data-test='lightbox-pipeline']").text()).toContain("two-stage-hq");
    expect(wrapper.get("[data-test='lightbox-took']").text()).toContain("1m 12s");
    expect(wrapper.get("[data-test='lightbox-file-size']").text()).toContain("12.5 MB");
    expect(wrapper.get("[data-test='lightbox-format']").text()).toContain("MP4");
    const loras = wrapper.findAll("[data-test='lightbox-lora']");
    expect(loras).toHaveLength(2);
    expect(loras[0]!.text()).toContain("detail.safetensors");
    expect(loras[0]!.text()).toContain("0.80");
    expect(text).toContain("mold 0.17.1");
  });

  it("omits every conditional row when its field is absent", () => {
    const wrapper = mountLightbox({ ...item, format: null, size_bytes: null });
    for (const row of [
      "lightbox-negative",
      "lightbox-original",
      "lightbox-batch",
      "lightbox-scheduler",
      "lightbox-cfg-plus",
      "lightbox-strength",
      "lightbox-video",
      "lightbox-pipeline",
      "lightbox-format",
      "lightbox-lora",
      "lightbox-identity-photo",
      "lightbox-identity",
      "lightbox-version",
    ]) {
      expect(wrapper.find(`[data-test='${row}']`).exists()).toBe(false);
    }
  });

  it("shows identity provenance — name, short digest, strength and start step", () => {
    const wrapper = mountLightbox({
      ...item,
      metadata: {
        ...item.metadata,
        id_image_name: "face.png",
        id_image_sha256: "a".repeat(64),
        id_weight: 0.8,
        id_start_step: 2,
      },
    });
    const photo = wrapper.get("[data-test='lightbox-identity-photo']");
    expect(photo.text()).toContain("face.png");
    expect(photo.text()).toContain("a".repeat(12));
    const knobs = wrapper.get("[data-test='lightbox-identity']");
    expect(knobs.text()).toContain("0.8");
    expect(knobs.text()).toContain("step 2");
  });

  it("shows the effective defaults for a print that recorded only the photo", () => {
    const wrapper = mountLightbox({
      ...item,
      metadata: { ...item.metadata, id_image_name: "face.png" },
    });
    expect(wrapper.get("[data-test='lightbox-identity-photo']").text()).toContain("face.png");
    expect(wrapper.get("[data-test='lightbox-identity']").text()).toContain("1");
  });

  it("shows a legacy single lora/lora_scale pair as a one-row stack", () => {
    const wrapper = mountLightbox({
      ...item,
      metadata: { ...item.metadata, lora: "old.safetensors", lora_scale: 0.7 },
    });
    const loras = wrapper.findAll("[data-test='lightbox-lora']");
    expect(loras).toHaveLength(1);
    expect(loras[0]!.text()).toContain("old.safetensors");
    expect(loras[0]!.text()).toContain("0.70");
  });

  it("identifies an upscaled print using its persisted generation dimensions", () => {
    const wrapper = mountLightbox({
      ...item,
      filename: "renamed-output.png",
      metadata: {
        ...item.metadata,
        width: 4096,
        height: 4096,
        generation_width: 1024,
        generation_height: 1024,
        upscale_model: "real-esrgan-x4plus",
      },
    });

    expect(wrapper.get("[data-test='upscaled-badge']").text()).toBe("Upscaled");
  });
});

describe("Lightbox a11y", () => {
  it("is a labelled modal dialog", () => {
    const wrapper = mountLightbox();
    const dialog = wrapper.get("[role='dialog']");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Picture 1 of 3");
  });

  it("labels the close and navigation controls", () => {
    const wrapper = mountLightbox();
    expect(wrapper.find("[aria-label='Close']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Previous picture']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Next picture']").exists()).toBe(true);
  });

  it("offers full image copy from the still-image context menu", async () => {
    const wrapper = mountLightbox();

    await wrapper.get('[data-test="lightbox-media"]').trigger("contextmenu");

    const menu = useContextMenuStore();
    expect(menu.visible).toBe(true);
    expect(menu.entries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "Copy image", disabled: false }),
        expect.objectContaining({ label: "Use as source", disabled: false }),
        expect.objectContaining({ label: "Copy file path" }),
      ]),
    );
    const useSource = menu.entries.find(
      (entry) => !("separator" in entry) && entry.label === "Use as source",
    );
    expect(useSource).toBeDefined();
    menu.activate(useSource!);
    expect(wrapper.emitted("useSource")).toHaveLength(1);
    wrapper.unmount();
  });
});

describe("Lightbox save action", () => {
  it("uses native save language for images and videos", () => {
    expect(mountLightbox().get("[data-test='save-media']").text()).toBe("Save a copy");
    const video: GalleryImage = { ...item, filename: "print-0001.mp4", format: "mp4" };
    const wrapper = mountLightbox(video, true);
    expect(wrapper.get("[data-test='save-media']").text()).toBe("Save a copy");
    expect(wrapper.get("[data-test='export-video']").text()).toContain("Export format");
  });

  it("confirms the exact saved file and folder", async () => {
    const wrapper = mountLightbox(item, false, {
      target: { baseUrl: "http://hal", apiKey: "secret" },
    });

    await wrapper.get("[data-test='save-media']").trigger("click");
    await vi.waitFor(() => expect(saveGalleryMedia).toHaveBeenCalledOnce());

    const toast = useToastStore().items.at(-1)!;
    expect(toast.message).toBe("Saved print-0001.png");
    expect(toast.description).toBe("To /Users/test/Downloads");
    expect(toast.action?.label).toBe("Show in folder");
  });

  it("closes export options after saving and confirms the converted filename", async () => {
    const video: GalleryImage = { ...item, filename: "print-0001.mp4", format: "mp4" };
    saveGalleryMedia.mockResolvedValueOnce({
      filename: "print-0001.gif",
      path: "/Users/test/Exports/print-0001.gif",
      directory: "/Users/test/Exports",
    });
    const wrapper = mountLightbox(video, true, {
      target: { baseUrl: "http://hal", apiKey: "secret" },
    });

    await wrapper.get("[data-test='export-video']").trigger("click");
    await vi.waitFor(() =>
      expect(wrapper.find("[data-test='video-export-dialog']").exists()).toBe(true),
    );
    await wrapper.get("[data-test='video-export-dialog'] form").trigger("submit");
    await vi.waitFor(() =>
      expect(wrapper.find("[data-test='video-export-dialog']").exists()).toBe(false),
    );

    expect(useToastStore().items.at(-1)?.message).toBe("Saved print-0001.gif");
  });
});

describe("Lightbox organization", () => {
  const organization = {
    title: "smurf 04",
    favorite: true,
    tags: ["smurf", "blue"],
    collections: ["smurfs"],
    trashedAt: null,
    purgeAt: null,
  };
  const collections = [
    { slug: "smurfs", name: "Smurfs", count: 9, hosts: [], cover: null },
    { slug: "river", name: "River studies", count: 6, hosts: [], cover: null },
  ];

  it("leads with the display title (prompt excerpt) and demotes the filename when it cannot organize", () => {
    const wrapper = mountLightbox();
    expect(wrapper.get("[data-test='lightbox-title']").text()).toBe("a lighthouse at dusk");
    expect(wrapper.get("[data-test='lightbox-filename']").text()).toBe("print-0001.png");
    expect(wrapper.find("[data-test='lightbox-favorite']").exists()).toBe(false);
    expect(wrapper.find("[data-test='lightbox-tags']").exists()).toBe(false);
    expect(wrapper.get("[data-test='lightbox-delete']").text()).toBe("Delete");
  });

  it("edits the title inline: Enter commits a rename, Escape reverts, empty clears", async () => {
    const wrapper = mountLightbox(item, false, { canOrganize: true, organization });
    const input = wrapper.get<HTMLInputElement>("[data-test='lightbox-title']");
    expect(input.element.value).toBe("smurf 04");
    expect(input.attributes("placeholder")).toBe("a lighthouse at dusk");
    await input.trigger("focus");
    await input.setValue("Smurf 05");
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("rename")).toEqual([["Smurf 05"]]);

    await input.trigger("focus");
    await input.setValue("discard me");
    await input.trigger("keydown", { key: "Escape" });
    expect(input.element.value).toBe("smurf 04");
    expect(wrapper.emitted("rename")).toHaveLength(1);

    await input.trigger("focus");
    await input.setValue("   ");
    await input.trigger("blur");
    expect(wrapper.emitted("rename")?.at(-1)).toEqual([null]);
  });

  it("rejects an invalid title inline without emitting", async () => {
    const wrapper = mountLightbox(item, false, { canOrganize: true, organization });
    const input = wrapper.get("[data-test='lightbox-title']");
    await input.trigger("focus");
    await input.setValue("x".repeat(121));
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("rename")).toBeUndefined();
    expect(wrapper.get("[data-test='lightbox-title-error']").text()).toContain("120");
  });

  it("toggles ♥, edits tags, and toggles collections through emits", async () => {
    const wrapper = mountLightbox(item, false, {
      canOrganize: true,
      organization,
      collections,
      tagSuggestions: [{ name: "outdoor", count: 2 }],
    });
    const fav = wrapper.get("[data-test='lightbox-favorite']");
    expect(fav.attributes("aria-pressed")).toBe("true");
    await fav.trigger("click");
    expect(wrapper.emitted("favorite")).toEqual([[false]]);

    const tagsEditor = wrapper.get("[data-test='lightbox-tags']");
    expect(tagsEditor.findAll("[data-test='tag-chip']").map((c) => c.find("span").text())).toEqual([
      "smurf",
      "blue",
    ]);
    await tagsEditor.findAll("[data-test='tag-remove']")[1]!.trigger("click");
    expect(wrapper.emitted("tags")).toEqual([[{ add: [], remove: ["blue"] }]]);
    const tagInput = tagsEditor.get("[data-test='tag-input']");
    await tagInput.setValue("outdoor");
    await tagInput.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("tags")?.at(-1)).toEqual([{ add: ["outdoor"], remove: [] }]);

    const picker = wrapper.get("[data-test='lightbox-collections']");
    const rows = picker.findAll("[data-test='collection-row']");
    expect(rows[0]!.attributes("aria-checked")).toBe("true");
    expect(rows[1]!.attributes("aria-checked")).toBe("false");
    await rows[1]!.trigger("click");
    expect(wrapper.emitted("collections")).toEqual([[{ slug: "river", checked: true }]]);
    await picker.get("[data-test='collection-new']").trigger("click");
    const newInput = picker.get("[data-test='collection-new-input']");
    await newInput.setValue("Halcyon");
    await newInput.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("collections")?.at(-1)).toEqual([{ name: "Halcyon", checked: true }]);
  });

  it("labels Delete as Move to trash on a trash-capable host, single press", async () => {
    const wrapper = mountLightbox(item, false, { canTrash: true });
    const button = wrapper.get("[data-test='lightbox-delete']");
    expect(button.text()).toBe("Move to trash");
    await button.trigger("click");
    expect(wrapper.emitted("delete")).toHaveLength(1);
  });

  it("shows the purge countdown with Restore / Delete forever (plain confirm) for a trashed print", async () => {
    const now = Date.now();
    const wrapper = mountLightbox(item, false, {
      canOrganize: true,
      trashed: true,
      organization: {
        ...organization,
        trashedAt: now / 1000 - 10,
        purgeAt: now / 1000 + 5 * 86_400,
      },
    });
    expect(wrapper.get("[data-test='lightbox-purge']").text()).toContain("purges in 5 d");
    expect(wrapper.find("[data-test='lightbox-delete']").exists()).toBe(false);
    await wrapper.get("[data-test='lightbox-restore']").trigger("click");
    expect(wrapper.emitted("restore")).toHaveLength(1);

    await wrapper.get("[data-test='lightbox-delete-forever']").trigger("click");
    const dialog = wrapper.get("[data-test='confirm-dialog']");
    expect(dialog.text()).toContain("Delete “smurf 04” forever?");
    expect(dialog.text()).toContain("This can't be undone.");
    expect(dialog.find("input").exists()).toBe(false);
    await wrapper.get("[data-test='confirm-accept']").trigger("click");
    expect(wrapper.emitted("deleteForever")).toHaveLength(1);
    wrapper.unmount();
  });

  it("saves under the download label — title, model, and seed", async () => {
    const wrapper = mountLightbox(item, false, {
      canOrganize: true,
      organization,
      target: { baseUrl: "http://hal", apiKey: "secret" },
    });
    await wrapper.get("[data-test='save-media']").trigger("click");
    await vi.waitFor(() => expect(saveGalleryMedia).toHaveBeenCalledOnce());
    expect(saveGalleryMedia).toHaveBeenCalledWith(
      { baseUrl: "http://hal", apiKey: "secret" },
      "print-0001.png",
      "smurf-04__flux-dev-q8__s42.png",
      null,
      false,
    );
  });
});

describe("Lightbox overlay stacking", () => {
  const organizationFor = (now: number) => ({
    title: "smurf 04",
    favorite: false,
    tags: [],
    collections: [],
    trashedAt: now / 1000 - 10,
    purgeAt: now / 1000 + 5 * 86_400,
  });

  it("Escape over the delete-forever question answers the question, not the lightbox", async () => {
    const now = Date.now();
    const wrapper = mount(Lightbox, {
      props: {
        item,
        index: 0,
        count: 3,
        video: false,
        canOrganize: true,
        trashed: true,
        organization: organizationFor(now),
      },
      global: { stubs: { AuthedMedia: { template: "<div />" } } },
      attachTo: document.body,
    });
    // The grid's own Escape handler lives on the window; the confirm above the
    // lightbox has to stop the key before it ever gets there, or the picture
    // the question is about disappears with the question.
    const heard: string[] = [];
    const witness = () => heard.push("window");
    window.addEventListener("keydown", witness);

    await wrapper.get("[data-test='lightbox-delete-forever']").trigger("click");
    expect(wrapper.find("[data-test='confirm-dialog']").exists()).toBe(true);

    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }),
    );
    await wrapper.vm.$nextTick();
    expect(wrapper.find("[data-test='confirm-dialog']").exists()).toBe(false);
    expect(heard).toEqual([]);
    expect(wrapper.emitted("deleteForever")).toBeUndefined();

    window.removeEventListener("keydown", witness);
    wrapper.unmount();
  });

  it("is itself an overlay, so anything it opens is registered above it", async () => {
    const now = Date.now();
    // Sibling tests leave their lightboxes mounted, so this reads its own
    // depth relative to whatever the register already holds.
    const before = overlayDepth();
    const wrapper = mount(Lightbox, {
      props: {
        item,
        index: 0,
        count: 3,
        video: false,
        canOrganize: true,
        trashed: true,
        organization: organizationFor(now),
      },
      global: { stubs: { AuthedMedia: { template: "<div />" } } },
      attachTo: document.body,
    });
    expect(overlayDepth()).toBe(before + 1);
    await wrapper.get("[data-test='lightbox-delete-forever']").trigger("click");
    expect(overlayDepth()).toBe(before + 2);
    wrapper.unmount();
    expect(overlayDepth()).toBe(before);
  });

  it("with no question open, Escape reaches the grid so the lightbox can close", async () => {
    const wrapper = mount(Lightbox, {
      props: { item, index: 0, count: 3, video: false },
      global: { stubs: { AuthedMedia: { template: "<div />" } } },
      attachTo: document.body,
    });
    const heard: string[] = [];
    const witness = () => heard.push("window");
    window.addEventListener("keydown", witness);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }),
    );
    expect(heard).toEqual(["window"]);
    window.removeEventListener("keydown", witness);
    wrapper.unmount();
  });
});

describe("Lightbox in the Trash", () => {
  const live = {
    title: "smurf 04",
    favorite: false,
    tags: [],
    collections: [],
    trashedAt: null,
    purgeAt: null,
  };

  function openMenu(wrapper: ReturnType<typeof mount>) {
    void wrapper.get('[data-test="lightbox-media"]').trigger("contextmenu");
    return useContextMenuStore().entries.map((entry) => ("separator" in entry ? "—" : entry.label));
  }

  /*
   * A trashed print's settings can be reused — the grid's own tile menu is
   * already gated there — but the Lightbox offered the whole reuse cluster on
   * a print that is on its way to being purged, which is neither a recipe the
   * user meant to pick up nor a photo that will still exist. Upscale and the
   * delete block already read `fromTrash`; these do now too.
   */
  it("offers the reuse actions on a live print", async () => {
    const wrapper = mountLightbox(item, false, { canOrganize: true, organization: live });
    expect(wrapper.find("[data-test='lightbox-primary-action']").exists()).toBe(true);
    expect(wrapper.find("[data-test='lightbox-use-source']").exists()).toBe(true);
    await wrapper.vm.$nextTick();
    expect(openMenu(wrapper)).toContain("Use as source");
    wrapper.unmount();
  });

  it("offers none of them once the print is in the trash", async () => {
    const now = Date.now();
    const wrapper = mountLightbox(item, false, {
      canOrganize: true,
      trashed: true,
      organization: { ...live, trashedAt: now / 1000 - 10, purgeAt: now / 1000 + 5 * 86_400 },
    });
    expect(wrapper.find("[data-test='lightbox-primary-action']").exists()).toBe(false);
    expect(wrapper.find("[data-test='lightbox-use-source']").exists()).toBe(false);
    await wrapper.vm.$nextTick();
    expect(openMenu(wrapper)).not.toContain("Use as source");
    // The trash's own actions are untouched.
    expect(wrapper.find("[data-test='lightbox-restore']").exists()).toBe(true);
    expect(wrapper.find("[data-test='lightbox-delete-forever']").exists()).toBe(true);
    wrapper.unmount();
  });

  it("gates on the row itself, not only on the Trash view", async () => {
    const trashedRow = { ...item, trashed_at: Math.floor(Date.now() / 1000) - 10 };
    const wrapper = mountLightbox(trashedRow, false, { canOrganize: true, organization: live });
    expect(wrapper.find("[data-test='lightbox-primary-action']").exists()).toBe(false);
    expect(wrapper.find("[data-test='lightbox-use-source']").exists()).toBe(false);
    await wrapper.vm.$nextTick();
    expect(openMenu(wrapper)).not.toContain("Use as source");
    wrapper.unmount();
  });
});
