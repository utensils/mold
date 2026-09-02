import { describe, expect, it, vi } from "vitest";
import { applyGalleryEntryAsSource, canUseGalleryEntryAsSource } from "./useAsSource";
import { newGenerateForm } from "../generateForm";
import type { MergedPrint } from "../../stores/gallery";
import type { GalleryImage } from "../api/types";

const image: GalleryImage = {
  filename: "subject.png",
  timestamp: 1,
  format: "png",
  metadata: {
    prompt: "a lighthouse",
    model: "flux2-klein",
    seed: 7,
    steps: 4,
    guidance: 1,
    width: 1024,
    height: 768,
  },
};

const entryFor = (item: GalleryImage): MergedPrint => ({
  item,
  sourceKey: "local",
  hostLabel: "This Mac",
  availableOn: [],
});

const bytes = (type: string) => async () => new Blob([new Uint8Array([65, 66, 67])], { type });

describe("applyGalleryEntryAsSource", () => {
  it("loads a still print into the composer's source image", async () => {
    const form = newGenerateForm();
    const outcome = await applyGalleryEntryAsSource(entryFor(image), form, bytes("image/png"));
    expect(outcome).toEqual({ ok: true, message: "Loaded as source" });
    expect(form.sourceImage).toBe("QUJD");
    expect(form.sourceImageName).toBe("subject.png");
    expect(form.sourceFit).toEqual({ mode: "crop-fill" });
  });

  // The Library has always allowed a rendered clip back in as LTX source
  // video; every menu that offers "Use as source" offers the same thing.
  it("loads a video print into the source-video field", async () => {
    const form = newGenerateForm();
    const clip = { ...image, filename: "clip.mp4", format: "mp4" } as GalleryImage;
    const outcome = await applyGalleryEntryAsSource(entryFor(clip), form, bytes("video/mp4"));
    expect(outcome).toEqual({ ok: true, message: "Loaded as source video" });
    expect(form.sourceVideo).toMatchObject({ filename: "clip.mp4", base64: "QUJD" });
    expect(form.sourceImage).toBeNull();
  });

  it("routes a still into MiniMax H3's ordered references", async () => {
    const form = newGenerateForm();
    form.model = "minimax-h3-ref2va:official-bf16";
    form.family = "minimax_h3";
    const outcome = await applyGalleryEntryAsSource(entryFor(image), form, bytes("image/png"));
    expect(outcome).toEqual({ ok: true, message: "Added as ordered reference" });
    expect(form.h3Authoring?.references).toHaveLength(1);
    expect(form.sourceImage).toBeNull();
  });

  it("reports a read failure instead of leaving the composer half-attached", async () => {
    const form = newGenerateForm();
    const outcome = await applyGalleryEntryAsSource(entryFor(image), form, async () => {
      throw new Error("Could not read subject.png (HTTP 404)");
    });
    expect(outcome).toEqual({ ok: false, error: "Could not read subject.png (HTTP 404)" });
    expect(form.sourceImage).toBeNull();
  });

  // An audio print has no pixels and is not a source; the menus disable the
  // item, and the rule that says so lives here with the attach itself.
  it("refuses an audio print without reading it", async () => {
    const form = newGenerateForm();
    const audio = { ...image, filename: "score.wav", format: "wav" } as GalleryImage;
    const read = vi.fn();
    const outcome = await applyGalleryEntryAsSource(entryFor(audio), form, read);
    expect(outcome.ok).toBe(false);
    expect(read).not.toHaveBeenCalled();
    expect(canUseGalleryEntryAsSource(audio)).toBe(false);
    expect(canUseGalleryEntryAsSource(image)).toBe(true);
  });
});
