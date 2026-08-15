import { describe, expect, it, vi } from "vitest";
import {
  metadataReferencesSource,
  restoreEditImages,
  restoreH3Boundaries,
  restoreSourceImage,
  sha256HexOfBase64,
} from "./sourceRestore";
import type { OutputMetadata } from "./api/types";

const baseMeta: OutputMetadata = {
  prompt: "p",
  model: "flux-dev:q8",
  seed: 1,
  steps: 4,
  guidance: 3.5,
  width: 8,
  height: 8,
};

describe("restoreSourceImage", () => {
  it("prefers the local stash by sha256", async () => {
    const deps = {
      stashGet: vi.fn().mockResolvedValue("STASHED"),
      galleryLookup: vi.fn(),
    };
    const meta = {
      ...baseMeta,
      source_image_sha256: "a".repeat(64),
      source_image_name: "pic.png",
    };
    const restored = await restoreSourceImage(meta, deps);
    expect(restored).toEqual({ base64: "STASHED", filename: "pic.png" });
    expect(deps.galleryLookup).not.toHaveBeenCalled();
  });

  it("falls back to a gallery filename match when the stash misses", async () => {
    const deps = {
      stashGet: vi.fn().mockResolvedValue(null),
      galleryLookup: vi.fn().mockResolvedValue("FROM_GALLERY"),
    };
    const meta = {
      ...baseMeta,
      source_image_sha256: "a".repeat(64),
      source_image_name: "mold-flux-1-2.png",
    };
    const restored = await restoreSourceImage(meta, deps);
    expect(restored).toEqual({ base64: "FROM_GALLERY", filename: "mold-flux-1-2.png" });
    expect(deps.galleryLookup).toHaveBeenCalledWith("mold-flux-1-2.png");
  });

  it("returns null when every lookup misses, and dep failures count as misses", async () => {
    const deps = {
      stashGet: vi.fn().mockRejectedValue(new Error("ipc down")),
      galleryLookup: vi.fn().mockResolvedValue(null),
    };
    const meta = { ...baseMeta, source_image_sha256: "a".repeat(64), source_image_name: "x.png" };
    expect(await restoreSourceImage(meta, deps)).toBeNull();
  });

  it("does nothing for metadata without provenance keys (old prints)", async () => {
    const deps = { stashGet: vi.fn(), galleryLookup: vi.fn() };
    expect(await restoreSourceImage(baseMeta, deps)).toBeNull();
    expect(deps.stashGet).not.toHaveBeenCalled();
    expect(deps.galleryLookup).not.toHaveBeenCalled();
  });
});

describe("metadataReferencesSource", () => {
  it("is true only when a restore key exists", () => {
    expect(metadataReferencesSource(baseMeta)).toBe(false);
    // strength alone (old servers) is not attemptable — no key to look up.
    expect(metadataReferencesSource({ ...baseMeta, strength: 0.75 })).toBe(false);
    expect(metadataReferencesSource({ ...baseMeta, source_image_name: "x.png" })).toBe(true);
    expect(metadataReferencesSource({ ...baseMeta, source_image_sha256: "a".repeat(64) })).toBe(
      true,
    );
    expect(metadataReferencesSource({ ...baseMeta, edit_image_sha256s: ["a".repeat(64)] })).toBe(
      true,
    );
  });
});

describe("restoreEditImages", () => {
  it("restores Qwen edit images from the local stash in their recorded order", async () => {
    const stashGet = vi.fn().mockResolvedValueOnce("TARGET").mockResolvedValueOnce("REFERENCE");
    await expect(
      restoreEditImages(
        { ...baseMeta, edit_image_sha256s: ["a".repeat(64), "b".repeat(64)] },
        { stashGet, galleryLookup: vi.fn() },
      ),
    ).resolves.toEqual({ images: ["TARGET", "REFERENCE"], missing: 0 });
  });

  it("restores the target and every still-available reference", async () => {
    const stashGet = vi.fn().mockResolvedValueOnce("TARGET").mockResolvedValueOnce(null);
    await expect(
      restoreEditImages(
        { ...baseMeta, edit_image_sha256s: ["a".repeat(64), "b".repeat(64)] },
        { stashGet, galleryLookup: vi.fn() },
      ),
    ).resolves.toEqual({ images: ["TARGET"], missing: 1 });
  });

  it("never promotes a reference into the Target role when the target is missing", async () => {
    const stashGet = vi.fn().mockResolvedValueOnce(null).mockResolvedValueOnce("REFERENCE");
    await expect(
      restoreEditImages(
        { ...baseMeta, edit_image_sha256s: ["a".repeat(64), "b".repeat(64)] },
        { stashGet, galleryLookup: vi.fn() },
      ),
    ).resolves.toEqual({ images: [], missing: 1 });
  });
});

describe("restoreH3Boundaries", () => {
  const h3Meta: OutputMetadata = {
    ...baseMeta,
    model: "minimax-h3-fl2va:official-bf16",
    frames: 124,
    source_image_name: "opening.png",
    source_image_sha256: "a".repeat(64),
  };

  it("restores the first frame from the stash by sha", async () => {
    const deps = {
      stashGet: vi.fn().mockResolvedValue("STASHED"),
      galleryLookup: vi.fn(),
    };
    const restored = await restoreH3Boundaries(h3Meta, deps);
    expect(restored.firstFrame).toEqual({
      base64: "STASHED",
      filename: "opening.png",
    });
    expect(restored.lastFrame).toBeNull();
    expect(restored.missing).toBe(0);
  });

  it("falls back to a cross-host gallery filename match", async () => {
    const deps = {
      stashGet: vi.fn().mockResolvedValue(null),
      galleryLookup: vi.fn().mockResolvedValue("FROM_GALLERY"),
    };
    const restored = await restoreH3Boundaries(h3Meta, deps);
    expect(restored.firstFrame).toEqual({
      base64: "FROM_GALLERY",
      filename: "opening.png",
    });
    expect(deps.galleryLookup).toHaveBeenCalledWith("opening.png");
  });

  it("restores the closing frame from the exact final-frame keyframe", async () => {
    const deps = {
      stashGet: vi.fn(async (sha: string) => (sha === "b".repeat(64) ? "CLOSING" : null)),
      galleryLookup: vi.fn(async (name: string) =>
        name === "opening.png" ? "OPENING" : null,
      ),
    };
    const restored = await restoreH3Boundaries(
      {
        ...h3Meta,
        keyframes: [{ frame: 123, name: "closing.png", sha256: "b".repeat(64) }],
      },
      deps,
    );
    expect(restored.firstFrame?.base64).toBe("OPENING");
    expect(restored.lastFrame).toEqual({
      base64: "CLOSING",
      filename: "closing.png",
    });
    expect(restored.missing).toBe(0);
  });

  it("counts keyed slots it could not resolve", async () => {
    const deps = {
      stashGet: vi.fn().mockResolvedValue(null),
      galleryLookup: vi.fn().mockResolvedValue(null),
    };
    const restored = await restoreH3Boundaries(
      {
        ...h3Meta,
        keyframes: [{ frame: 123, name: "closing.png", sha256: "b".repeat(64) }],
      },
      deps,
    );
    expect(restored.firstFrame).toBeNull();
    expect(restored.lastFrame).toBeNull();
    expect(restored.missing).toBe(2);
  });

  it("ignores keyframes that are not the closing frame", async () => {
    const deps = {
      stashGet: vi.fn().mockResolvedValue(null),
      galleryLookup: vi.fn().mockResolvedValue(null),
    };
    const {
      source_image_name: _name,
      source_image_sha256: _sha,
      ...unkeyed
    } = h3Meta;
    const restored = await restoreH3Boundaries(
      {
        ...unkeyed,
        keyframes: [{ frame: 0, name: "opening.png", sha256: "b".repeat(64) }],
      },
      deps,
    );
    expect(restored.firstFrame).toBeNull();
    expect(restored.lastFrame).toBeNull();
    expect(restored.missing).toBe(0);
  });
});

describe("sha256HexOfBase64", () => {
  it("hashes the DECODED bytes — matching the server's hash of the wire image", async () => {
    // base64("abc") = "YWJj"; sha256("abc") is the classic test vector.
    expect(await sha256HexOfBase64("YWJj")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});
