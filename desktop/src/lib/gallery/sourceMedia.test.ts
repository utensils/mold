import { beforeEach, describe, expect, it, vi } from "vitest";
import { readGalleryMediaBase64 } from "./sourceMedia";
import type { GalleryMediaAuthority } from "./sourceMedia";
import type { MergedPrint } from "../../stores/gallery";

const { apiFetch, apiFetchTo, inTauri, fetchGalleryMedia } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
  inTauri: vi.fn(() => false),
  fetchGalleryMedia: vi.fn(),
}));
vi.mock("../api/client", () => ({ apiFetch, apiFetchTo }));
vi.mock("../ipc", () => ({ inTauri, ipc: { fetchGalleryMedia } }));

const entry = {
  sourceKey: "local",
  hostLabel: "This Mac",
  availableOn: [],
  item: {
    filename: "source.png",
    timestamp: 1,
    metadata: {
      prompt: "",
      model: "flux",
      seed: 1,
      steps: 1,
      guidance: 1,
      width: 64,
      height: 64,
      version: "test",
    },
  },
} as MergedPrint;

describe("gallery source-media authority", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    apiFetch.mockReset();
    apiFetchTo.mockReset();
    inTauri.mockReturnValue(false);
    fetchGalleryMedia.mockReset();
  });

  it("reads a host row through the native client in the desktop app", async () => {
    inTauri.mockReturnValue(true);
    fetchGalleryMedia.mockResolvedValue(new Uint8Array([65, 66, 67]).buffer);
    const target = { baseUrl: "http://hal9000:7680", apiKey: null };
    const gallery: GalleryMediaAuthority = {
      mediaSourceOf: () => "host",
      targetOf: () => target,
    };

    await expect(
      readGalleryMediaBase64({ ...entry, sourceKey: "hal9000-7680" }, gallery),
    ).resolves.toBe("QUJD");
    expect(fetchGalleryMedia).toHaveBeenCalledWith(target, "source.png");
    expect(apiFetchTo).not.toHaveBeenCalled();
  });

  it("falls back to the authenticated HTTP route when the native read refuses", async () => {
    inTauri.mockReturnValue(true);
    fetchGalleryMedia.mockRejectedValue(new Error("The gallery file is unexpectedly large."));
    apiFetchTo.mockResolvedValue(new Response(new Uint8Array([65, 66, 67]), { status: 200 }));
    const target = { baseUrl: "http://hal9000:7680", apiKey: null };
    const gallery: GalleryMediaAuthority = {
      mediaSourceOf: () => "host",
      targetOf: () => target,
    };

    await expect(
      readGalleryMediaBase64({ ...entry, sourceKey: "hal9000-7680" }, gallery),
    ).resolves.toBe("QUJD");
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/source.png");
  });

  it("reads native-only This-Mac media through mold-local without HTTP fallback", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(new Uint8Array([65, 66, 67]), { status: 200 }));
    const gallery: GalleryMediaAuthority = {
      mediaSourceOf: () => "local",
      targetOf: () => null,
    };

    await expect(readGalleryMediaBase64(entry, gallery)).resolves.toBe("QUJD");
    expect(fetchMock).toHaveBeenCalledWith("mold-local://localhost/source.png");
    expect(apiFetch).not.toHaveBeenCalled();
    expect(apiFetchTo).not.toHaveBeenCalled();
  });

  it("reads a host row from its exact authenticated origin", async () => {
    apiFetchTo.mockResolvedValue(new Response(new Uint8Array([65, 66, 67]), { status: 200 }));
    const target = { baseUrl: "http://plato:7680", apiKey: "secret" };
    const gallery: GalleryMediaAuthority = {
      mediaSourceOf: () => "host",
      targetOf: () => target,
    };

    await expect(
      readGalleryMediaBase64({ ...entry, sourceKey: "plato-7680" }, gallery),
    ).resolves.toBe("QUJD");
    expect(apiFetchTo).toHaveBeenCalledWith(target, "/api/gallery/image/source.png");
  });

  it("rejects an error body before it can become invalid source base64", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("missing", { status: 404 }));
    const gallery: GalleryMediaAuthority = {
      mediaSourceOf: () => "local",
      targetOf: () => null,
    };

    await expect(readGalleryMediaBase64(entry, gallery)).rejects.toThrow(
      "Could not read source.png (HTTP 404)",
    );
  });
});
