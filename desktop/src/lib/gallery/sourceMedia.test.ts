import { beforeEach, describe, expect, it, vi } from "vitest";
import { readGalleryMediaBase64 } from "./sourceMedia";
import type { GalleryMediaAuthority } from "./sourceMedia";
import type { MergedPrint } from "../../stores/gallery";

const { apiFetch, apiFetchTo } = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../api/client", () => ({ apiFetch, apiFetchTo }));

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
  beforeEach(() => vi.restoreAllMocks());

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
