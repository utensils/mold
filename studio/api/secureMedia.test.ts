import { beforeEach, expect, it, vi } from "vitest";
const lookup = vi.hoisted(() => vi.fn());
vi.mock("./secureStore", () => ({
  secureHostForOrigin: lookup,
  isKnownSecureCredential: async () => false,
}));
import { configureSecureMediaAdapter, secureMediaUrl } from "./secureMedia";
const open = vi.fn();
beforeEach(() => {
  lookup.mockResolvedValue(null);
  open.mockReset();
  configureSecureMediaAdapter({ open, close: vi.fn() });
});
it("leaves an explicit Regular stream on its direct HTTP media path", async () => {
  expect(
    await secureMediaUrl("http://203.0.113.7:7680/api/gallery/image/clip.mp4", {
      "x-mold-connection-mode": "regular",
      "x-api-key": "regular-key",
    }),
  ).toBeNull();
  expect(open).not.toHaveBeenCalled();
});
it("never downgrades a missing Secure media pairing", async () => {
  await expect(
    secureMediaUrl("http://host.test/api/gallery/image/clip.mp4"),
  ).rejects.toThrow(/securely/);
  expect(open).not.toHaveBeenCalled();
});
it("applies proxy and secure-credential boundaries before returning a direct media URL", async () => {
  await expect(
    secureMediaUrl(
      "https://x.mold-link.urandom.io/api/gallery/image/clip.mp4",
      { "x-mold-connection-mode": "regular" },
    ),
  ).rejects.toThrow(/proxy/);
  lookup.mockResolvedValue({ apiKey: "secure-key", endpoints: [] });
  await expect(
    secureMediaUrl("http://host.test/api/gallery/image/clip.mp4", {
      "x-mold-connection-mode": "regular",
      authorization: "Bearer secure-key",
    }),
  ).rejects.toThrow(/Secure credential/);
  expect(open).not.toHaveBeenCalled();
});
