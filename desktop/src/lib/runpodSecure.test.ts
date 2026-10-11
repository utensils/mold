import { beforeEach, describe, expect, it, vi } from "vitest";
import { provisionSecureRunPod, requireSecureRunPod } from "./runpodSecure";

const f = vi.hoisted(() => ({
  create: vi.fn(),
  volumeOrigin: vi.fn(),
  remember: vi.fn(),
  pairing: vi.fn(),
  save: vi.fn(),
  lookup: vi.fn(),
  identity: vi.fn(),
  close: vi.fn(),
  free: vi.fn(),
  binding: vi.fn(),
}));
vi.mock("./ipc", () => ({
  ipc: {
    runpodCreate: f.create,
    runpodPairing: f.pairing,
    runpodVolumeOrigin: f.volumeOrigin,
    runpodRememberPairing: f.remember,
  },
}));
vi.mock("@studio/api/secureStore", () => ({
  deviceIdentity: f.identity,
  saveSecureHost: f.save,
  secureHostForOrigin: f.lookup,
}));
vi.mock("@studio/api/secureCrypto", () => ({ binding: f.binding }));
const publicBinding = {
  signingKey: "public-signing",
  noiseKey: "public-noise",
  signature: "binding-signature",
};
const profile = { namespace: "paired", origins: ["https://pod123-7680.proxy.runpod.net"] };

describe("encrypted RunPod provisioning", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    f.identity.mockResolvedValue({ close: f.close, free: f.free });
    f.binding.mockReturnValue(publicBinding);
    f.lookup.mockResolvedValue(null);
    f.volumeOrigin.mockResolvedValue(null);
    f.remember.mockResolvedValue(undefined);
    f.create.mockResolvedValue({ pod: { id: "pod123" }, profile });
    f.pairing.mockResolvedValue(profile);
    f.save.mockResolvedValue(undefined);
  });
  it("passes only the existing public device binding and persists pairing before returning", async () => {
    const input = { name: "test" } as never;
    await expect(provisionSecureRunPod(input)).resolves.toEqual({ id: "pod123" });
    expect(f.create).toHaveBeenCalledWith(input, publicBinding, null);
    expect(f.save).toHaveBeenCalledWith(profile);
    expect(f.close).toHaveBeenCalledOnce();
    expect(f.free).toHaveBeenCalledOnce();
  });
  it("reuses an existing secure profile when switching or resuming a pod", async () => {
    f.lookup.mockResolvedValue(profile);
    await requireSecureRunPod("pod123");
    expect(f.remember).toHaveBeenCalledWith("pod123", publicBinding, profile);
    expect(f.pairing).not.toHaveBeenCalled();
    expect(f.save).not.toHaveBeenCalled();
  });
  it("recovers a created pod's pairing through the canonical encrypted store", async () => {
    await requireSecureRunPod("pod123");
    expect(f.pairing).toHaveBeenCalledWith("pod123", publicBinding);
    expect(f.save).toHaveBeenCalledWith(profile);
  });
  it("refuses an unknown legacy pod with a pairing step instead of an unencrypted connection", async () => {
    f.pairing.mockResolvedValue(null);
    await expect(requireSecureRunPod("pod123")).rejects.toThrow("Pair this instance");
    expect(f.save).not.toHaveBeenCalled();
    expect(f.close).toHaveBeenCalledOnce();
  });
  it("passes the current canonical pairing when replacing known-volume compute", async () => {
    f.volumeOrigin.mockResolvedValue("https://old-7680.proxy.runpod.net");
    f.lookup.mockResolvedValue(profile);
    const input = { networkVolumeId: "volume1" } as never;
    await provisionSecureRunPod(input);
    expect(f.lookup).toHaveBeenCalledWith("https://old-7680.proxy.runpod.net");
    expect(f.create).toHaveBeenCalledWith(input, publicBinding, profile);
  });
  it("never retries paid creation when saving its pairing fails", async () => {
    f.save.mockRejectedValue(new Error("disk"));
    await expect(provisionSecureRunPod({} as never)).rejects.toThrow("created");
    expect(f.create).toHaveBeenCalledOnce();
    expect(f.close).toHaveBeenCalledOnce();
  });
});
