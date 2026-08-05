import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

const { sendNativeNotification, sendNotification } = vi.hoisted(() => ({
  sendNativeNotification: vi.fn(),
  sendNotification: vi.fn(),
}));

vi.mock("./ipc", () => ({
  inTauri: () => true,
  ipc: { sendNativeNotification },
}));

vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn().mockResolvedValue(true),
  requestPermission: vi.fn().mockResolvedValue("granted"),
  sendNotification,
}));

import { notifyGenerated, notifyUpdateAvailable } from "./notify";

describe("desktop notifications", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    sendNativeNotification.mockReset();
    sendNotification.mockReset();
  });

  it("uses the native notification path when it handles the app icon", async () => {
    sendNativeNotification.mockResolvedValue(true);

    notifyGenerated("a deer at sunrise", "mold-deer-42.png");

    await vi.waitFor(() =>
      expect(sendNativeNotification).toHaveBeenCalledWith(
        "Generated — a deer at sunrise",
        undefined,
        { kind: "gallery", filename: "mold-deer-42.png" },
      ),
    );
    expect(sendNotification).not.toHaveBeenCalled();
  });

  it("falls back to the Tauri plugin when the native path is unavailable", async () => {
    sendNativeNotification.mockResolvedValue(false);

    notifyGenerated("a deer at sunrise", "mold-deer-42.png");

    await vi.waitFor(() =>
      expect(sendNotification).toHaveBeenCalledWith({ title: "Generated — a deer at sunrise" }),
    );
  });

  it("notifies backgrounded users when an update is available", async () => {
    sendNativeNotification.mockResolvedValue(true);

    notifyUpdateAvailable("0.18.0");

    await vi.waitFor(() =>
      expect(sendNativeNotification).toHaveBeenCalledWith(
        "Mold 0.18.0 is available",
        "Open Mold to update and restart.",
        undefined,
      ),
    );
  });

  it("keeps notification delivery failures best effort", async () => {
    sendNativeNotification.mockResolvedValue(false);
    sendNotification.mockImplementationOnce(() => {
      throw new Error("notification center unavailable");
    });

    notifyGenerated("a deer at sunrise");

    await vi.waitFor(() => expect(sendNotification).toHaveBeenCalledOnce());
  });
});
