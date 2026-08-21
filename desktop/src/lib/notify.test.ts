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

import {
  notifyChainFinished,
  notifyGenerated,
  notifyGenerationFailed,
  notifyPulled,
  notifyPullFailed,
  notifyUpdateAvailable,
} from "./notify";
import { notificationRoute } from "./notificationAction";

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
        { kind: "updates" },
      ),
    );
  });

  it.each([
    [
      () => notifyGenerated("a deer at sunrise"),
      "Generated — a deer at sunrise",
      { kind: "gallery" },
    ],
    [() => notifyGenerationFailed("host offline"), "Generation failed", { kind: "create" }],
    [() => notifyChainFinished(81), "Chain finished · 81 frames", { kind: "gallery" }],
    [() => notifyPulled("flux-dev:q4"), "Pulled flux-dev:q4", { kind: "models" }],
    [
      () => notifyPullFailed("flux-dev:q4", "disk full"),
      "Couldn't pull flux-dev:q4",
      { kind: "models" },
    ],
  ])("gives %s an intentional click destination", async (dispatch, title, action) => {
    sendNativeNotification.mockResolvedValue(true);

    dispatch();

    await vi.waitFor(() => expect(sendNativeNotification).toHaveBeenCalled());
    expect(sendNativeNotification.mock.calls.at(-1)?.[0]).toBe(title);
    expect(sendNativeNotification.mock.calls.at(-1)?.[2]).toEqual(action);
  });

  it("maps every native action through the internal route allowlist", () => {
    expect(notificationRoute({ kind: "gallery", filename: "print.png" })).toEqual({
      path: "/library",
      query: { print: "print.png" },
    });
    expect(notificationRoute({ kind: "gallery" })).toEqual({ path: "/library" });
    expect(notificationRoute({ kind: "create" })).toEqual({ path: "/create" });
    expect(notificationRoute({ kind: "models" })).toEqual({ path: "/models" });
    expect(notificationRoute({ kind: "updates" })).toEqual({
      path: "/settings",
      query: { section: "updates" },
    });
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
