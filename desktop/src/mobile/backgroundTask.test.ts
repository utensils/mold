import { beforeEach, describe, expect, it, vi } from "vitest";

const { invoke, isNativeIOSRuntime } = vi.hoisted(() => ({
  invoke: vi.fn(),
  isNativeIOSRuntime: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke }));
vi.mock("./platform", () => ({ isNativeIOSRuntime }));

import { beginMobileBackgroundTask } from "./backgroundTask";

describe("mobile background task lease", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    isNativeIOSRuntime.mockReturnValue(true);
    invoke.mockResolvedValue("mobile-background-1");
  });

  it("starts and idempotently releases an iOS assertion", async () => {
    const lease = await beginMobileBackgroundTask("Checking generation placement");

    expect(lease.active).toBe(true);
    expect(invoke).toHaveBeenNthCalledWith(1, "begin_mobile_background_task", {
      name: "Checking generation placement",
    });

    await lease.release();
    await lease.release();

    expect(invoke).toHaveBeenCalledTimes(2);
    expect(invoke).toHaveBeenNthCalledWith(2, "end_mobile_background_task", {
      token: "mobile-background-1",
    });
  });

  it("is a no-op outside the native iOS shell", async () => {
    isNativeIOSRuntime.mockReturnValue(false);

    const lease = await beginMobileBackgroundTask("Preparing generation");
    await lease.release();

    expect(lease.active).toBe(false);
    expect(invoke).not.toHaveBeenCalled();
  });

  it("does not block generation when iOS declines the assertion", async () => {
    invoke.mockRejectedValueOnce(new Error("background time unavailable"));

    const lease = await beginMobileBackgroundTask("Preparing generation");
    await lease.release();

    expect(lease.active).toBe(false);
    expect(invoke).toHaveBeenCalledTimes(1);
  });
});
