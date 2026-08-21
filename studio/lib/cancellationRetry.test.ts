import { describe, expect, it, vi } from "vitest";
import { confirmCancellation } from "./cancellationRetry";

describe("confirmCancellation", () => {
  it("retries an idempotent cancellation until the server confirms it", async () => {
    const cancel = vi
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(new Error("offline"))
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue(undefined);
    const wait = vi.fn(async () => undefined);

    await confirmCancellation(cancel, wait);

    expect(cancel).toHaveBeenCalledTimes(3);
    expect(wait).toHaveBeenCalledTimes(2);
  });

  it("rejects when cancellation remains unconfirmed", async () => {
    const cancel = vi.fn(async () => {
      throw new Error("offline");
    });

    await expect(
      confirmCancellation(cancel, async () => undefined),
    ).rejects.toThrow("offline");
    expect(cancel).toHaveBeenCalledTimes(3);
  });
});
