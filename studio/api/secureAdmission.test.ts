import { afterEach, expect, it, vi } from "vitest";
import { PreAdmissionProxyBusy, retrySecureAdmission } from "./secureAdmission";
afterEach(() => vi.useRealTimers());
it("retries only busy admission up to three times", async () => {
  vi.useFakeTimers();
  const open = vi
    .fn()
    .mockRejectedValueOnce(new PreAdmissionProxyBusy())
    .mockRejectedValueOnce(new PreAdmissionProxyBusy())
    .mockResolvedValue("admitted");
  const result = retrySecureAdmission(open);
  await vi.advanceTimersByTimeAsync(1000);
  await expect(result).resolves.toBe("admitted");
  expect(open).toHaveBeenCalledTimes(3);
  const alwaysBusy = vi.fn().mockRejectedValue(new PreAdmissionProxyBusy());
  const exhausted = expect(retrySecureAdmission(alwaysBusy)).rejects.toThrow(
    /busy/,
  );
  await vi.advanceTimersByTimeAsync(1000);
  await exhausted;
  expect(alwaysBusy).toHaveBeenCalledTimes(3);
});
it("does not retry authentication errors and cancels before another attempt", async () => {
  vi.useFakeTimers();
  const refused = vi
    .fn()
    .mockRejectedValue(new Error("authentication refused"));
  await expect(retrySecureAdmission(refused)).rejects.toThrow(/authentication/);
  expect(refused).toHaveBeenCalledTimes(1);
  const controller = new AbortController();
  const busy = vi.fn().mockRejectedValue(new PreAdmissionProxyBusy());
  const cancelled = expect(
    retrySecureAdmission(busy, controller.signal),
  ).rejects.toThrow(/cancelled/);
  await vi.advanceTimersByTimeAsync(1);
  controller.abort(new Error("cancelled"));
  await vi.advanceTimersByTimeAsync(1000);
  await cancelled;
  expect(busy).toHaveBeenCalledTimes(1);
});
it("ends admission attempts within fifteen seconds", async () => {
  vi.useFakeTimers();
  const pending = vi.fn(
    (signal: AbortSignal) =>
      new Promise((_, reject) =>
        signal.addEventListener("abort", () => reject(signal.reason)),
      ),
  );
  const expired = expect(retrySecureAdmission(pending)).rejects.toThrow(
    /deadline/,
  );
  await vi.advanceTimersByTimeAsync(15000);
  await expired;
  expect(pending).toHaveBeenCalledTimes(1);
});
