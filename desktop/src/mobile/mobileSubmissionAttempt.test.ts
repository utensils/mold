import { describe, expect, it, vi } from "vitest";
import type { MobileBackgroundTaskLease } from "./backgroundTask";
import { MobileSubmissionAttempts } from "./mobileSubmissionAttempt";

function backgroundTask(): MobileBackgroundTaskLease & { release: ReturnType<typeof vi.fn> } {
  return {
    active: true,
    release: vi.fn(() => Promise.resolve()),
  };
}

describe("mobile generation submission attempt", () => {
  it("releases its background task exactly once across repeated cleanup", async () => {
    const lease = backgroundTask();
    const attempt = new MobileSubmissionAttempts().begin(lease);

    await attempt.releaseOwnedResources();
    await attempt.releaseOwnedResources();

    expect(lease.release).toHaveBeenCalledTimes(1);
  });

  it("stops owning the background task after admission handoff", async () => {
    const lease = backgroundTask();
    const attempt = new MobileSubmissionAttempts().begin(lease);

    expect(attempt.handoffBackgroundTask()).toBe(lease);
    await attempt.releaseOwnedResources();

    expect(lease.release).not.toHaveBeenCalled();
  });

  it("supersedes and aborts an earlier attempt", () => {
    const attempts = new MobileSubmissionAttempts();
    const first = attempts.begin(backgroundTask());
    const firstSignal = first.signal;

    const second = attempts.begin(backgroundTask());

    expect(first.isCurrent()).toBe(false);
    expect(firstSignal.aborted).toBe(true);
    expect(second.isCurrent()).toBe(true);
  });

  it("invalidates and aborts the active attempt", () => {
    const attempts = new MobileSubmissionAttempts();
    const attempt = attempts.begin(backgroundTask());
    const signal = attempt.signal;

    attempts.invalidate();

    expect(attempt.isCurrent()).toBe(false);
    expect(signal.aborted).toBe(true);
  });
});
