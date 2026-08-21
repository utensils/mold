const RETRY_DELAYS_MS = [0, 150, 500] as const;

/** Retry an idempotent cancellation handoff before declaring it unconfirmed. */
export async function confirmCancellation(
  cancel: () => Promise<unknown>,
  wait: (delayMs: number) => Promise<void> = (delayMs) =>
    new Promise((resolve) => setTimeout(resolve, delayMs)),
): Promise<void> {
  let lastError: unknown;
  for (const delayMs of RETRY_DELAYS_MS) {
    if (delayMs > 0) await wait(delayMs);
    try {
      await cancel();
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}
