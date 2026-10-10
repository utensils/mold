/** Transfer ends at the server's worker-dispatch boundary (Running).
 * Older servers retain their existing Held-only atomic protocol. */
export function queueTransferEligible(
  state: string,
  preRenderTransfer: boolean,
): boolean {
  return (
    state === "held" ||
    (preRenderTransfer && (state === "queued" || state === "paused"))
  );
}
