/**
 * Which held prints this session has already offered a missing-model pull for.
 *
 * The offer must fire once per hold, not once per poll — and it must be
 * FORGOTTEN when the print stops being held. Remembering a job id forever
 * leaked one entry per print for the life of the tab, and, worse, silently
 * suppressed the offer for a resumed print that was parked again for the same
 * missing model: the user got one offer and then nothing.
 */
export class HeldPullOffers {
  private readonly offered = new Set<string>();

  /** True the first time this print is offered a pull; false while the same
   * hold stands. */
  claim(jobId: string): boolean {
    if (this.offered.has(jobId)) return false;
    this.offered.add(jobId);
    return true;
  }

  /** Keep only the prints that are still held. Anything else settled, was
   * resumed, or left the rail. */
  retain(heldJobIds: Iterable<string>): void {
    const held = new Set(heldJobIds);
    for (const jobId of [...this.offered]) {
      if (!held.has(jobId)) this.offered.delete(jobId);
    }
  }
}
