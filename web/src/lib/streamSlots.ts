/**
 * Browser HTTP/1.1 implementations commonly allow only six connections per
 * origin. Generation SSE responses remain open for the whole render, so Mold
 * reserves two connections for gallery, queue, and download traffic.
 */
export class StreamSlotPool {
  private activeCount = 0;
  private readonly waiting: Array<() => void> = [];

  constructor(private readonly limit: number) {}

  get active(): number {
    return this.activeCount;
  }

  schedule(
    signal: AbortSignal,
    start: (release: () => void) => void,
  ): () => void {
    let state: "waiting" | "active" | "done" = "waiting";

    const release = () => {
      if (state === "done") return;
      if (state === "waiting") {
        const index = this.waiting.indexOf(begin);
        if (index >= 0) this.waiting.splice(index, 1);
      } else {
        this.activeCount = Math.max(0, this.activeCount - 1);
      }
      state = "done";
      signal.removeEventListener("abort", release);
      this.promote();
    };

    const begin = () => {
      if (state !== "waiting" || signal.aborted) {
        release();
        return;
      }
      state = "active";
      this.activeCount += 1;
      start(release);
    };

    signal.addEventListener("abort", release, { once: true });
    if (this.activeCount < this.limit) begin();
    else this.waiting.push(begin);
    return release;
  }

  private promote() {
    while (this.activeCount < this.limit && this.waiting.length > 0) {
      this.waiting.shift()?.();
    }
  }
}
