export interface SourceFitTarget {
  width: number;
  height: number;
}

type CacheEntry<T> = {
  key: string;
  value: Promise<T>;
};

async function contentHash(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
  }

  // Web Crypto is unavailable on plain-HTTP LAN origins. Keep caching usable
  // there with two independently seeded FNV-1a passes plus the byte length.
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (const byte of bytes) {
    first = Math.imul(first ^ byte, 0x01000193);
    second = Math.imul(second ^ byte, 0x01000193);
  }
  return `${bytes.length.toString(16)}:${(first >>> 0).toString(16)}:${(second >>> 0).toString(16)}`;
}

/**
 * Per-composer, single-entry memoization for source preprocessing.
 *
 * Upscale and fit are separate layers so resolution/policy/mask changes only
 * invalidate the cheap fit result. Host is intentionally absent from the
 * upscale key: upscaler output is deterministic, so a route change can reuse
 * bytes already produced by another host. A cache miss still runs on the
 * resolved generation host and preserves same-host model auto-downloads.
 * Pending promises are cached too, allowing concurrent batch siblings to
 * share one preprocessing pass.
 */
export class SourceFitPreprocessCache {
  private upscaleEntry: CacheEntry<string> | null = null;
  private fitEntry: CacheEntry<unknown> | null = null;

  async upscale(
    source: string,
    model: string,
    produce: () => Promise<string>,
  ): Promise<string> {
    const key = `${await contentHash(source)}:${model}`;
    if (this.upscaleEntry?.key === key) return this.upscaleEntry.value;

    const value = produce();
    this.upscaleEntry = { key, value };
    try {
      return await value;
    } catch (error) {
      if (this.upscaleEntry?.value === value) this.upscaleEntry = null;
      throw error;
    }
  }

  async fit<T>(
    source: string,
    mask: string | null,
    policy: unknown,
    target: SourceFitTarget,
    produce: () => Promise<T>,
  ): Promise<T> {
    const [sourceHash, maskHash] = await Promise.all([
      contentHash(source),
      mask === null ? Promise.resolve("none") : contentHash(mask),
    ]);
    const key = `${sourceHash}:${maskHash}:${JSON.stringify(policy)}:${target.width}x${target.height}`;
    if (this.fitEntry?.key === key) return this.fitEntry.value as Promise<T>;

    const value = produce();
    this.fitEntry = { key, value };
    try {
      return await value;
    } catch (error) {
      if (this.fitEntry?.value === value) this.fitEntry = null;
      throw error;
    }
  }

  clear(): void {
    this.upscaleEntry = null;
    this.fitEntry = null;
  }
}
