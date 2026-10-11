import { blobToBase64 } from "./base64";

/** Matches the server's per-input inline video/audio bound. */
export const MAX_INLINE_MEDIA_BYTES = 64 * 1024 * 1024;
export type MediaFileField = "sourceVideo" | "extendVideo" | "audioFile";
export interface ReadMediaFile {
  filename: string;
  base64: string;
}
type ReadResult =
  { kind: "committed" | "stale" } | { kind: "refused"; message: string };

const MESSAGES: Record<
  MediaFileField,
  { empty: string; oversized: string; failed: string }
> = {
  sourceVideo: {
    empty: "Source video cannot be empty.",
    oversized: "Source videos must be 64 MiB or smaller.",
    failed: "Couldn’t read that source video. Try choosing it again.",
  },
  extendVideo: {
    empty: "The video to continue cannot be empty.",
    oversized: "Videos to continue must be 64 MiB or smaller.",
    failed: "Couldn’t read that video. Try choosing it again.",
  },
  audioFile: {
    empty: "Conditioning audio cannot be empty.",
    oversized: "Conditioning audio must be 64 MiB or smaller.",
    failed: "Couldn’t read that audio file. Try choosing it again.",
  },
};

/** Owns validation and asynchronous pick lifetime, never the authoring form.
 * The synchronous commit keeps an optional aggregate budget authoritative when
 * different fields finish reading in the same turn. Removing media, changing
 * draft identity and unmounting must invalidate their pending reads. */
export class MediaFileReader {
  private epochs = new Map<MediaFileField, number>();
  private disposed = false;

  constructor(
    private readonly options: {
      encode?: (file: File) => Promise<string>;
      remainingBytes?: (field: MediaFileField) => number;
      budgetMessage?: string;
    } = {},
  ) {}

  invalidate(field?: MediaFileField): void {
    for (const key of field
      ? [field]
      : (Object.keys(MESSAGES) as MediaFileField[])) {
      this.epochs.set(key, (this.epochs.get(key) ?? 0) + 1);
    }
  }

  dispose(): void {
    this.disposed = true;
    this.invalidate();
  }

  async read(
    field: MediaFileField,
    file: File,
    commit: (media: ReadMediaFile) => void,
  ): Promise<ReadResult> {
    this.invalidate(field);
    const epoch = this.epochs.get(field);
    const current = () => !this.disposed && this.epochs.get(field) === epoch;
    const messages = MESSAGES[field];
    if (!current()) return { kind: "stale" };
    if (file.size === 0) return { kind: "refused", message: messages.empty };
    if (file.size > MAX_INLINE_MEDIA_BYTES)
      return { kind: "refused", message: messages.oversized };
    const budgetError = () =>
      this.options.remainingBytes &&
      file.size > this.options.remainingBytes(field)
        ? {
            kind: "refused" as const,
            message:
              this.options.budgetMessage ??
              "Combined generation media is too large.",
          }
        : null;
    const initial = budgetError();
    if (initial) return initial;
    let base64: string;
    try {
      base64 = await (this.options.encode ?? blobToBase64)(file);
    } catch {
      return current()
        ? { kind: "refused", message: messages.failed }
        : { kind: "stale" };
    }
    if (!current()) return { kind: "stale" };
    const remaining = budgetError();
    if (remaining) return remaining;
    commit({ filename: file.name, base64 });
    return { kind: "committed" };
  }
}
