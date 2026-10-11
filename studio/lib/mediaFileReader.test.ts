import { describe, expect, it, vi } from "vitest";
import { MediaFileReader, MAX_INLINE_MEDIA_BYTES } from "./mediaFileReader";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe("MediaFileReader", () => {
  it("refuses empty and oversized files before reading their bytes", async () => {
    const encode = vi.fn();
    const reader = new MediaFileReader({ encode });
    const commit = vi.fn();
    expect(
      await reader.read("sourceVideo", new File([], "empty.mp4"), commit),
    ).toMatchObject({
      kind: "refused",
      message: "Source video cannot be empty.",
    });
    const large = new File(["x"], "large.mp4");
    Object.defineProperty(large, "size", { value: MAX_INLINE_MEDIA_BYTES + 1 });
    expect(await reader.read("sourceVideo", large, commit)).toMatchObject({
      kind: "refused",
      message: "Source videos must be 64 MiB or smaller.",
    });
    expect(encode).not.toHaveBeenCalled();
    expect(commit).not.toHaveBeenCalled();
  });

  it("keeps the last committed attachment and explains a failed read", async () => {
    const reader = new MediaFileReader({
      encode: async () => {
        throw new Error("disk failure");
      },
    });
    const commit = vi.fn();
    expect(
      await reader.read("audioFile", new File(["audio"], "song.wav"), commit),
    ).toEqual({
      kind: "refused",
      message: "Couldn’t read that audio file. Try choosing it again.",
    });
    expect(commit).not.toHaveBeenCalled();
  });

  it("commits unchanged bytes only for the latest pick of the field", async () => {
    const older = deferred<string>();
    const reader = new MediaFileReader({
      encode: (file) =>
        file.name === "old.mp4" ? older.promise : Promise.resolve("new-bytes"),
    });
    const commit = vi.fn();
    const pending = reader.read(
      "sourceVideo",
      new File(["old"], "old.mp4"),
      commit,
    );
    expect(
      await reader.read("sourceVideo", new File(["new"], "new.mp4"), commit),
    ).toEqual({ kind: "committed" });
    older.resolve("old-bytes");
    expect(await pending).toEqual({ kind: "stale" });
    expect(commit).toHaveBeenCalledExactlyOnceWith({
      filename: "new.mp4",
      base64: "new-bytes",
    });
  });

  it.each(["invalidate", "dispose"] as const)(
    "does not resurrect cleared or unmounted media after %s",
    async (action) => {
      const pending = deferred<string>();
      const reader = new MediaFileReader({ encode: () => pending.promise });
      const commit = vi.fn();
      const read = reader.read(
        "extendVideo",
        new File(["x"], "clip.mp4"),
        commit,
      );
      reader[action]();
      pending.resolve("bytes");
      expect(await read).toEqual({ kind: "stale" });
      expect(commit).not.toHaveBeenCalled();
    },
  );

  it("checks the aggregate budget again and commits atomically across fields", async () => {
    const pending = deferred<string>();
    let used = 0;
    const reader = new MediaFileReader({
      encode: () => pending.promise,
      remainingBytes: () => 5 - used,
      budgetMessage: "Media budget exceeded.",
    });
    const commit = vi.fn(() => {
      used += 4;
    });
    const video = reader.read(
      "sourceVideo",
      new File(["1234"], "clip.mp4"),
      commit,
    );
    const audio = reader.read(
      "audioFile",
      new File(["1234"], "audio.wav"),
      commit,
    );
    pending.resolve("bytes");
    expect(await video).toEqual({ kind: "committed" });
    expect(await audio).toEqual({
      kind: "refused",
      message: "Media budget exceeded.",
    });
    expect(commit).toHaveBeenCalledTimes(1);
  });
});
