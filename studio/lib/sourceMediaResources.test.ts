import { flushPromises } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { SourcePreviewCache } from "./sourceMediaResources";
import { groupSourceMedia, type SourceMediaOwner } from "./sourceMediaBrowser";
const owner: SourceMediaOwner = {
  key: "a",
  filename: "a.png",
  title: "A",
  hostLabel: "Host",
  target: { baseUrl: "http://a", apiKey: null },
};
const member = {
  member_id: "1",
  role: "source_image",
  display_name: "a",
  size_bytes: 1,
};
describe("source resources", () => {
  it("groups only verified bytes, keeping every role and owner", () => {
    const digest = "a".repeat(64);
    const selections = [
      { owner, member: { ...member, content_sha256: digest } },
      {
        owner: { ...owner, key: "b" },
        member: { ...member, role: "keyframes", content_sha256: digest },
      },
      { owner, member: { ...member, member_id: "2" } },
    ];
    const groups = groupSourceMedia(selections);
    expect(groups).toHaveLength(2);
    expect(groups[0]!.usages).toHaveLength(2);
  });
  it("coalesces subscribers without one cancellation killing the other", async () => {
    let finish!: (blob: Blob) => void;
    const load = vi.fn(
      (_signal: AbortSignal) => new Promise<Blob>((r) => (finish = r)),
    );
    const cache = new SourcePreviewCache(16, 4);
    const a = cache.acquire("x", load),
      b = cache.acquire("x", load);
    a.release();
    await flushPromises();
    expect(load.mock.calls[0]![0].aborted).toBe(false);
    finish(new Blob(["abc"]));
    expect((await b.promise).size).toBe(3);
    b.release();
    const c = cache.acquire("x", load);
    await c.promise;
    expect(load).toHaveBeenCalledTimes(1);
    c.release();
  });
  it("bounds inflight, cancels last subscriber and evicts by bytes", async () => {
    const cache = new SourcePreviewCache(3, 1);
    let finish!: (b: Blob) => void;
    const load = vi.fn(
      (_s: AbortSignal) => new Promise<Blob>((r) => (finish = r)),
    );
    const a = cache.acquire("a", load);
    const b = cache.acquire("b", load);
    await flushPromises();
    expect(load).toHaveBeenCalledTimes(1);
    finish(new Blob(["abcd"]));
    await a.promise;
    await flushPromises();
    a.release();
    expect(load).toHaveBeenCalledTimes(2);
    b.promise.catch(() => {});
    b.release();
    expect(load.mock.calls[1]![0].aborted).toBe(true);
    expect(cache.bytes).toBe(0);
  });
});
it("does not cache a late completion after authority revocation", async () => {
  const cache = new SourcePreviewCache(16, 4);
  let finish!: (blob: Blob) => void;
  const first = cache.acquire(
    "owner\nmember",
    () => new Promise((resolve) => (finish = resolve)),
  );
  const failure = first.promise.catch((error) => error.name);
  await flushPromises();
  cache.invalidate("owner\n");
  finish(new Blob(["secret"]));
  expect(await failure).toBe("AbortError");
  await Promise.resolve();
  const next = vi.fn(async () => new Blob(["new"]));
  const lease = cache.acquire("owner\nmember", next);
  await lease.promise;
  expect(next).toHaveBeenCalledOnce();
  lease.release();
});
it("evicts least recently used preview bytes", async () => {
  const cache = new SourcePreviewCache(3, 4);
  const load = vi.fn(async () => new Blob(["ab"]));
  for (const key of ["a", "b", "a"]) {
    const lease = cache.acquire(key, load);
    await lease.promise;
    lease.release();
    await Promise.resolve();
  }
  expect(load).toHaveBeenCalledTimes(3);
  expect(cache.bytes).toBe(2);
});
it("recovers its request slots after a synchronous loader throw", async () => {
  const cache = new SourcePreviewCache(16, 1);
  const failed = cache.acquire("bad", () => {
    throw new Error("decode failed");
  });
  await expect(failed.promise).rejects.toThrow("decode failed");
  await Promise.resolve();
  const good = cache.acquire("good", async () => new Blob(["ok"]));
  expect((await good.promise).size).toBe(2);
  good.release();
  failed.release();
});
