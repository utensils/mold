import { describe, expect, it } from "vitest";
import {
  filterSourceMedia,
  sourceHostIdentity,
  type SourceMediaSelection,
} from "./sourceMediaBrowser";
const source = (host: string, kind = "image"): SourceMediaSelection => ({
  owner: {
    key: host,
    filename: "print.png",
    title: "Print",
    hostLabel: "Same name",
    target: { baseUrl: host, apiKey: "key" },
    instanceId: "instance",
  },
  member: {
    member_id: host,
    role: "source_image",
    display_name: "Original",
    size_bytes: 100,
    media_kind: kind,
    content_sha256: "a".repeat(64),
  },
});
describe("source library filters", () => {
  it("deduplicates identical bytes across hosts while retaining exact filtered usages", () => {
    const a = source("https://a"),
      b = source("https://b");
    expect(filterSourceMedia([a, b], {}).map((g) => g.usages.length)).toEqual([
      2,
    ]);
    expect(
      filterSourceMedia([a, b], { host: sourceHostIdentity(b.owner) })[0]
        ?.usages,
    ).toEqual([b]);
  });
  it("combines kind and case-insensitive text without guessing unknown identities", () => {
    const a = source("https://a"),
      b = source("https://b", "audio");
    delete b.member.content_sha256;
    expect(
      filterSourceMedia([a, b], { kind: "audio", search: " ORIGINAL " })[0]
        ?.usages,
    ).toEqual([b]);
    expect(filterSourceMedia([a, b], { kind: "video" })).toEqual([]);
  });
  it("separates matching host labels and rotated credentials", () => {
    const a = source("https://a");
    expect(sourceHostIdentity(a.owner)).not.toBe(
      sourceHostIdentity({
        ...a.owner,
        target: { ...a.owner.target, apiKey: "new" },
      }),
    );
  });
});

it("preserves several roles and frame indices within one owning print", () => {
  const a = source("https://a");
  const b = {
    ...a,
    member: {
      ...a.member,
      member_id: "end",
      role: "keyframes",
      frame_index: 48,
    },
  };
  const c = {
    ...a,
    member: {
      ...a.member,
      member_id: "start",
      role: "keyframes",
      frame_index: 0,
    },
  };
  expect(filterSourceMedia([a, b, c], {})[0]?.usages).toEqual([a, b, c]);
});
