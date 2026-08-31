import { describe, expect, it } from "vitest";
import { RetainedSourceReuseAuthority } from "./retainedSourceReuseAuthority";

describe("mobile retained source reuse authority", () => {
  it("fences late inventory after reset, new print, or source clear", () => {
    const authority = new RetainedSourceReuseAuthority<{ filename: string }>();
    const version = authority.begin();
    authority.invalidate();
    expect(authority.setIfCurrent(version, { filename: "old.png" })).toBe(false);
    expect(authority.snapshot()).toBeNull();
  });

  it("binds a relay snapshot to the exact reuse draft", () => {
    const authority = new RetainedSourceReuseAuthority<{ filename: string }>();
    const version = authority.begin();
    expect(authority.setIfCurrent(version, { filename: "old.png" })).toBe(true);
    const snapshot = authority.snapshot();
    authority.begin();
    expect(snapshot?.value.filename).toBe("old.png");
    expect(authority.isCurrent(snapshot?.version ?? -1)).toBe(false);
  });
});
