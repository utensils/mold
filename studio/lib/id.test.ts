import { describe, expect, it, vi } from "vitest";
import { createUuid } from "./id";

describe("createUuid", () => {
  it("uses the native UUID implementation when the browser exposes it", () => {
    const randomUUID = vi.fn(() => "native-id");
    expect(createUuid({ randomUUID })).toBe("native-id");
    expect(randomUUID).toHaveBeenCalledOnce();
  });

  it("creates an RFC 4122 v4 UUID when randomUUID is unavailable", () => {
    const getRandomValues = vi.fn((bytes: Uint8Array) => {
      bytes.fill(0);
      return bytes;
    });
    expect(createUuid({ getRandomValues })).toBe(
      "00000000-0000-4000-8000-000000000000",
    );
    expect(getRandomValues).toHaveBeenCalledOnce();
  });
});
