import { describe, expect, it } from "vitest";
import { base64ToBlob, blobToBase64 } from "./base64";

describe("base64 helpers", () => {
  it("round-trips ASCII bytes through Blob → base64 → Blob", async () => {
    const input = new Blob(["hello, mold"], { type: "text/plain" });
    const b64 = await blobToBase64(input);
    expect(b64).toBe("aGVsbG8sIG1vbGQ=");

    const roundTripped = base64ToBlob(b64, "text/plain");
    const text = await roundTripped.text();
    expect(text).toBe("hello, mold");
    expect(roundTripped.type).toBe("text/plain");
  });

  it("preserves binary bytes (including high-bit) across the chunked encode path", async () => {
    // Exercise the 0x8000 chunking in blobToBase64 — span more than one chunk
    // and include every possible byte value.
    const bytes = new Uint8Array(0x8000 * 2 + 17);
    for (let i = 0; i < bytes.length; i++) bytes[i] = i & 0xff;
    const input = new Blob([bytes]);

    const b64 = await blobToBase64(input);
    const out = base64ToBlob(b64, "application/octet-stream");
    const outBytes = new Uint8Array(await out.arrayBuffer());

    expect(outBytes.length).toBe(bytes.length);
    expect(outBytes).toEqual(bytes);
  });

  it("encodes the empty blob as the empty string", async () => {
    const b64 = await blobToBase64(new Blob([]));
    expect(b64).toBe("");
    const back = base64ToBlob(b64, "image/png");
    expect(back.size).toBe(0);
  });

  it("emits raw base64 with no data-URI prefix", async () => {
    // Regression: server rejects GenerateRequest.source_image when the
    // client sends `data:image/png;base64,...` instead of bare bytes.
    const b64 = await blobToBase64(new Blob([new Uint8Array([0, 1, 2, 3])]));
    expect(b64.startsWith("data:")).toBe(false);
    expect(b64).toMatch(/^[A-Za-z0-9+/]+=*$/);
  });
});
