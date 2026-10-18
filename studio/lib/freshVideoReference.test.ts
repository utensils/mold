import { describe, expect, it, vi } from "vitest";
import { captureFreshVideoReference } from "./freshVideoReference";
import type { VideoNormalizationCapabilities } from "../api/referenceNormalizationTypes";
const cap: VideoNormalizationCapabilities = {
  available: true,
  protocol_version: 1,
  input_roles: ["reference_video"],
  candidate_session_path: "/api/generate/reference-normalization-sessions",
  finalize_path: "/api/generate/reference-normalization-finalize",
  candidate_digest_header: "x-mold-reference-candidate-sha256",
  max_original_file_bytes: 268435456,
  max_canonical_file_bytes: 268435456,
  max_session_bytes: 1073741824,
  accepted_mime_types: [
    "video/mp4",
    "video/quicktime",
    "video/x-m4v",
    "video/x-matroska",
  ],
  accepted_extensions: ["mp4", "mov", "m4v", "mkv"],
  accepted_video_codecs: ["h264", "hevc"],
  canonical_mime_type: "video/mp4",
};
describe("fresh typed video candidates", () => {
  it("captures original bytes/hash without probing or inventing canonical facts", async () => {
    const file = new File(["not locally decodable"], "reference.mov", {
      type: "video/quicktime",
    });
    const value = await captureFreshVideoReference(file, cap);
    expect(value.original).toMatchObject({
      mime_type: "video/quicktime",
      name: "reference.mov",
      byte_length: 21,
    });
    expect(value.original.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(Object.keys(value)).toEqual(["kind", "original", "body"]);
    expect(value.body).toBe(file);
    expect(value).not.toHaveProperty("fps");
    expect(value).not.toHaveProperty("duration_ms");
  });
  it("derives a negotiated hint for an empty MIME MKV without trusting it as decoded evidence", async () => {
    const value = await captureFreshVideoReference(
      new File(["original"], "clip.MKV"),
      cap,
    );
    expect(value.original.mime_type).toBe("video/x-matroska");
    await expect(
      captureFreshVideoReference(new File(["x"], "clip.unknown"), cap),
    ).rejects.toThrow(/container/i);
  });
  it("refuses unsupported capability, empty and oversized input before reading bytes", async () => {
    const file = new File(["original"], "clip.mov", {
      type: "video/quicktime",
    });
    const read = vi.spyOn(file, "arrayBuffer");
    await expect(
      captureFreshVideoReference(file, {
        ...cap,
        input_roles: ["source_video"],
      }),
    ).rejects.toThrow(/negotiated/i);
    await expect(
      captureFreshVideoReference(file, { ...cap, max_original_file_bytes: 2 }),
    ).rejects.toThrow(/limit/i);
    await expect(
      captureFreshVideoReference(new File([], "clip.mov"), cap),
    ).rejects.toThrow(/empty/i);
    expect(read).not.toHaveBeenCalled();
  });
});
