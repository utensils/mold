import { beforeEach, describe, expect, it, vi } from "vitest";
import { sha256 } from "@noble/hashes/sha2.js";
import { prepareH3VideoReferences } from "./prepareH3VideoReferences";
import type { MinimaxH3AuthoringState } from "./minimaxH3Authoring";
import type { ReferenceUploadCapabilities } from "../api/referenceUploads";
import { prepareNormalizedReferenceUploads } from "../api/referenceNormalization";
vi.mock("../api/referenceNormalization", () => ({
  prepareNormalizedReferenceUploads: vi.fn(),
}));
const capabilities = {
  max_file_bytes: 268435456,
  video_normalization: { max_original_file_bytes: 268435456 },
} as ReferenceUploadCapabilities;
function state(): MinimaxH3AuthoringState {
  return {
    firstFrame: null,
    lastFrame: null,
    references: [
      {
        reference: {
          kind: "image",
          mime_type: "image/png",
          width: 64,
          height: 64,
          media: { authority: "inline", data: "YWJj" },
          provenance: { name: "image.png" },
        },
      },
      {
        reference: {
          kind: "fresh_video",
          original: {
            mime_type: "video/quicktime",
            name: "clip.mov",
            byte_length: 8,
            sha256: "a".repeat(64),
          },
          body: new Blob(["original"]),
        },
      },
      {
        reference: {
          kind: "image",
          mime_type: "image/png",
          width: 64,
          height: 64,
          media: { authority: "server_path", path: "retained/image.png" },
          provenance: { sha256: "b".repeat(64) },
        },
      },
    ],
  };
}
const options = (value = state()) => ({
  state: value,
  request: { model: "minimax-h3-ref2va", prompt: "test" },
  target: { baseUrl: "https://gpu.example", apiKey: "secret" },
  expectedInstanceId: "instance",
  capabilities,
  current: () => true,
});
beforeEach(() => vi.clearAllMocks());
describe("frozen mixed normalization candidates", () => {
  it("keeps input order and existing canonical facts while uploading only immutable original bodies", () => {
    const value = state();
    prepareH3VideoReferences(options(value));
    const call = vi.mocked(prepareNormalizedReferenceUploads).mock.calls[0]![0];
    expect(
      call.candidates.map((candidate) => [candidate.index, candidate.type]),
    ).toEqual([
      [1, "canonical"],
      [2, "fresh_video"],
      [3, "canonical"],
    ]);
    expect(call.candidates[0]).toMatchObject({
      upload: true,
      byte_length: 3,
      reference: {
        media: { authority: "descriptor" },
        provenance: {
          name: "image.png",
          sha256: Array.from(sha256(new TextEncoder().encode("abc")), (byte) =>
            byte.toString(16).padStart(2, "0"),
          ).join(""),
        },
      },
    });
    expect(call.candidates[2]).toMatchObject({
      upload: false,
      reference: {
        media: { authority: "server_path", path: "retained/image.png" },
      },
    });
    expect([...call.uploadBodies.keys()]).toEqual([1, 2]);
    expect(value.references[0]!.reference.media?.authority).toBe("inline");
  });
  it("rejects a previous request lease instead of promoting its handle into a new session", () => {
    const value = state();
    value.references[0]!.reference = {
      kind: "image",
      mime_type: "image/png",
      width: 64,
      height: 64,
      media: { authority: "upload", handle: "old-secret" },
    };
    expect(() => prepareH3VideoReferences(options(value))).toThrow(
      /previous upload lease/i,
    );
    expect(prepareNormalizedReferenceUploads).not.toHaveBeenCalled();
  });
  it("bounds canonical inline bodies before decoding and allocating their payload", () => {
    expect(() =>
      prepareH3VideoReferences({
        ...options(),
        capabilities: { ...capabilities, max_file_bytes: 2 },
      }),
    ).toThrow(/limit/i);
    expect(prepareNormalizedReferenceUploads).not.toHaveBeenCalled();
  });
});
