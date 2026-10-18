import { describe, expect, it, vi } from "vitest";
import { missingModelReferenceRetry } from "./missingModelReferenceRetry";
import type { GenerateRequest } from "./api/types";
const request: GenerateRequest = {
  model: "minimax-h3-ref2va",
  width: 64,
  height: 64,
  steps: 1,
  prompt: "test",
  references: [
    {
      kind: "image",
      mime_type: "image/png",
      width: 64,
      height: 64,
      media: { authority: "upload", handle: "expired-secret" },
    },
  ],
};
describe("missing-model retry preparation", () => {
  it("offers download without retaining or replaying a finalized reference lease", () => {
    const cancel = vi.fn(async () => {});
    const value = missingModelReferenceRetry({
      model: request.model,
      request,
      requestOptions: {
        preparedReferenceLease: {
          request,
          expiresAtMs: 1,
          requestScopeSha256: "a".repeat(64),
          cancel,
        },
      },
    });
    expect(value.resumeAfterPull).toBe(false);
    expect(value.request).not.toHaveProperty("references");
    expect(value.requestOptions).toEqual({});
    expect(JSON.stringify(value)).not.toContain("expired-secret");
    expect(request.references?.[0]?.media.authority).toBe("upload");
  });
  it("preserves ordinary canonical pull-and-resume behavior", () => {
    const value = {
      request: { model: "flux2", prompt: "test", width: 64, height: 64, steps: 1 },
      requestOptions: {},
      resumeAfterPull: true,
    };
    expect(missingModelReferenceRetry(value)).toBe(value);
  });
});
