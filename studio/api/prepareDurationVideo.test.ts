import { beforeEach, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({
  stage: vi.fn(),
  should: vi.fn(),
  reuse: vi.fn(),
}));
vi.mock("./referenceUploads", () => ({
  prepareReferenceUploadBatch: mocks.stage,
  requestShouldUseReferenceUploads: mocks.should,
}));
vi.mock("./gallerySourceMedia", () => ({
  createRetainedSourceMediaReuseSession: mocks.reuse,
}));
import { prepareDurationVideo } from "./prepareDurationVideo";
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "installation",
  target: { baseUrl: "http://plato", apiKey: "private" },
};
const request = {
  duration_ms: "30000",
  generation: {
    model: "selected",
    prompt: String.raw`literal\new`,
    seed: 123,
    batch_size: 1,
  },
};
beforeEach(() => {
  Object.values(mocks).forEach((mock) => mock.mockReset());
  mocks.should.mockReturnValue(true);
});
it("binds retained sources to the exact staged sibling and retains cancellation without media bytes", async () => {
  const staged = {
    ...request.generation,
    references: [{ media: { authority: "upload" } }],
  };
  const release = vi.fn(async () => {});
  mocks.stage.mockResolvedValue({ requests: [staged], release });
  mocks.reuse.mockResolvedValue({ session_handle: "grant" });
  const prepared = await prepareDurationVideo(owner, request, {
    retainedSource: { filename: "original.mp4", memberIds: ["one"] },
  });
  expect(prepared.request.generation).toBe(staged);
  expect(mocks.reuse).toHaveBeenCalledWith(
    "original.mp4",
    ["one"],
    staged,
    owner.target,
  );
  expect(prepared.media).toEqual({ retainedMediaSession: "grant" });
  expect(release).not.toHaveBeenCalled();
  await prepared.release?.();
  expect(release).toHaveBeenCalledOnce();
});
it("uses a supplied normalized reference lease once without restaging or stripping metadata", async () => {
  const cancel = vi.fn(async () => {});
  const lease = {
    request: request.generation,
    expiresAtMs: Date.now() + 60_000,
    requestScopeSha256: "scope",
    cancel,
  };
  const prepared = await prepareDurationVideo(owner, request, {
    preparedReferenceLease: lease,
  });
  expect(mocks.stage).not.toHaveBeenCalled();
  expect(prepared.request).toBe(request);
  await prepared.release?.();
  expect(cancel).toHaveBeenCalledOnce();
});
it("releases unused uploads when retained-source preparation definitely fails", async () => {
  const release = vi.fn(async () => {});
  mocks.stage.mockResolvedValue({ requests: [request.generation], release });
  mocks.reuse.mockRejectedValue(new Error("retained source missing"));
  await expect(
    prepareDurationVideo(owner, request, {
      retainedSource: { filename: "original.mp4", memberIds: ["one"] },
    }),
  ).rejects.toThrow("retained source missing");
  expect(release).toHaveBeenCalledOnce();
});
