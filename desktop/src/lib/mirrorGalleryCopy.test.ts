import { beforeEach, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({ copy: vi.fn(), fetch: vi.fn() }));
vi.mock("@studio/api/galleryCopy", async (original) => ({
  ...(await original<typeof import("@studio/api/galleryCopy")>()),
  copyGalleryPrint: mocks.copy,
}));
vi.mock("@studio/api/client", async (original) => ({
  ...(await original<typeof import("@studio/api/client")>()),
  apiFetchTo: mocks.fetch,
}));
import { mirrorGalleryCopy } from "./mirrorGalleryCopy";
const source = { baseUrl: "https://source", apiKey: "source-private" };
const local = { baseUrl: "https://local", apiKey: "local-private" };
function dependencies() {
  return {
    ensureLocal: vi.fn(async () => local),
    nativeHooks: vi.fn(() => ({})),
    storage: { read: vi.fn(async () => null), write: vi.fn(async () => undefined) },
  };
}
beforeEach(() => {
  mocks.copy.mockReset();
  mocks.fetch.mockReset();
  mocks.fetch.mockImplementation(
    async (target) =>
      new Response(
        JSON.stringify({ instance_id: target.baseUrl === source.baseUrl ? "s1" : "d1" }),
      ),
  );
  mocks.copy.mockResolvedValue({
    stage: "complete",
    destinationFilename: "actual.mp4",
    outcome: "complete",
  });
});
it("routes automatic and manual local copies through the complete shared copier and fixed journal", async () => {
  const deps = dependencies();
  await expect(
    mirrorGalleryCopy(source, "video.mp4", null, deps, {
      id: "configured-source",
      instanceId: "s1",
    }),
  ).resolves.toBe("actual.mp4");
  const options = mocks.copy.mock.calls[0]![0];
  expect(options.source).toMatchObject({
    id: "configured-source",
    instanceId: "s1",
    target: source,
  });
  expect(options.destination).toMatchObject({ id: "local", instanceId: "d1", target: local });
  expect(options.journal).toBeDefined();
  expect(options.verifySourceOffer).toBeTypeOf("function");
  mocks.copy.mockResolvedValueOnce({ stage: "sources", destinationFilename: "partial.mp4" });
  await expect(mirrorGalleryCopy(source, "video.mp4", null, deps)).rejects.toThrow("not finished");
});
it("preserves the expected listing recipe fence while forwarding exact authoritative raw data", async () => {
  const raw =
    '{"seed":18446744073709551615,"prompt":"literal\\\\n\\r\\nnext","negative_prompt":"literal\\\\t"}';
  const expected = JSON.parse(raw);
  let outputMutation = false;
  mocks.copy.mockImplementation(async (options) => {
    options.verifySourceOffer({ metadata: JSON.parse(raw), rawMetadata: raw });
    outputMutation = true;
    return { stage: "complete", destinationFilename: "actual.mp4" };
  });
  await mirrorGalleryCopy(source, "video.mp4", expected, dependencies());
  expect(outputMutation).toBe(true);
  outputMutation = false;
  await expect(
    mirrorGalleryCopy(
      source,
      "video.mp4",
      { ...expected, negative_prompt: "changed" },
      dependencies(),
    ),
  ).rejects.toThrow("settings changed");
  expect(outputMutation).toBe(false);
});
it("refuses a replaced captured source and snapshots metadata before asynchronous local startup", async () => {
  await expect(
    mirrorGalleryCopy(source, "video.mp4", null, dependencies(), {
      id: "configured",
      instanceId: "old",
    }),
  ).rejects.toThrow("installation changed");
  expect(mocks.copy).not.toHaveBeenCalled();
  const metadata = { prompt: "clicked" };
  const deps = dependencies();
  deps.ensureLocal.mockImplementation(async () => {
    metadata.prompt = "replacement";
    return local;
  });
  mocks.copy.mockImplementation(async (options) => {
    options.verifySourceOffer({
      metadata: { prompt: "replacement" },
      rawMetadata: '{"prompt":"replacement"}',
    });
    return { stage: "complete", destinationFilename: "bad.mp4" };
  });
  await expect(mirrorGalleryCopy(source, "video.mp4", metadata, deps)).rejects.toThrow(
    "settings changed",
  );
});
it("refuses current-owner changes and failed local startup without copying", async () => {
  const deps = dependencies();
  await expect(
    mirrorGalleryCopy(source, "video.mp4", null, deps, {
      id: "configured",
      instanceId: "s1",
      current: () => false,
    }),
  ).rejects.toThrow("changed");
  expect(deps.ensureLocal).not.toHaveBeenCalled();
  deps.ensureLocal.mockRejectedValueOnce(new Error("engine unavailable"));
  await expect(mirrorGalleryCopy(source, "video.mp4", null, deps)).rejects.toThrow(
    "engine unavailable",
  );
  expect(mocks.copy).not.toHaveBeenCalled();
});
it("rejects an exact raw expected recipe mismatch that rounded display integers cannot distinguish", async () => {
  const { captureMirrorSource } = await import("./mirrorGalleryCopy");
  const live = { ...source, instanceId: "s1" };
  const context = captureMirrorSource("configured", "s1", source, () => live);
  expect(context.current!()).toBe(true);
  live.instanceId = "replacement";
  expect(context.current!()).toBe(false);
  mocks.copy.mockImplementation(async (options) => {
    options.verifySourceOffer({
      metadata: { seed: Number("18446744073709551614") },
      rawMetadata: '{"seed":18446744073709551614}',
    });
    return { stage: "complete", destinationFilename: "bad.mp4" };
  });
  await expect(
    mirrorGalleryCopy(
      source,
      "video.mp4",
      { seed: Number("18446744073709551615") },
      dependencies(),
      { id: "configured", instanceId: "s1", rawMetadata: '{"seed":18446744073709551615}' },
    ),
  ).rejects.toThrow("settings changed");
});
it("aborts an in-flight native copy when its configured source changes", async () => {
  const { reactive } = await import("vue");
  const { captureMirrorSource } = await import("./mirrorGalleryCopy");
  const live = reactive({ ...source, instanceId: "s1" });
  mocks.copy.mockImplementation(
    (options) =>
      new Promise((_resolve, reject) =>
        options.signal?.addEventListener("abort", () => reject(options.signal.reason), {
          once: true,
        }),
      ),
  );
  const result = mirrorGalleryCopy(
    source,
    "video.mp4",
    null,
    dependencies(),
    captureMirrorSource("configured", "s1", source, () => live),
  );
  const rejected = expect(result).rejects.toThrow("changed");
  await vi.waitFor(() => expect(mocks.copy).toHaveBeenCalled());
  live.instanceId = "replacement";
  await rejected;
});
it("returns a completed legacy copy with a separate unavailable-input warning", async () => {
  const warning = vi.fn();
  mocks.copy.mockResolvedValueOnce({
    stage: "complete",
    destinationFilename: "legacy.mp4",
    outcome: "legacy_inputs_unavailable",
  });
  await expect(
    mirrorGalleryCopy(source, "video.mp4", null, dependencies(), {
      id: "configured",
      instanceId: "s1",
      onWarning: warning,
    }),
  ).resolves.toBe("legacy.mp4");
  expect(warning).toHaveBeenCalledWith(expect.stringContaining("original inputs"));
});
