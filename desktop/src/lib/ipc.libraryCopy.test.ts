import { afterEach, beforeEach, expect, it, vi } from "vitest";
const native = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke: native }));
import { ipc, nativeLibraryCopyHooks } from "./ipc";
const source = {
  id: "source",
  label: "Source",
  instanceId: "source-installation",
  target: { baseUrl: "https://source", apiKey: "source-key", connectionMode: "regular" as const },
};
const destination = {
  id: "destination",
  label: "Destination",
  instanceId: "destination-installation",
  target: { baseUrl: "https://destination", apiKey: "dest-key", connectionMode: "secure" as const },
};
const offer = {
  archive_identity_sha256: "a".repeat(64),
  output_sha256: "b".repeat(64),
  output_size_bytes: 400 * 1024 * 1024,
  rawMetadata: '{"seed":18446744073709551615}',
  metadata: {},
  members: [],
};
beforeEach(() => {
  native.mockReset();
  Object.defineProperty(window, "__TAURI_INTERNALS__", { configurable: true, value: {} });
});
afterEach(() => {
  delete (window as any).__TAURI_INTERNALS__;
});
it("wires native transfer stages without source bytes crossing IPC and preserves connection authority", async () => {
  native.mockResolvedValue("copy.png");
  const hooks = nativeLibraryCopyHooks();
  expect(hooks.nativeImportOutput).toBeTypeOf("function");
  expect(hooks.nativeTransferSources).toBeTypeOf("function");
  expect(hooks.nativeMatchesOutput).toBeTypeOf("function");
  await hooks.nativeImportOutput!(source, destination, "print.png", offer);
  expect(native).toHaveBeenCalledWith(
    "library_copy_output",
    expect.objectContaining({
      source: source.target,
      destination: destination.target,
      sourceInstance: source.instanceId,
      destinationInstance: destination.instanceId,
      expected: offer,
    }),
  );
  expect(native.mock.calls[0]![1]).not.toHaveProperty("bytes");
});
it("Stop cancels the exact native operation and preserves the rejected copy for journal repair", async () => {
  let reject!: (error: Error) => void;
  native.mockImplementation((command) =>
    command === "library_copy_output"
      ? new Promise((_resolve, fail) => {
          reject = fail;
        })
      : Promise.resolve(),
  );
  const controller = new AbortController();
  const promise = ipc.copyGalleryOutput(source, destination, "print.png", offer, controller.signal);
  await vi.waitFor(() =>
    expect(native).toHaveBeenCalledWith("library_copy_output", expect.anything()),
  );
  const operationId = native.mock.calls[0]![1].operationId;
  controller.abort();
  await vi.waitFor(() =>
    expect(native).toHaveBeenCalledWith("library_copy_cancel", { operationId }),
  );
  reject(new Error("Native transfer stopped"));
  await expect(promise).rejects.toThrow("stopped");
});
it("refuses an already aborted operation before invoking native work", async () => {
  const controller = new AbortController();
  controller.abort();
  await expect(
    ipc.copyGalleryOutput(source, destination, "print.png", offer, controller.signal),
  ).rejects.toThrow();
  expect(native).not.toHaveBeenCalled();
});
