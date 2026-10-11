import { afterEach, expect, it, vi } from "vitest";
import { effectScope, nextTick, ref } from "vue";
import { useRetainedSourceRestore } from "./useRetainedSourceRestore";
import {
  automatedReuseApply,
  clearRetainedSourceReuseIntent,
  retainedSourceReuseState,
  setRetainedSourceReuseIntent,
} from "../lib/retainedSourceReuse";
import { restoreRetainedDraftMedia } from "@studio/lib/retainedDraftMedia";
vi.mock("@studio/lib/retainedDraftMedia", () => ({
  restoreRetainedDraftMedia: vi.fn(),
}));
const inventory = { availability: "available" as const, members: [] };
afterEach(() => {
  clearRetainedSourceReuseIntent();
  vi.clearAllMocks();
});

it("waits for model capabilities and prevents completion after scope disposal", async () => {
  let complete!: (value: {
    patch: Record<string, unknown>;
    inventory: typeof inventory;
  }) => void;
  vi.mocked(restoreRetainedDraftMedia).mockImplementation(
    () =>
      new Promise((resolve) => {
        complete = resolve;
      }),
  );
  setRetainedSourceReuseIntent({
    filename: "old.png",
    origin: { baseUrl: "http://host", apiKey: null },
    inventory,
  });
  const loaded = ref(false);
  const apply = vi.fn();
  const scope = effectScope();
  const restore = scope.run(() =>
    useRetainedSourceRestore({
      loaded,
      read: () => ({}),
      layout: () => ({
        web: true,
        boundary: false,
        sourceMode: "single",
        h3: false,
      }),
      apply,
      onError: vi.fn(),
    }),
  )!;
  expect(restore.pending.value).toBe(true);
  expect(restoreRetainedDraftMedia).not.toHaveBeenCalled();
  loaded.value = true;
  await nextTick();
  const signal = vi.mocked(restoreRetainedDraftMedia).mock.calls[0]![0].signal;
  scope.stop();
  complete({ patch: { audioFile: "stale" }, inventory });
  await Promise.resolve();
  expect(signal?.aborted).toBe(true);
  expect(apply).not.toHaveBeenCalled();
});

it("clears the pending state when the user abandons a retained restore", async () => {
  setRetainedSourceReuseIntent({
    filename: "old.png",
    origin: { baseUrl: "http://host", apiKey: null },
    inventory,
  });
  const scope = effectScope();
  const restore = scope.run(() =>
    useRetainedSourceRestore({
      loaded: ref(false),
      read: () => ({}),
      layout: () => ({
        web: true,
        boundary: false,
        sourceMode: "single",
        h3: false,
      }),
      apply: vi.fn(),
      onError: vi.fn(),
    }),
  )!;
  clearRetainedSourceReuseIntent();
  await nextTick();
  expect(restore.pending.value).toBe(false);
  scope.stop();
});

it("leaves an unretained print to the Library's stash and gallery fallback", async () => {
  // A print made before durable retention resolves `unavailable_legacy`. The
  // Library door restores its source from the local stash or the gallery by
  // name and names what it cannot find; a blocking error here kept Generate
  // disabled even after that fallback had filled the well.
  setRetainedSourceReuseIntent({
    filename: "old.png",
    origin: { baseUrl: "http://host", apiKey: null },
    inventory: { availability: "unavailable_legacy", members: [] },
    metadata: { source_image_sha256: "a".repeat(64) },
  });
  const onError = vi.fn();
  const scope = effectScope();
  const restore = scope.run(() =>
    useRetainedSourceRestore({
      loaded: ref(true),
      read: () => ({}),
      layout: () => ({
        web: true,
        boundary: false,
        sourceMode: "single",
        h3: false,
      }),
      apply: vi.fn(),
      onError,
    }),
  )!;
  await nextTick();
  expect(restoreRetainedDraftMedia).not.toHaveBeenCalled();
  expect(restore.pending.value).toBe(false);
  expect(restore.error.value).toBeNull();
  expect(onError).not.toHaveBeenCalled();
  scope.stop();
});

it("lets a hand-attached input retire a failed retained restore", async () => {
  vi.mocked(restoreRetainedDraftMedia).mockRejectedValueOnce(
    new Error("relay failed"),
  );
  setRetainedSourceReuseIntent({
    filename: "old.png",
    origin: { baseUrl: "http://host", apiKey: null },
    inventory,
  });
  const attached = ref(0);
  const scope = effectScope();
  const restore = scope.run(() =>
    useRetainedSourceRestore({
      loaded: ref(true),
      read: () => ({}),
      layout: () => ({
        web: true,
        boundary: false,
        sourceMode: "single",
        h3: false,
      }),
      attachedMedia: () => attached.value,
      apply: vi.fn(),
      onError: vi.fn(),
    }),
  )!;
  await nextTick();
  await Promise.resolve();
  expect(restore.error.value).toBe("relay failed");

  attached.value = 1;
  await nextTick();

  expect(restore.error.value).toBeNull();
  // Giving up the archive: nothing archived is relayed beside the new input.
  expect(retainedSourceReuseState.value).toBeNull();
  scope.stop();
});

it("does not let an automated restore retire a real failure", async () => {
  vi.mocked(restoreRetainedDraftMedia).mockRejectedValueOnce(
    new Error("relay failed"),
  );
  setRetainedSourceReuseIntent({
    filename: "old.png",
    origin: { baseUrl: "http://host", apiKey: null },
    inventory,
  });
  const attached = ref(0);
  const scope = effectScope();
  const restore = scope.run(() =>
    useRetainedSourceRestore({
      loaded: ref(true),
      read: () => ({}),
      layout: () => ({
        web: true,
        boundary: false,
        sourceMode: "single",
        h3: false,
      }),
      attachedMedia: () => attached.value,
      apply: vi.fn(),
      onError: vi.fn(),
    }),
  )!;
  await nextTick();
  await Promise.resolve();
  expect(restore.error.value).toBe("relay failed");

  // The Library's own stash fallback filling a well is not a person's choice.
  automatedReuseApply(() => {
    attached.value = 1;
  });
  await nextTick();

  expect(restore.error.value).toBe("relay failed");
  expect(retainedSourceReuseState.value).not.toBeNull();
  scope.stop();
});
