import { afterEach, expect, it, vi } from "vitest";
import { effectScope, nextTick, ref } from "vue";
import { useRetainedSourceRestore } from "./useRetainedSourceRestore";
import {
  clearRetainedSourceReuseIntent,
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
