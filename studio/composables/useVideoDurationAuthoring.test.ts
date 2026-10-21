import { computed, effectScope, nextTick, ref } from "vue";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
const inspect = vi.hoisted(() => vi.fn());
vi.mock("../api/videoLibraryExtension", () => ({
  inspectVideoLibraryExtension: inspect,
}));
vi.mock("../api/videoJobs", () => ({
  validateVideoJob: vi.fn(),
  admitVideoJob: vi.fn(),
  getVideoJob: vi.fn(),
  mutateVideoJob: vi.fn(),
  assertVideoJobOwner: vi.fn(),
  planningVideoRequest: (request: unknown) => request,
}));
import { requestVideoDurationReset } from "../lib/videoLibraryExtensionHandoff";
import { createVideoDurationAuthoring } from "./useVideoDurationAuthoring";
let scope: ReturnType<typeof effectScope>;
beforeEach(() => {
  localStorage.clear();
  inspect.mockReset();
  vi.useFakeTimers();
  scope = effectScope();
});
afterEach(() => {
  scope.stop();
  vi.useRealTimers();
});
function setup(
  generation: Record<string, unknown> = {
    model: "selected",
    prompt: "ordinary",
  },
) {
  const frames = ref(107),
    supported = ref(true);
  const owner = computed(() => ({
    hostId: "plato",
    label: "Plato",
    instanceId: "installation",
    target: { baseUrl: "http://plato", apiKey: "private" },
  }));
  const state = scope.run(() =>
    createVideoDurationAuthoring({
      key: "test-duration",
      clip: computed(() => true),
      legacyActive: computed(() => false),
      supported: computed(() => supported.value),
      owner,
      frames: computed(() => frames.value),
      fps: computed(() => 24),
      generation: computed(() => generation as { model: string }),
    }),
  )!;
  return { state, frames, supported, owner };
}
it("starts at the short recipe length and preserves an authored duration on model/frame changes", async () => {
  const { state, frames } = setup();
  expect(state.draft.value?.durationMs).toBe("4458");
  state.draft.value = { durationMs: "30000", scenesEnabled: false, scenes: [] };
  frames.value = 241;
  await nextTick();
  expect(state.draft.value.durationMs).toBe("30000");
  expect(JSON.parse(localStorage.getItem("test-duration")!).durationMs).toBe(
    "30000",
  );
});
it("keeps a saved duration visible and blocked when its host loses the affirmative capability", async () => {
  const { state, supported } = setup();
  state.draft.value = { durationMs: "30000", scenesEnabled: false, scenes: [] };
  supported.value = false;
  await nextTick();
  expect(state.visible.value).toBe(true);
  expect(state.enabled.value).toBe(false);
  expect(state.blocker.value).toContain("Update");
});
it("does not enable duration on an older host with no prior duration draft", () => {
  localStorage.removeItem("test-duration");
  const state = scope.run(() =>
    createVideoDurationAuthoring({
      key: "old-host",
      clip: computed(() => true),
      legacyActive: computed(() => false),
      supported: computed(() => false),
      owner: computed(() => null),
      frames: computed(() => 107),
      fps: computed(() => 24),
      generation: computed(() => null),
    }),
  )!;
  expect(state.visible.value).toBe(false);
  expect(state.draft.value).toBeNull();
});
it("keeps a Library prefix bound to its installation without persisting credentials or grants", async () => {
  const { state, owner } = setup();
  state.extend({
    owner: owner.value,
    source: {
      filename: "original.mp4",
      expected_archive_identity_sha256: "a".repeat(64),
    },
    title: "Original",
    metadata: {},
    inventory: { availability: "available", members: [] },
  });
  await nextTick();
  const saved = localStorage.getItem("test-duration")!;
  expect(saved).toContain("original.mp4");
  expect(saved).not.toContain("private");
  expect(state.draft.value?.extension?.owner.instanceId).toBe("installation");
  state.reset();
  expect(state.draft.value?.extension).toBeUndefined();
});

it("re-inspects a cold prefix by exact saved archive identity without restoring old endpoints", async () => {
  const extension = {
    source: {
      filename: "old.mp4",
      expected_archive_identity_sha256: "a".repeat(64),
    },
    owner: {
      hostId: "plato",
      instanceId: "installation",
      baseUrl: "http://plato",
    },
    title: "Original",
    machine: "Plato",
    references: [],
  };
  localStorage.setItem(
    "test-duration",
    JSON.stringify({
      durationMs: "30000",
      scenesEnabled: false,
      scenes: [],
      extension,
    }),
  );
  inspect.mockResolvedValue({
    originalReferences: [
      {
        descriptor: { kind: "image" },
        member: { display_name: "hero.png", role: "references" },
      },
    ],
  });
  const { state } = setup();
  await nextTick();
  await Promise.resolve();
  await nextTick();
  expect(inspect.mock.calls[0]?.[1]).toBe("old.mp4");
  expect(inspect.mock.calls[0]?.[5]).toBe("a".repeat(64));
  expect(state.draft.value?.durationMs).toBe("30000");
  expect(state.draft.value?.extension?.references[0]?.name).toBe("hero.png");
});

it("preserves and blocks an explicitly attached frame-zero keyframe during continuous Extend", () => {
  const generation = {
    model: "selected",
    prompt: "ordinary",
    keyframes: [{ frame: 0, image: "chosen" }],
  };
  const { state, owner } = setup(generation);
  state.extend({
    owner: owner.value,
    source: { filename: "old.mp4" },
    title: "Original",
    metadata: {},
    inventory: { availability: "unavailable_legacy", members: [] },
  });
  state.planner.plan.value = {
    join_mode: "continuous",
  } as import("../lib/api/videoJobTypes").VideoJobPlan;
  expect(state.blocker.value).toContain("Remove the selected opening image");
  expect(generation.keyframes[0]?.image).toBe("chosen");
});
it("an explicit new draft or Use these settings clears a stored prefix even before Create remounts", async () => {
  localStorage.setItem(
    "test-duration",
    JSON.stringify({
      durationMs: "30000",
      scenesEnabled: false,
      scenes: [],
      extension: {
        source: { filename: "old.mp4" },
        owner: {
          hostId: "plato",
          instanceId: "installation",
          baseUrl: "http://plato",
        },
        title: "Original",
        machine: "Plato",
        references: [],
      },
    }),
  );
  requestVideoDurationReset();
  const { state } = setup();
  await nextTick();
  expect(state.draft.value?.extension).toBeUndefined();
  expect(state.draft.value?.durationMs).toBe("4458");
  expect(inspect).not.toHaveBeenCalled();
});
