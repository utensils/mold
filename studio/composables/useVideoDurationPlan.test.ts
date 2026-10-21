import { computed, ref } from "vue";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
const validate = vi.hoisted(() => vi.fn());
vi.mock("../api/videoJobs", async (original) => ({
  ...(await original<typeof import("../api/videoJobs")>()),
  validateVideoJob: validate,
}));
import { createVideoDurationPlan } from "./useVideoDurationPlan";
const machine = {
  hostId: "plato",
  label: "Plato",
  instanceId: "installation",
  target: { baseUrl: "http://plato", apiKey: "private" },
};
const input = () => ({
  duration_ms: "30000",
  generation: {
    model: "opaque",
    prompt: "new scene",
    width: 768,
    height: 768,
    steps: 2,
    source_image: "private original bytes",
  },
});
beforeEach(() => {
  vi.useFakeTimers();
  validate.mockReset();
});
afterEach(() => vi.useRealTimers());
it("debounces scalar plans and never sends source bytes while the user edits", async () => {
  const request = ref(input());
  const owner = ref(machine);
  validate.mockResolvedValue({ clip_count: "7" });
  const state = createVideoDurationPlan(
    computed(() => true),
    computed(() => owner.value),
    computed(() => request.value),
  );
  request.value.duration_ms = "60000";
  await vi.advanceTimersByTimeAsync(199);
  expect(validate).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1);
  expect(validate).toHaveBeenCalledTimes(1);
  expect(validate.mock.calls[0]![1]).toEqual({
    duration_ms: "60000",
    generation: {
      model: "opaque",
      prompt: "new scene",
      width: 768,
      height: 768,
      steps: 2,
    },
  });
  expect(validate.mock.calls[0]![2].planningOnly).toBe(true);
  state.dispose();
});
it("cannot publish a stale plan after the duration or installation changes", async () => {
  const request = ref(input());
  const owner = ref(machine);
  let finish!: (value: unknown) => void;
  validate.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  const state = createVideoDurationPlan(
    computed(() => true),
    computed(() => owner.value),
    computed(() => request.value),
  );
  await vi.advanceTimersByTimeAsync(200);
  owner.value = { ...machine, instanceId: "replacement" };
  request.value.duration_ms = "60000";
  await vi.advanceTimersByTimeAsync(0);
  finish({ clip_count: "7" });
  await vi.advanceTimersByTimeAsync(0);
  expect(state.plan.value).toBeNull();
  expect(validate.mock.calls[0]![2].isCurrent()).toBe(false);
  state.dispose();
});
it("does no background work without a supported authenticated owner and stops on disposal", async () => {
  const enabled = ref(false);
  const state = createVideoDurationPlan(
    computed(() => enabled.value),
    computed(() => machine),
    computed(input),
  );
  await vi.advanceTimersByTimeAsync(1000);
  expect(validate).not.toHaveBeenCalled();
  enabled.value = true;
  await vi.advanceTimersByTimeAsync(0);
  state.dispose();
  await vi.advanceTimersByTimeAsync(1000);
  expect(validate).not.toHaveBeenCalled();
});
it("captures reactive scene prompts and source locators without retaining a live draft", async () => {
  const request = ref({
    ...input(),
    source: { filename: "source.mp4" },
    scenes: [{ prompt: "opening", duration_ms: "30000" }],
  });
  validate.mockResolvedValue({ clip_count: "7" });
  const state = createVideoDurationPlan(
    computed(() => true),
    computed(() => machine),
    computed(() => request.value),
  );
  await vi.advanceTimersByTimeAsync(200);
  expect(validate.mock.calls[0]![1].scenes).toEqual([
    { prompt: "opening", duration_ms: "30000" },
  ]);
  request.value.scenes[0]!.prompt = "changed";
  expect(validate.mock.calls[0]![1].scenes[0].prompt).toBe("opening");
  state.dispose();
});
