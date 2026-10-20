import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import VideoClips from "./VideoClips.vue";
import { ApiError } from "../api/client";
import { CHAIN_ADMISSION_KEY } from "../api/chainJobs";
const api = vi.hoisted(() => ({
  limits: vi.fn(),
  validate: vi.fn(),
  admit: vi.fn(),
  job: vi.fn(),
  owner: vi.fn(),
  cancel: vi.fn(),
  capture: vi.fn(),
}));
vi.mock("../api/chainJobs", async (original) => ({
  ...(await original<typeof import("../api/chainJobs")>()),
  fetchChainLimits: api.limits,
  validateClips: api.validate,
  admitClips: api.admit,
  getChainJob: api.job,
  assertChainOwner: api.owner,
  cancelChainOperation: api.cancel,
  captureChainOwner: api.capture,
}));
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "original",
  target: { baseUrl: "http://plato", apiKey: null },
};
const request = {
  model: "test",
  prompt: "Robot waters a plant",
  width: 768,
  height: 512,
  frames: 107,
  fps: 24,
  steps: 9,
  guidance: 0,
  output_format: "mp4",
};
const limits = {
  model: "test",
  supports_sequence: true,
  frames_per_clip_cap: 345,
  frames_per_clip_recommended: 107,
  max_stages: 16,
  authoring: {
    motion_tail_frames: 1,
    carry_mode: "rgb",
    min_frames: 107,
    frame_step: 17,
    frame_offset: 5,
  },
  linked_parts: { supported: true },
};
let wrapper: ReturnType<typeof mount>;
beforeEach(() => {
  localStorage.clear();
  Object.values(api).forEach((mock) => mock.mockReset());
  api.owner.mockResolvedValue(undefined);
  api.limits.mockResolvedValue(limits);
  api.validate.mockResolvedValue({
    motion_tail_frames: 1,
    estimated_total_frames: 213,
    estimated_duration_ms: 8875,
    fps: 24,
    warnings: [],
  });
});
afterEach(() => wrapper?.unmount());
describe("compact video clips", () => {
  it("starts with one quiet action and authors ordered H3 prompts using the host frame grid", async () => {
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    expect(wrapper.findAll("textarea")).toHaveLength(0);
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(wrapper.findAll("textarea")).toHaveLength(2);
    expect(
      wrapper.get('input[aria-label="Clip 1 length"]').attributes(),
    ).toMatchObject({ min: "107", step: "17", max: "345" });
    await wrapper
      .get('textarea[aria-label="Clip 2 prompt"]')
      .setValue("Robot waves");
    const check = wrapper
      .findAll("button")
      .find((button) => button.text() === "Check length")!;
    await check.trigger("click");
    await flushPromises();
    expect(api.validate.mock.calls[0]?.[1]).toMatchObject({
      motion_tail_frames: 1,
      stages: [{ prompt: "Robot waters a plant" }, { prompt: "Robot waves" }],
    });
    expect(wrapper.text()).toContain("8.88 seconds delivered");
    expect(api.admit).not.toHaveBeenCalled();
  });
  it("freezes the clicked source's latest take and settings, refusing an older print", async () => {
    api.job.mockResolvedValue({
      id: "old",
      state: "completed",
      finalizes: [{ gallery_filename: "latest.mp4", take: 999 }],
      script: {
        chain: {
          ...request,
          seed: "18446744073709551615",
          motion_tail_frames: 1,
          continue_from: { job_id: "previous-part", finalize_take: 5 },
        },
        stage: [{ prompt: "Old", frames: 107 }],
      },
    });
    wrapper = mount(VideoClips, {
      props: { owner, source: { jobId: "old", filename: "latest.mp4" } },
    });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    await wrapper.get("textarea").setValue("A new action");
    const check = wrapper
      .findAll("button")
      .find((button) => button.text() === "Check length")!;
    await check.trigger("click");
    await flushPromises();
    expect(api.validate.mock.calls[0]?.[1]).toMatchObject({
      continue_from: { job_id: "old", finalize_take: 1 },
      width: 768,
      height: 512,
    });
    expect(api.validate.mock.calls[0]?.[1]).not.toHaveProperty("seed");
    expect(wrapper.text()).toContain("next separate video part");
  });
  it("refuses continuation before POST on hosts missing affirmative linked support", async () => {
    api.job.mockResolvedValue({
      id: "old",
      state: "completed",
      finalizes: [{ gallery_filename: "old.mp4" }],
      script: { chain: request, stage: [{ prompt: "Old", frames: 107 }] },
    });
    api.limits.mockResolvedValue({ ...limits, linked_parts: undefined });
    wrapper = mount(VideoClips, {
      props: { owner, source: { jobId: "old", filename: "old.mp4" } },
    });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("Update this machine");
    expect(api.admit).not.toHaveBeenCalled();
  });
  it("blocks stale machine and unsupported ordinary fields without dropping them", async () => {
    wrapper = mount(VideoClips, {
      props: {
        enabled: true,
        request: { ...request, mask_image: "mask" },
        owner,
      },
    });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("mask_image");
    await wrapper.setProps({ owner: { ...owner, instanceId: "replacement" } });
    expect(wrapper.text()).toContain("machine selection or connection changed");
    expect(
      wrapper.get("[data-test=video-clips-submit]").attributes("disabled"),
    ).toBeDefined();
  });
  it("discards a stale validation result when visible clip words change", async () => {
    let finish!: (value: unknown) => void;
    api.validate.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    await wrapper
      .get('textarea[aria-label="Clip 2 prompt"]')
      .setValue("Second");
    const check = wrapper
      .findAll("button")
      .find((button) => button.text() === "Check length")!;
    await check.trigger("click");
    await flushPromises();
    await wrapper
      .get('textarea[aria-label="Clip 2 prompt"]')
      .setValue("Changed while checking");
    finish({
      motion_tail_frames: 1,
      estimated_total_frames: 213,
      estimated_duration_ms: 8875,
      warnings: [],
    });
    await flushPromises();
    expect(wrapper.text()).not.toContain("seconds delivered");
    expect(api.admit).not.toHaveBeenCalled();
  });
  it("keeps unsent clips across reopen and returns the reviewed first prompt when going back to one clip", async () => {
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    await wrapper
      .get('textarea[aria-label="Clip 1 prompt"]')
      .setValue("Reviewed first action");
    await wrapper
      .get('textarea[aria-label="Clip 2 prompt"]')
      .setValue("Reviewed second action");
    await wrapper
      .findAll("button")
      .find((button) => button.text() === "Back to one clip")!
      .trigger("click");
    expect(wrapper.emitted("single")?.[0]).toEqual([
      { prompt: "Reviewed first action", frames: 107 },
    ]);
    wrapper.unmount();
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(
      (
        wrapper.get('textarea[aria-label="Clip 2 prompt"]')
          .element as HTMLTextAreaElement
      ).value,
    ).toBe("Reviewed second action");
    expect(wrapper.text()).toContain("Restored your clips");
  });
  it("requires affirmative fitted-source support instead of letting an older server silently drop policy", async () => {
    wrapper = mount(VideoClips, {
      props: {
        enabled: true,
        request: {
          ...request,
          source_image: "AAAA",
          source_fit: { mode: "crop-fill" },
        },
        owner,
      },
    });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain(
      "Update this machine before using fitted source images",
    );
    expect(
      wrapper.get("[data-test=video-clips-submit]").attributes("disabled"),
    ).toBeDefined();
  });
  it("can cancel the persisted operation while the create reply has not returned", async () => {
    api.admit.mockImplementation(async (_, __, persist) => {
      persist({
        operationId: "unanswered",
        hostId: "plato",
        instanceId: "original",
        baseUrl: "http://plato",
        createdAt: 1,
      });
      return new Promise(() => {});
    });
    api.cancel.mockResolvedValue(undefined);
    api.job.mockResolvedValue(null);
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    await wrapper
      .get('textarea[aria-label="Clip 2 prompt"]')
      .setValue("Second");
    await wrapper.get("[data-test=video-clips-submit]").trigger("click");
    await flushPromises();
    const cancel = wrapper
      .findAll("button")
      .find((button) => button.text() === "Cancel video")!;
    expect(cancel.attributes("disabled")).toBeUndefined();
    await cancel.trigger("click");
    await flushPromises();
    expect(api.cancel.mock.calls[0]?.[1]).toBe("unanswered");
    expect(wrapper.text()).toContain("Cancelled before queueing");
    expect(
      JSON.parse(localStorage.getItem(CHAIN_ADMISSION_KEY)!)[0],
    ).toMatchObject({ cancelRequested: true, cancelAcknowledged: true });
  });
  it("keeps failed jobs reachable after closing and reopening", async () => {
    localStorage.setItem(
      CHAIN_ADMISSION_KEY,
      JSON.stringify([
        {
          operationId: "failed",
          hostId: "plato",
          instanceId: "original",
          baseUrl: "http://plato",
          createdAt: 1,
        },
      ]),
    );
    api.job.mockResolvedValue({
      id: "failed",
      state: "failed",
      stage_count: 2,
      current_stage: 1,
      stages: [],
      error: "Temporary GPU error",
    });
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await flushPromises();
    expect(wrapper.text()).toContain("Resume video");
    await wrapper
      .findAll("button")
      .find((button) => button.text() === "Done")!
      .trigger("click");
    expect(
      JSON.parse(localStorage.getItem(CHAIN_ADMISSION_KEY)!)[0].operationId,
    ).toBe("failed");
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("Temporary GPU error");
    expect(wrapper.text()).toContain("Resume video");
  });
  it("does not claim Library publication without the actual final filename", async () => {
    localStorage.setItem(
      CHAIN_ADMISSION_KEY,
      JSON.stringify([
        {
          operationId: "done",
          hostId: "plato",
          instanceId: "original",
          baseUrl: "http://plato",
          createdAt: 1,
        },
      ]),
    );
    api.job.mockResolvedValue({
      id: "done",
      state: "completed",
      stage_count: 2,
      current_stage: 1,
      stages: [],
      finalizes: [],
    });
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await flushPromises();
    expect(wrapper.text()).toContain("did not report a saved Library video");
    expect(wrapper.text()).not.toContain("Saved to Library");
  });
  it("does not offer a smooth clip shorter than its inherited tail on the model's frame grid", async () => {
    api.limits.mockResolvedValue({
      ...limits,
      authoring: undefined,
      frames_per_clip_cap: 97,
      frames_per_clip_recommended: 97,
    });
    wrapper = mount(VideoClips, {
      props: {
        enabled: true,
        request: { ...request, frames: 97 },
        owner,
        frameContract: {
          family: "ltx2",
          min_frames: 1,
          frame_step: 8,
          frame_offset: 1,
        },
      },
    });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(
      wrapper.get('input[aria-label="Clip 1 length"]').attributes("min"),
    ).toBe("25");
  });
  it("hydrates Ref2VA originals explicitly before reusing them in a continued part", async () => {
    const reference = {
      kind: "image",
      width: 32,
      height: 32,
      mime_type: "image/png",
      media: { authority: "inline", data: "AAAA" },
    };
    api.limits.mockResolvedValue({
      ...limits,
      authoring: {
        ...limits.authoring,
        motion_tail_frames: 0,
        carry_mode: "independent",
      },
    });
    api.job.mockImplementation(async (_, __, originals) => ({
      id: "ref-source",
      state: "completed",
      finalizes: [{ gallery_filename: "ref.mp4" }],
      script: {
        chain: { ...request, motion_tail_frames: 0 },
        stage: [
          {
            prompt: "Previous action",
            frames: 107,
            generation_references: [
              {
                ...reference,
                media: originals
                  ? reference.media
                  : { authority: "descriptor" },
              },
            ],
          },
        ],
      },
    }));
    wrapper = mount(VideoClips, {
      props: { owner, source: { jobId: "ref-source", filename: "ref.mp4" } },
    });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(api.job.mock.calls.some((call) => call[2] === true)).toBe(true);
    await wrapper.get("textarea").setValue("New action");
    await wrapper
      .findAll("button")
      .find((button) => button.text() === "Check length")!
      .trigger("click");
    await flushPromises();
    expect(api.validate.mock.calls[0]?.[1]).toMatchObject({
      motion_tail_frames: 0,
      stages: [{ generation_references: [reference] }],
    });
    expect(wrapper.text()).toContain("1 reference input used in every clip");
  });
  it("captures an unstamped Library origin from its authenticated status before continuing", async () => {
    api.capture.mockResolvedValue(owner);
    api.job.mockResolvedValue({
      id: "origin-job",
      state: "completed",
      finalizes: [{ gallery_filename: "origin.mp4" }],
      script: {
        chain: { ...request, motion_tail_frames: 1 },
        stage: [{ prompt: "Previous", frames: 107 }],
      },
    });
    wrapper = mount(VideoClips, {
      props: {
        owner: { ...owner, instanceId: undefined },
        source: { jobId: "origin-job", filename: "origin.mp4" },
      },
    });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    expect(api.capture).toHaveBeenCalledTimes(1);
    expect(wrapper.find("textarea").exists()).toBe(true);
    await wrapper.get("textarea").setValue("Next origin clip");
    await wrapper
      .findAll("button")
      .find((button) => button.text() === "Check length")!
      .trigger("click");
    await flushPromises();
    expect(api.validate.mock.calls[0]?.[0].instanceId).toBe("original");
  });
  it("clears a definitely rejected admission and allows correcting the visible clips", async () => {
    api.admit.mockImplementation(async (_, __, persist) => {
      persist({
        operationId: "rejected",
        hostId: "plato",
        instanceId: "original",
        baseUrl: "http://plato",
        createdAt: 1,
      });
      throw new ApiError("Clip 2 is invalid", 422);
    });
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    await wrapper
      .get('textarea[aria-label="Clip 2 prompt"]')
      .setValue("Second");
    await wrapper.get("[data-test=video-clips-submit]").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("Clip 2 is invalid");
    expect(JSON.parse(localStorage.getItem(CHAIN_ADMISSION_KEY)!)).toEqual([]);
    expect(
      wrapper
        .get('textarea[aria-label="Clip 2 prompt"]')
        .attributes("disabled"),
    ).toBeUndefined();
    expect(api.job).not.toHaveBeenCalled();
  });
  it("recovers an ambiguous POST by its same owner and UUID after remount without duplicate creation", async () => {
    api.admit.mockImplementation(async (_, __, persist) => {
      persist({
        operationId: "known-operation",
        hostId: "plato",
        instanceId: "original",
        baseUrl: "http://plato",
        createdAt: 1,
      });
      throw new Error("Reply lost");
    });
    api.job.mockResolvedValue({
      id: "known-operation",
      state: "running",
      stage_count: 2,
      current_stage: 1,
      stages: [],
    });
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await wrapper.get("[data-test=video-clips-open]").trigger("click");
    await flushPromises();
    await wrapper
      .get('textarea[aria-label="Clip 2 prompt"]')
      .setValue("Second");
    await wrapper.get("[data-test=video-clips-submit]").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("Making clip 2 of 2");
    wrapper.unmount();
    wrapper = mount(VideoClips, { props: { enabled: true, request, owner } });
    await flushPromises();
    expect(wrapper.text()).toContain("Making clip 2 of 2");
    expect(api.admit).toHaveBeenCalledTimes(1);
    expect(
      api.job.mock.calls.every(
        (call) => call[0].hostId === "plato" && call[1] === "known-operation",
      ),
    ).toBe(true);
  });
});
