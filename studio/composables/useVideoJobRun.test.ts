import { beforeEach, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
const api = vi.hoisted(() => ({
  validate: vi.fn(),
  admit: vi.fn(),
  owner: vi.fn(),
  get: vi.fn(),
  mutate: vi.fn(),
}));
vi.mock("../api/videoJobs", () => ({
  validateVideoJob: api.validate,
  admitVideoJob: api.admit,
  assertVideoJobOwner: api.owner,
  getVideoJob: api.get,
  mutateVideoJob: api.mutate,
}));
import { createVideoJobRun } from "./useVideoJobRun";
import {
  readVideoJobOperations,
  saveVideoJobOperation,
} from "../lib/videoJobRecovery";
const owner = {
  hostId: "plato",
  label: "Plato",
  instanceId: "installation",
  target: { baseUrl: "http://plato", apiKey: "private" },
};
const request = {
  duration_ms: "30000",
  generation: {
    model: "opaque",
    prompt: "a new scene",
    width: 768,
    height: 768,
    steps: 2,
  },
};
const plan = {
  validation_revision: "proof",
  model: "opaque",
  width: 768,
  height: 768,
  fps: 24,
  single_clip_max_frames: 107,
  overlap_frames: 1,
  join_mode: "continuous" as const,
  clip_count: "7",
  part_count: "1",
  added_frames: "720",
  source_frames: "0",
  total_frames: "720",
  added_duration_ms: "30000",
  total_duration_ms: "30000",
  warnings: [],
};
const admission = {
  kind: "video_job" as const,
  operationId: "known",
  hostId: "plato",
  instanceId: "installation",
  baseUrl: "http://plato",
  createdAt: 1,
};
const job = {
  id: "known",
  state: "queued",
  plan,
  completed_clips: "0",
  can_cancel: true,
  can_resume: false,
  can_delete: false,
  created_at_ms: 1,
  updated_at_ms: 1,
};
beforeEach(() => {
  localStorage.clear();
  Object.values(api).forEach((mock) => mock.mockReset());
  api.owner.mockResolvedValue(undefined);
  api.validate.mockResolvedValue(plan);
  api.get.mockResolvedValue(null);
  api.admit.mockImplementation(async (_, __, ___, id, persist) => {
    persist({ ...admission, operationId: id });
    return { ...job, id };
  });
});
it("uses a full prepared validation before create and lets durable Queue own a confirmed parent", async () => {
  const state = createVideoJobRun(() => true);
  await state.submit(owner, request);
  expect(api.validate).toHaveBeenCalledBefore(api.admit);
  expect(state.detail.value?.state).toBe("queued");
  expect(readVideoJobOperations()).toEqual([]);
});
it("blocks another admission after a lost reply and recovers only the persisted exact ID", async () => {
  api.admit.mockImplementation(async (_, __, ___, id, persist) => {
    persist({ ...admission, operationId: id });
    throw new Error("lost reply");
  });
  const state = createVideoJobRun(() => true);
  await state.submit(owner, request);
  const known = readVideoJobOperations()[0]!;
  expect(state.uncertain.value).toBe(true);
  await state.submit(owner, request);
  expect(api.admit).toHaveBeenCalledTimes(1);
  api.get.mockResolvedValue({ ...job, id: known.operationId });
  await state.recover(owner, known);
  expect(state.detail.value?.id).toBe(known.operationId);
  expect(api.admit).toHaveBeenCalledTimes(1);
  expect(readVideoJobOperations()).toEqual([]);
});
it("refuses edits during full validation before creating any operation", async () => {
  let finish!: (value: unknown) => void;
  api.validate.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  let current = true;
  const state = createVideoJobRun(() => true);
  const submission = state.submit(owner, request, {
    isDraftCurrent: () => current,
  });
  current = false;
  finish(plan);
  await submission;
  expect(api.admit).not.toHaveBeenCalled();
  expect(readVideoJobOperations()).toEqual([]);
});
it("retains a wrong-job terminal reply and never claims somebody else's saved video", async () => {
  saveVideoJobOperation(admission);
  api.get.mockResolvedValue({
    ...job,
    id: "different",
    state: "completed",
    output_filename: "other.mp4",
  });
  const state = createVideoJobRun(() => true);
  await state.recover(owner, admission);
  expect(state.detail.value).toBeNull();
  expect(readVideoJobOperations()).toEqual([admission]);
  expect(state.error.value).toContain("different");
});
it("replays a persisted Stop only on its captured installation after a lost acknowledgement", async () => {
  saveVideoJobOperation({ ...admission, cancelRequested: true });
  api.mutate
    .mockRejectedValueOnce(new Error("lost Stop reply"))
    .mockResolvedValue(undefined);
  const state = createVideoJobRun(() => true);
  await state.recover(owner, readVideoJobOperations()[0]!);
  expect(readVideoJobOperations()[0]!.cancelRequested).toBe(true);
  api.get.mockRejectedValue(
    new ApiError("cancelled before admission", 410, {
      code: "VIDEO_OPERATION_CANCELLED",
    }),
  );
  await state.refresh();
  expect(
    api.mutate.mock.calls.map(([machine, id, action]) => [
      machine.instanceId,
      id,
      action,
    ]),
  ).toEqual([
    ["installation", "known", "cancel"],
    ["installation", "known", "cancel"],
  ]);
  expect(state.cancelledBeforeAdmission.value).toBe(true);
  expect(readVideoJobOperations()).toEqual([]);
});
it("does not delete recovery authority or mutate when its route was replaced", async () => {
  saveVideoJobOperation(admission);
  const state = createVideoJobRun(() => false);
  await state.recover(owner, admission);
  await state.cancel();
  expect(api.get).not.toHaveBeenCalled();
  expect(api.mutate).not.toHaveBeenCalled();
  expect(readVideoJobOperations()).toEqual([admission]);
});
it("keeps uploaded input authority while POST may still be consuming it and releases after exact recovery", async () => {
  const release = vi.fn(async () => {});
  api.admit.mockImplementation(async (_, __, ___, id, persist) => {
    persist({ ...admission, operationId: id });
    throw new Error("lost reply");
  });
  const state = createVideoJobRun(() => true);
  await state.submit(owner, request, {
    prepare: async () => ({ request, release }),
  });
  expect(release).not.toHaveBeenCalled();
  const known = readVideoJobOperations()[0]!;
  api.get.mockResolvedValue({ ...job, id: known.operationId });
  await state.refresh();
  expect(release).toHaveBeenCalledTimes(1);
});
it("rejects a wrong-ID create acknowledgement without erasing the known operation", async () => {
  api.admit.mockImplementation(async (_, __, ___, id, persist) => {
    persist({ ...admission, operationId: id });
    return { ...job, id: "different" };
  });
  const state = createVideoJobRun(() => true);
  await state.submit(owner, request);
  expect(state.detail.value).toBeNull();
  expect(readVideoJobOperations()).toHaveLength(1);
});
it.each([408, 409, 429, 500])(
  "keeps uncertain HTTP %s recoverable",
  async (status) => {
    api.admit.mockImplementation(async (_, __, ___, id, persist) => {
      persist({ ...admission, operationId: id });
      throw new ApiError("ambiguous", status);
    });
    const state = createVideoJobRun(() => true);
    await state.submit(owner, request);
    expect(readVideoJobOperations()).toHaveLength(1);
  },
);
it("does not overwrite an unreadable operation journal or admit a replacement", async () => {
  localStorage.setItem(
    "mold.video-jobs.operations.v1",
    "private unreadable receipt",
  );
  const state = createVideoJobRun(() => true);
  await state.submit(owner, request);
  expect(api.admit).not.toHaveBeenCalled();
  expect(state.error.value).toContain("unreadable");
  expect(localStorage.getItem("mold.video-jobs.operations.v1")).toBe(
    "private unreadable receipt",
  );
});
it("persists Stop intent for an already acknowledged parent before mutation and retains lost Stop replies", async () => {
  const state = createVideoJobRun(() => true);
  api.get.mockResolvedValue(job);
  await state.recover(owner, admission);
  expect(readVideoJobOperations()).toEqual([]);
  api.mutate.mockImplementation(async () => {
    expect(readVideoJobOperations()[0]?.cancelRequested).toBe(true);
    throw new Error("lost Stop reply");
  });
  await state.cancel();
  expect(readVideoJobOperations()[0]?.operationId).toBe("known");
  expect(readVideoJobOperations()[0]?.cancelRequested).toBe(true);
});
it("ignores an older overlapping Queue snapshot after a newer terminal snapshot", async () => {
  saveVideoJobOperation(admission);
  api.get.mockResolvedValue(job);
  const state = createVideoJobRun(() => true);
  await state.recover(owner, admission);
  let resolveOld!: (value: unknown) => void;
  api.get.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        resolveOld = resolve;
      }),
  );
  const old = state.refresh();
  await vi.waitFor(() => expect(api.get).toHaveBeenCalledTimes(2));
  api.get.mockResolvedValue({
    ...job,
    state: "completed",
    output_filename: "final.mp4",
  });
  await state.refresh();
  resolveOld(job);
  await old;
  expect(state.detail.value?.state).toBe("completed");
  expect(state.detail.value?.output_filename).toBe("final.mp4");
});
it("keeps a confirmed video accepted when best-effort lease cleanup fails", async () => {
  const state = createVideoJobRun(() => true);
  await state.submit(owner, request, {
    prepare: async () => ({
      request,
      release: async () => {
        throw new Error("cleanup unavailable");
      },
    }),
  });
  expect(state.detail.value?.state).toBe("queued");
  expect(state.error.value).toBe("");
  expect(readVideoJobOperations()).toEqual([]);
});
it("retires an earlier lost Stop intent before explicit Resume so refresh cannot stop it again", async () => {
  api.get.mockResolvedValue({ ...job, state: "paused", can_resume: true });
  const state = createVideoJobRun(() => true);
  await state.recover(owner, admission);
  api.mutate.mockRejectedValueOnce(new Error("lost Stop"));
  await state.cancel();
  expect(readVideoJobOperations()[0]?.cancelRequested).toBe(true);
  api.mutate.mockImplementation(async (_owner, _id, action) => {
    if (action === "resume") {
      expect(readVideoJobOperations()[0]?.cancelRequested).toBe(false);
      throw new Error("lost Resume reply");
    }
  });
  await state.resume();
  const saved = readVideoJobOperations()[0]!;
  expect(saved.cancelRequested).toBe(false);
  const cold = createVideoJobRun(() => true);
  await cold.recover(owner, saved);
  await cold.refresh();
  expect(
    api.mutate.mock.calls.filter((call) => call[2] === "cancel"),
  ).toHaveLength(1);
  expect(
    api.mutate.mock.calls.filter((call) => call[2] === "resume"),
  ).toHaveLength(1);
});

it("retires an acknowledged missing job but preserves an unanswered attempt", async () => {
  const accepted = createVideoJobRun(() => true);
  await accepted.recover(owner, admission, { acknowledged: true });
  expect(accepted.removed.value).toBe(true);
  expect(accepted.admission.value).toBeNull();
  const unanswered = createVideoJobRun(() => true);
  await unanswered.recover(owner, admission);
  expect(unanswered.removed.value).toBe(false);
  expect(unanswered.uncertain.value).toBe(true);
});

it("does not resurrect a deleted accepted job from a delayed earlier read", async () => {
  const state = createVideoJobRun(() => true);
  api.get.mockResolvedValue({
    ...job,
    state: "completed",
    can_cancel: false,
    can_delete: true,
  });
  await state.recover(owner, admission);
  let finish!: (value: unknown) => void;
  api.get.mockReturnValueOnce(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  const oldRead = state.refresh();
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  await state.remove();
  finish({ ...job, state: "completed" });
  await oldRead;
  expect(state.removed.value).toBe(true);
  expect(state.detail.value).toBeNull();
});

it("keeps accepted recovery data visible after an offline delete failure", async () => {
  const state = createVideoJobRun(() => true);
  api.get.mockResolvedValue({
    ...job,
    state: "completed",
    can_cancel: false,
    can_delete: true,
  });
  await state.recover(owner, admission);
  api.mutate.mockRejectedValue(new Error("offline"));
  await expect(state.remove()).rejects.toThrow("offline");
  expect(state.detail.value?.id).toBe("known");
  expect(state.removed.value).toBe(false);
});

it("lets the creating owner forget only its matching acknowledged removed result", async () => {
  const state = createVideoJobRun(() => true);
  api.get.mockResolvedValue({ ...job, state: "completed", can_delete: true });
  await state.recover(owner, admission);
  state.forgetRemoved("another");
  expect(state.detail.value?.id).toBe("known");
  state.forgetRemoved("known");
  expect(state.detail.value).toBeNull();
  expect(state.removed.value).toBe(true);
});

it("keeps acknowledged recovery visible when persisting local retirement fails", async () => {
  const state = createVideoJobRun(() => true);
  api.get.mockResolvedValue({ ...job, state: "paused", can_resume: true });
  await state.recover(owner, admission);
  saveVideoJobOperation(admission);
  state.admission.value = admission;
  const saved = readVideoJobOperations();
  const write = vi.spyOn(localStorage, "setItem").mockImplementation(() => {
    throw new Error("device storage unavailable");
  });
  api.get.mockResolvedValue(null);
  await state.refresh();
  expect(state.removed.value).toBe(false);
  expect(state.detail.value?.id).toBe("known");
  expect(state.error.value).toContain("device storage unavailable");
  write.mockRestore();
  expect(readVideoJobOperations()).toEqual(saved);
  await state.refresh();
  expect(state.removed.value).toBe(true);
});
