import { describe, expect, it } from "vitest";
import {
  QUEUE_SETTINGS_PENDING_NOTICE,
  queueEntryDetailModel,
  type QueueDetailMetadata,
} from "./queueEntryDetail";
import type { QueueEntry, QueuePlan } from "../api/queuePlan";

function entry(overrides: Partial<QueueEntry> = {}): QueueEntry {
  return {
    id: "job-1",
    model: "flux-dev:q8",
    state: "queued",
    started_at_unix_ms: 1_700_000_000_000,
    position: 3,
    ...overrides,
  };
}

const metadata: QueueDetailMetadata = {
  prompt: "a cat on a porch",
  negative_prompt: "blurry",
  model: "flux-dev:q8",
  seed: 42,
  steps: 28,
  guidance: 3.5,
  width: 1024,
  height: 1024,
};

function model(
  input: Parameters<typeof queueEntryDetailModel>[0] = {} as never,
) {
  return queueEntryDetailModel({
    entry: entry(),
    hostLabel: "plato",
    modelLabel: "FLUX.1 [dev] Q8",
    nowMs: 1_700_000_060_000,
    ...input,
  });
}

describe("queueEntryDetailModel", () => {
  it("resolves the shared waiting vocabulary rather than a raw position", () => {
    expect(model({ entry: entry({ position: 0 }) }).stateCode).toBe("NEXT UP");
    expect(model({ entry: entry({ position: 3 }) }).waitLabel).toBe(
      "#3 in line",
    );
    expect(
      model({ entry: entry({ state: "running", gpu: 1 }) }).stateCode,
    ).toBe("RUNNING · GPU 1");
    expect(model({ entry: entry({ state: "held" }) }).stateCode).toBe("HELD");
  });

  it("lets an actionable blocked reason outrank the position", () => {
    const plan: QueuePlan = {
      plan_version: 1,
      state_version: 1,
      optimizer_state: "clean",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [
        {
          work_id: "w1",
          parent_id: "job-1",
          work_kind: "generation",
          priority_class: "normal",
          queue_rank: 0,
          bypass_count: 0,
          estimate_confidence: "high",
          blocked_reason: "model_not_installed",
        },
      ],
    };
    expect(model({ plan }).waitLabel).not.toBe("#3 in line");
    expect(model({ plan }).waitLabel.length).toBeGreaterThan(0);
  });

  it("keeps ordinary serialization falling through to the position", () => {
    const plan: QueuePlan = {
      plan_version: 1,
      state_version: 1,
      optimizer_state: "clean",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [
        {
          work_id: "w1",
          parent_id: "job-1",
          work_kind: "generation",
          priority_class: "normal",
          queue_rank: 0,
          bypass_count: 0,
          estimate_confidence: "high",
          blocked_reason: "no_idle_device",
        },
      ],
    };
    expect(model({ plan }).waitLabel).toBe("#3 in line");
  });

  it("groups host-supplied settings the way the Create inspector does", () => {
    const detail = model({ metadata });
    expect(detail.metadataSource).toBe("host");
    expect(detail.settingsNotice).toBeNull();
    expect(detail.prompt).toBe("a cat on a porch");
    expect(detail.negativePrompt).toBe("blurry");
    const output = detail.groups.find((group) => group.title === "Output");
    expect(output?.fields).toContainEqual({
      label: "Size",
      value: "1024×1024",
      mono: true,
    });
    const sampling = detail.groups.find((group) => group.title === "Sampling");
    expect(sampling?.fields.map((field) => field.label)).toEqual([
      "Steps",
      "Guidance",
      "Seed",
    ]);
  });

  it("omits fields the host did not supply rather than guessing them", () => {
    const detail = model({ metadata });
    const labels = detail.groups.flatMap((group) =>
      group.fields.map((field) => field.label),
    );
    expect(labels).not.toContain("Frames");
    expect(labels).not.toContain("Strength");
    expect(labels).not.toContain("Collection");
  });

  it("renders an unpinned seed as Random and an explicit zero as 0", () => {
    const unpinned = model({
      entry: entry({ seed_pinned: false }),
      metadata: { ...metadata, seed: 0 },
    });
    expect(
      unpinned.groups
        .flatMap((group) => group.fields)
        .find((field) => field.label === "Seed")?.value,
    ).toBe("Random");
    const pinned = model({
      entry: entry({ seed_pinned: true }),
      metadata: { ...metadata, seed: 0 },
    });
    expect(
      pinned.groups
        .flatMap((group) => group.fields)
        .find((field) => field.label === "Seed")?.value,
    ).toBe("0");
  });

  it("falls back to this client's own submitted settings when the listing has none", () => {
    const detail = model({ localMetadata: metadata });
    expect(detail.metadataSource).toBe("local");
    expect(detail.prompt).toBe("a cat on a porch");
    expect(detail.settingsNotice).toBeNull();
    expect(detail.reuse.available).toBe(true);
  });

  it("prefers the host's own settings over the local fallback", () => {
    const detail = model({
      metadata: { ...metadata, prompt: "host copy" },
      localMetadata: { ...metadata, prompt: "local copy" },
    });
    expect(detail.metadataSource).toBe("host");
    expect(detail.prompt).toBe("host copy");
  });

  it("explains an absent request without telling anyone to upgrade a server", () => {
    const detail = model();
    expect(detail.metadataSource).toBeNull();
    expect(detail.settingsNotice).toBe(QUEUE_SETTINGS_PENDING_NOTICE);
    expect(detail.settingsNotice).not.toMatch(/upgrade/i);
    expect(detail.reuse.available).toBe(false);
    expect(detail.reuse.blockedReason).toBe(QUEUE_SETTINGS_PENDING_NOTICE);
  });

  it("surfaces the whole hold reason and error as one copyable problem", () => {
    const detail = model({
      entry: entry({
        state: "held",
        held_reason: "dispatch budget exhausted",
        error: "CUDA error: an illegal memory access was encountered",
        dispatch_attempts: 2,
      }),
    });
    expect(detail.problem?.detail).toContain("dispatch budget exhausted");
    expect(detail.problem?.detail).toContain("illegal memory access");
    expect(detail.copyText).toContain("illegal memory access");
  });

  it("does not repeat one message twice when held_reason and error agree", () => {
    const detail = model({
      entry: entry({ state: "held", held_reason: "same", error: "same" }),
    });
    expect(detail.problem?.detail).toBe("same");
  });

  it("offers cancel for queued work on every host", () => {
    expect(model().cancel.available).toBe(true);
  });

  it("offers cancel for running work only where cooperative cancellation is advertised", () => {
    const running = entry({ state: "running" });
    expect(model({ entry: running }).cancel.available).toBe(false);
    expect(model({ entry: running }).cancel.blockedReason).toMatch(/running/i);
    expect(
      model({ entry: running, canCancelRunning: true }).cancel.available,
    ).toBe(true);
  });

  it("offers retry only for a held row the host itself fenced", () => {
    const retryAuthority = {
      instanceId: "i",
      batchId: "b",
      clientBatchId: "c",
      jobId: "job-1",
    };
    expect(model({ retryAuthority }).retry.available).toBe(false);
    const held = entry({ state: "held", retryable: true });
    expect(model({ entry: held, retryAuthority }).retry.available).toBe(true);
    expect(
      model({
        entry: entry({ state: "held", retryable: false }),
        retryAuthority,
      }).retry.available,
    ).toBe(false);
  });

  it("says why a retryable hold cannot be retried from here instead of hiding it", () => {
    const detail = model({ entry: entry({ state: "held", retryable: true }) });
    expect(detail.retry.applicable).toBe(true);
    expect(detail.retry.available).toBe(false);
    expect(detail.retry.blockedReason).toMatch(/submitted/i);
  });

  it("reports queue facts the durable listing carries even with no settings", () => {
    const detail = model({
      entry: entry({
        durable: true,
        replayed: true,
        dispatch_attempts: 1,
      }),
      mine: true,
    });
    const facts = new Map(
      detail.facts.map((field) => [field.label, field.value]),
    );
    expect(facts.get("Host")).toBe("plato");
    expect(facts.get("Durable")).toBe("Yes");
    expect(facts.get("Replayed")).toBe("Yes");
    expect(facts.get("Dispatch attempts")).toBe("1");
    expect(facts.get("Owner")).toBe("This app");
    expect(facts.get("Submitted")).toBeDefined();
  });

  it("polls a preview only for a running row", () => {
    expect(model().preview).toBe(false);
    expect(model({ entry: entry({ state: "running" }) }).preview).toBe(true);
  });

  it("carries the plan's own lane and estimate when it has them", () => {
    const plan: QueuePlan = {
      plan_version: 1,
      state_version: 1,
      optimizer_state: "clean",
      dirty_since_unix_ms: null,
      next_replan_at_unix_ms: null,
      work_items: [
        {
          work_id: "w1",
          parent_id: "job-1",
          work_kind: "generation",
          priority_class: "normal",
          queue_rank: 0,
          bypass_count: 0,
          estimate_confidence: "high",
          gpu: 2,
          estimated_start_unix_ms: 1_700_000_120_000,
        },
      ],
    };
    const facts = new Map(
      model({ plan }).facts.map((field) => [field.label, field.value]),
    );
    expect(facts.get("Lane")).toBe("GPU 2");
    expect(facts.get("Starts in")).toBeDefined();
  });

  it("keeps the raw model id available while displaying the resolved name", () => {
    const detail = model();
    expect(detail.modelLabel).toBe("FLUX.1 [dev] Q8");
    expect(detail.modelId).toBe("flux-dev:q8");
  });
});
