import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import QueueEntryDetail from "./QueueEntryDetail.vue";
import {
  queueEntryDetailModel,
  type QueueDetailMetadata,
} from "../lib/queueEntryDetail";
import type { QueueEntry } from "../api/queuePlan";

const metadata: QueueDetailMetadata = {
  prompt: "a cat on a porch",
  model: "flux-dev:q8",
  seed: 42,
  steps: 28,
  guidance: 3.5,
  width: 1024,
  height: 1024,
};

function model(entry: Partial<QueueEntry> = {}, extra = {}) {
  return queueEntryDetailModel({
    entry: {
      id: "job-1",
      model: "flux-dev:q8",
      state: "queued",
      started_at_unix_ms: 1_700_000_000_000,
      position: 2,
      ...entry,
    },
    hostLabel: "plato",
    modelLabel: "FLUX.1 [dev] Q8",
    nowMs: 1_700_000_060_000,
    metadata,
    ...extra,
  });
}

describe("QueueEntryDetail", () => {
  it("renders the prompt, the settings groups, and the queue facts", () => {
    const wrapper = mount(QueueEntryDetail, { props: { model: model() } });

    expect(wrapper.get('[data-test="queue-detail-prompt"]').text()).toBe(
      "a cat on a porch",
    );
    expect(
      wrapper.get('[data-test="queue-detail-group-output"]').text(),
    ).toContain("1024×1024");
    expect(wrapper.get('[data-test="queue-detail-facts"]').text()).toContain(
      "plato",
    );
    expect(
      wrapper.find('[data-test="queue-detail-settings-notice"]').exists(),
    ).toBe(false);
  });

  it("never tells anyone to upgrade a server when settings are not loaded yet", () => {
    const wrapper = mount(QueueEntryDetail, {
      props: {
        model: queueEntryDetailModel({
          entry: {
            id: "job-1",
            model: "flux-dev:q8",
            state: "queued",
            started_at_unix_ms: 1,
            position: 0,
          },
          hostLabel: "plato",
          modelLabel: "FLUX",
          nowMs: 2,
        }),
      },
    });

    expect(wrapper.text()).not.toMatch(/upgrade/i);
    expect(
      wrapper.get('[data-test="queue-detail-settings-notice"]').text(),
    ).toMatch(/once this machine loads the job/i);
    expect(
      wrapper.get('[data-test="queue-detail-reuse"]').attributes("disabled"),
    ).toBeDefined();
  });

  it("emits reuse, cancel, and close for the shell to route", async () => {
    const wrapper = mount(QueueEntryDetail, { props: { model: model() } });

    await wrapper.get('[data-test="queue-detail-reuse"]').trigger("click");
    await wrapper.get('[data-test="queue-detail-cancel"]').trigger("click");
    await wrapper.get('[data-test="queue-detail-close"]').trigger("click");
    expect(wrapper.emitted("reuse")).toHaveLength(1);
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("arms a two-step cancel before emitting when the shell asks for it", async () => {
    const wrapper = mount(QueueEntryDetail, {
      props: { model: model(), confirm: "inline" },
    });

    const button = wrapper.get('[data-test="queue-detail-cancel"]');
    await button.trigger("click");
    expect(wrapper.emitted("cancel")).toBeUndefined();
    expect(button.text()).toBe("Cancel job?");
    await button.trigger("click");
    expect(wrapper.emitted("cancel")).toHaveLength(1);
  });

  it("keeps a running row's stop control disabled without cooperative cancellation", () => {
    const wrapper = mount(QueueEntryDetail, {
      props: { model: model({ state: "running" }) },
    });

    expect(
      wrapper.get('[data-test="queue-detail-cancel"]').attributes("disabled"),
    ).toBeDefined();
    expect(wrapper.text()).toContain("cannot stop a running job");
  });

  it("shows a hold in full and offers retry only with the host's fence", () => {
    const held = model({
      state: "held",
      retryable: true,
      held_reason: "dispatch budget exhausted",
      error: "CUDA error: an illegal memory access was encountered",
    });
    const wrapper = mount(QueueEntryDetail, { props: { model: held } });

    expect(wrapper.get('[data-test="queue-detail-problem"]').text()).toContain(
      "illegal memory access",
    );
    expect(
      wrapper.get('[data-test="queue-detail-retry"]').attributes("disabled"),
    ).toBeDefined();
    expect(wrapper.get('[data-test="queue-detail-retry-hint"]').text()).toMatch(
      /submitted/i,
    );
  });

  it("renders the running preview and its step counter", () => {
    const wrapper = mount(QueueEntryDetail, {
      props: {
        model: model({ state: "running" }),
        preview: { preview_image: "AAA", step: 7, total: 28 },
      },
    });

    expect(wrapper.get('[data-test="queue-detail-preview"]').text()).toContain(
      "Step 7 of 28",
    );
    expect(wrapper.get('[data-test="queue-detail-preview"] img').attributes("src")).toBe(
      "data:image/png;base64,AAA",
    );
  });

  it("keeps the step counter on a host that renders without previews", () => {
    const wrapper = mount(QueueEntryDetail, {
      props: {
        model: model({ state: "running" }),
        preview: { preview_image: null, step: 7, total: 28 },
      },
    });

    expect(wrapper.get('[data-test="queue-detail-preview"]').text()).toContain(
      "Step 7 of 28",
    );
    expect(wrapper.find('[data-test="queue-detail-preview"] img').exists()).toBe(
      false,
    );
  });

  it("shows an action failure inline instead of leaving it to a toast", () => {
    const wrapper = mount(QueueEntryDetail, {
      props: { model: model(), error: "plato refused the cancellation" },
    });

    expect(wrapper.get('[data-test="queue-detail-error"]').text()).toBe(
      "plato refused the cancellation",
    );
  });
});
