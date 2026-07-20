import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ExpansionPullStatus from "./ExpansionPullStatus.vue";
import type { ExpansionPullView } from "../../lib/expansionPull";

const wrappers: ReturnType<typeof mount>[] = [];
function mountStatus(status: ExpansionPullView) {
  const wrapper = mount(ExpansionPullStatus, {
    attachTo: document.body,
    props: {
      model: "qwen3-expand:q8",
      hostLabel: "Studio GPU",
      error: "The expansion model qwen3-expand:q8 isn't installed on Studio GPU.",
      status,
      etaSeconds: 42,
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

afterEach(() => {
  while (wrappers.length) wrappers.pop()!.unmount();
});

describe("ExpansionPullStatus", () => {
  it("keeps the initial missing-model recovery inline and keyboard reachable", async () => {
    const wrapper = mountStatus({ kind: "missing", job: null });
    const button = wrapper.get('[data-test="pull-expand-model"]');
    expect(button.element.tagName).toBe("BUTTON");
    expect(button.text()).toBe("Pull expansion model");
    await button.trigger("click");
    expect(wrapper.emitted("pull")).toHaveLength(1);
  });

  it.each([
    ["missing", null],
    ["connecting", null],
    ["starting", null],
    ["queued", downloadJob("queued")],
    ["pulling", downloadJob("active")],
    ["ready", downloadJob("completed")],
    ["failed", downloadJob("failed", { error: "disk full" })],
    ["cancelled", downloadJob("cancelled")],
  ] as const)("keeps exact model and host identity visible in the %s state", (kind, job) => {
    const wrapper = mountStatus({ kind, job });
    expect(wrapper.get('[role="status"]').attributes("aria-live")).toBe("polite");
    expect(wrapper.text()).toContain("qwen3-expand:q8");
    expect(wrapper.text()).toContain("Studio GPU");
  });

  it("renders exact pull progress, bytes, file detail, ETA, and progressbar semantics", () => {
    const wrapper = mountStatus({
      kind: "pulling",
      job: downloadJob("active", {
        bytes_done: 750_000_000,
        bytes_total: 1_000_000_000,
        files_done: 1,
        files_total: 2,
        current_file: "text_encoder/model.safetensors",
      }),
    });

    const progress = wrapper.get('[role="progressbar"]');
    expect(progress.attributes("aria-valuemin")).toBe("0");
    expect(progress.attributes("aria-valuemax")).toBe("100");
    expect(progress.attributes("aria-valuenow")).toBe("75");
    expect(wrapper.text()).toContain("75%");
    expect(wrapper.text()).toContain("750.0 MB / 1.0 GB");
    expect(wrapper.text()).toContain("1/2 files");
    expect(wrapper.text()).toContain("text_encoder/model.safetensors");
    expect(wrapper.text()).toContain("ETA 42s");
  });

  it("offers expansion retry only after the exact pull is ready", async () => {
    const wrapper = mountStatus({ kind: "ready", job: downloadJob("completed") });
    const button = wrapper.get('[data-test="retry-expansion"]');
    expect(button.attributes("aria-label")).toContain("qwen3-expand:q8");
    expect(button.attributes("aria-label")).toContain("Studio GPU");
    await button.trigger("click");
    expect(wrapper.emitted("retry-expansion")).toHaveLength(1);
  });

  it.each(["failed", "cancelled"] as const)(
    "offers a same-host pull retry after %s",
    async (kind) => {
      const wrapper = mountStatus({
        kind,
        job: downloadJob(kind, { error: kind === "failed" ? "disk full" : null }),
      });
      expect(wrapper.text()).toContain("Studio GPU");
      const button = wrapper.get('[data-test="retry-expand-pull"]');
      expect(button.text()).toContain("qwen3-expand:q8");
      expect(button.text()).toContain("Studio GPU");
      await button.trigger("click");
      expect(wrapper.emitted("pull")).toHaveLength(1);
    },
  );
});

function downloadJob(
  status: "queued" | "active" | "completed" | "failed" | "cancelled",
  overrides: Record<string, unknown> = {},
) {
  return {
    id: "pull-1",
    model: "qwen3-expand:q8",
    status,
    files_done: 0,
    files_total: 0,
    bytes_done: 0,
    bytes_total: 0,
    ...overrides,
  };
}
