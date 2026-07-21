import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import MobileExpansionPullStatus from "./MobileExpansionPullStatus.vue";
import type { ExpansionPullView } from "../lib/expansionPull";

function job(status: "queued" | "active" | "completed" | "failed" | "cancelled") {
  return {
    id: "pull-A",
    model: "qwen3-expand:q8",
    status,
    files_done: 1,
    files_total: 2,
    bytes_done: 750_000_000,
    bytes_total: 1_000_000_000,
    current_file: "model.safetensors",
    error: status === "failed" ? "disk full" : null,
  };
}

function mountStatus(status: ExpansionPullView) {
  return mount(MobileExpansionPullStatus, {
    props: {
      model: "qwen3-expand",
      hostLabel: "Studio",
      error: "qwen3-expand isn't installed on Studio.",
      status,
      etaSeconds: 42,
    },
  });
}

describe("MobileExpansionPullStatus", () => {
  it.each([
    ["missing", null],
    ["connecting", null],
    ["starting", null],
    ["queued", job("queued")],
    ["pulling", job("active")],
    ["ready", job("completed")],
    ["failed", job("failed")],
    ["cancelled", job("cancelled")],
  ] as const)("keeps exact model and host visible for %s", (kind, pullJob) => {
    const wrapper = mountStatus({ kind, job: pullJob });
    expect(wrapper.text()).toContain("qwen3-expand");
    expect(wrapper.text()).toContain("Studio");
    expect(wrapper.get('[role="status"]').attributes("aria-live")).toBe("polite");
  });

  it("shows accessible detailed progress and 44pt recovery actions", async () => {
    const pulling = mountStatus({ kind: "pulling", job: job("active") });
    expect(pulling.get('[role="progressbar"]').attributes("aria-valuenow")).toBe("75");
    expect(pulling.text()).toContain("750.0 MB / 1.0 GB");
    expect(pulling.text()).toContain("1/2 files");
    expect(pulling.text()).toContain("ETA 42s");

    const ready = mountStatus({ kind: "ready", job: job("completed") });
    const retry = ready.get('[data-test="mobile-retry-expansion"]');
    expect(retry.classes()).toContain("mobile-touch-action");
    await retry.trigger("click");
    expect(ready.emitted("retry-expansion")).toHaveLength(1);
  });
});
