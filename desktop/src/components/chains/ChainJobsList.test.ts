import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../../lib/api/client", () => ({
  apiFetch: vi.fn().mockResolvedValue({}),
  apiFetchTo: vi.fn(),
  apiJson: vi.fn().mockResolvedValue({ jobs: [] }),
  apiJsonTo: vi.fn(),
}));
vi.mock("../../lib/api/sse", () => ({ sseStream: vi.fn().mockResolvedValue(undefined) }));

import ChainJobsList from "./ChainJobsList.vue";
import { useChainJobsStore } from "../../stores/chainJobs";

function job(id: string, state: string) {
  return {
    id,
    state,
    model: "ltx-video-0.9.6:bf16",
    stage_count: 2,
    current_stage: 1,
    created_at_unix_ms: 1,
    updated_at_unix_ms: 2,
    error: null,
    ephemeral: false,
  };
}

beforeEach(() => setActivePinia(createPinia()));

describe("ChainJobsList clearing", () => {
  it("offers Clear finished only when terminal jobs exist and clears them", async () => {
    const chains = useChainJobsStore();
    chains.jobs = [job("a", "completed"), job("b", "running"), job("c", "failed")] as never;
    const clear = vi.spyOn(chains, "clearFinished").mockResolvedValue({ cleared: 2, failed: 0 });

    const wrapper = mount(ChainJobsList);
    await wrapper.get("[data-test='clear-finished']").trigger("click");
    await flushPromises();
    expect(clear).toHaveBeenCalledOnce();
  });

  it("hides Clear finished while only active jobs remain", () => {
    const chains = useChainJobsStore();
    chains.jobs = [job("a", "running"), job("b", "queued")] as never;

    const wrapper = mount(ChainJobsList);
    expect(wrapper.find("[data-test='clear-finished']").exists()).toBe(false);
  });

  it("labels the disk sweep as words, not a GC glyph", () => {
    const chains = useChainJobsStore();
    chains.jobs = [job("a", "completed")] as never;

    const wrapper = mount(ChainJobsList);
    const sweep = wrapper.get("[data-test='jobs-cleanup']");
    expect(sweep.text()).toBe("Clean up disk");
    expect(wrapper.text()).not.toMatch(/\bGC\b/);
  });
});
