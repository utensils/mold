import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import DownloadsDrawer from "./DownloadsDrawer.vue";

const makeJob = (over: Record<string, unknown> = {}) => ({
  id: "a",
  model: "flux-dev:q4",
  status: "active",
  files_done: 1,
  files_total: 5,
  bytes_done: 500_000,
  bytes_total: 1_000_000,
  current_file: "transformer.gguf",
  started_at: Date.now(),
  completed_at: null,
  error: null,
  ...over,
});

describe("DownloadsDrawer", () => {
  it("renders active job with progress text", () => {
    const wrapper = mount(DownloadsDrawer, {
      props: {
        open: true,
        active: makeJob() as never,
        queued: [],
        history: [],
        etaSeconds: 5,
      },
    });
    expect(wrapper.text()).toContain("flux-dev:q4");
    expect(wrapper.text()).toContain("transformer.gguf");
    expect(wrapper.text()).toMatch(/1\s*\/\s*5/);
  });

  it("renders queued chips with position", () => {
    const wrapper = mount(DownloadsDrawer, {
      props: {
        open: true,
        active: null,
        queued: [
          makeJob({ id: "q1", model: "sd1.5:fp16", status: "queued" }) as never,
          makeJob({
            id: "q2",
            model: "flux-schnell:q4",
            status: "queued",
          }) as never,
        ],
        history: [],
        etaSeconds: null,
      },
    });
    expect(wrapper.text()).toContain("sd1.5:fp16");
    expect(wrapper.text()).toContain("flux-schnell:q4");
    expect(wrapper.text()).toMatch(/#1/);
    expect(wrapper.text()).toMatch(/#2/);
  });

  it("shows retry button for failed history entries", () => {
    const onRetry = vi.fn();
    const wrapper = mount(DownloadsDrawer, {
      props: {
        open: true,
        active: null,
        queued: [],
        history: [
          makeJob({
            id: "h1",
            status: "failed",
            error: "network blip",
          }) as never,
        ],
        etaSeconds: null,
        onRetry,
      },
    });
    const btn = wrapper.get("[data-test=retry-h1]");
    btn.trigger("click");
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
