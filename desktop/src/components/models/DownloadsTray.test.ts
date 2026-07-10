import { beforeEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import DownloadsTray from "./DownloadsTray.vue";
import { useDownloadsStore } from "../../stores/downloads";
import type { DownloadJob } from "../../lib/api/types";

function job(overrides: Partial<DownloadJob> = {}): DownloadJob {
  return {
    id: "j1",
    model: "flux-dev:q4",
    status: "active",
    files_done: 1,
    files_total: 4,
    bytes_done: 25,
    bytes_total: 100,
    ...overrides,
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("DownloadsTray a11y", () => {
  it("exposes each download as a determinate progressbar", () => {
    const store = useDownloadsStore();
    store.active = job();
    const wrapper = mount(DownloadsTray);

    const bar = wrapper.get('[role="progressbar"]');
    expect(bar.attributes("aria-valuemin")).toBe("0");
    expect(bar.attributes("aria-valuemax")).toBe("100");
    expect(bar.attributes("aria-valuenow")).toBe("25");
    expect(bar.attributes("aria-label")).toContain("flux-dev:q4");
  });

  it("labels the cancel control with the model name", () => {
    const store = useDownloadsStore();
    store.active = job();
    const wrapper = mount(DownloadsTray);

    const cancel = wrapper.get("button");
    expect(cancel.attributes("aria-label")).toBe("Cancel download of flux-dev:q4");
  });
});
