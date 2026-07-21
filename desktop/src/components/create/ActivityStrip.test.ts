import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import ActivityStrip from "./ActivityStrip.vue";
import { useGenerationStore } from "../../stores/generation";
import type { Job } from "../../stores/generation";

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

function baseJob(): Job {
  return {
    clientId: 1,
    prompt: "a lighthouse",
    status: "queued",
    step: 0,
    total: 10,
    error: null,
  } as Job;
}

describe("ActivityStrip", () => {
  it("is hidden when nothing is in flight", () => {
    const wrapper = mount(ActivityStrip);
    expect(wrapper.find("[data-test='activity-strip']").exists()).toBe(false);
  });

  it("mirrors the running print with its prompt and percent", () => {
    const generation = useGenerationStore();
    generation.jobs = [{ ...baseJob(), status: "denoising", step: 5, total: 10 }];
    const wrapper = mount(ActivityStrip);
    expect(wrapper.get("[data-test='activity-strip']").text()).toContain("a lighthouse");
    expect(wrapper.text()).toContain("50%");
  });

  it("lists queued siblings with a working cancel", async () => {
    const generation = useGenerationStore();
    const cancel = vi.spyOn(generation, "cancel").mockResolvedValue(undefined as never);
    generation.jobs = [{ ...baseJob(), clientId: 7, status: "queued", prompt: "queued one" }];
    const wrapper = mount(ActivityStrip);
    const pill = wrapper.get("[data-test='activity-queued']");
    expect(pill.text()).toContain("queued one");
    await pill.get("button").trigger("click");
    expect(cancel).toHaveBeenCalledWith(7);
  });
});
