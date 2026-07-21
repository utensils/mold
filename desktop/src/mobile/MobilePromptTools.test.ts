import { flushPromises, mount } from "@vue/test-utils";
import { reactive } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchHistoryFrom } from "../lib/api/history";
import { newGenerateForm } from "../lib/generateForm";
import MobilePromptTools from "./MobilePromptTools.vue";

vi.mock("../lib/api/history", () => ({ fetchHistoryFrom: vi.fn() }));

const target = { baseUrl: "https://studio.tailnet.ts.net", apiKey: "secret" };

describe("MobilePromptTools", () => {
  beforeEach(() => {
    vi.mocked(fetchHistoryFrom)
      .mockReset()
      .mockResolvedValue([{ prompt: "older storm prompt", model: "flux:fp16", used_at: 2 }]);
  });

  it("emits Batch 1 Expand and Undo while MobileApp owns the frozen snapshot", async () => {
    const form = reactive(newGenerateForm());
    form.prompt = "small owl";
    const wrapper = mount(MobilePromptTools, {
      props: { form, target, running: false, canUndo: true, blocked: false },
    });

    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await wrapper.get("[data-test='mobile-prompt-undo']").trigger("click");
    expect(wrapper.emitted("expand")).toHaveLength(1);
    expect(wrapper.emitted("undo")).toHaveLength(1);
    expect(wrapper.find("[data-test='mobile-prompt-options']").exists()).toBe(false);
  });

  it("turns Expand into one Batch-aware preparation action", async () => {
    const form = reactive(newGenerateForm());
    form.prompt = "storm";
    form.batchSize = 3;
    const wrapper = mount(MobilePromptTools, {
      props: { form, target, running: false, canUndo: false, blocked: false },
    });

    expect(wrapper.get("[data-test='mobile-prompt-expand']").text()).toBe("Prepare 3 variations");
    await wrapper.setProps({ running: true });
    expect(wrapper.get("[data-test='mobile-prompt-expand']").text()).toBe(
      "Preparing 3 variations…",
    );
  });

  it("preserves Recent prompts and refreshes them for the selected target", async () => {
    const form = reactive(newGenerateForm());
    const wrapper = mount(MobilePromptTools, {
      props: { form, target, running: false, canUndo: false, blocked: false },
    });
    await wrapper.get("[data-test='mobile-prompt-recent']").trigger("click");
    await flushPromises();
    expect(fetchHistoryFrom).toHaveBeenCalledWith(target, "", 20);
    await wrapper.get("[data-test='mobile-prompt-history-item']").trigger("click");
    expect(form.prompt).toBe("older storm prompt");
    expect(form.originalPrompt).toBeNull();

    await wrapper.get("[data-test='mobile-prompt-recent']").trigger("click");
    const otherTarget = { baseUrl: "https://render.tailnet.ts.net", apiKey: "other" };
    vi.mocked(fetchHistoryFrom).mockResolvedValue([
      { prompt: "prompt from render", model: "sdxl:fp16", used_at: 3 },
    ]);
    await wrapper.setProps({ target: otherTarget });
    await flushPromises();
    expect(fetchHistoryFrom).toHaveBeenLastCalledWith(otherTarget, "", 20);
    expect(wrapper.get("[data-test='mobile-prompt-history-item']").text()).toContain(
      "prompt from render",
    );
  });
});
