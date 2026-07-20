import { flushPromises, mount } from "@vue/test-utils";
import { nextTick, reactive } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { expandPrompt } from "../lib/api/expand";
import { fetchHistoryFrom } from "../lib/api/history";
import { newGenerateForm } from "../lib/generateForm";
import MobilePromptTools from "./MobilePromptTools.vue";

vi.mock("../lib/api/expand", () => ({ expandPrompt: vi.fn() }));
vi.mock("../lib/api/history", () => ({ fetchHistoryFrom: vi.fn() }));

const target = { baseUrl: "https://studio.tailnet.ts.net", apiKey: "secret" };

describe("MobilePromptTools", () => {
  beforeEach(() => {
    vi.mocked(expandPrompt)
      .mockReset()
      .mockResolvedValue({
        original: "small owl",
        expanded: ["a small owl under cinematic moonlight"],
      });
    vi.mocked(fetchHistoryFrom)
      .mockReset()
      .mockResolvedValue([{ prompt: "older storm prompt", model: "flux:fp16", used_at: 2 }]);
  });

  it("expands and restores a prompt on the selected host", async () => {
    const form = reactive(newGenerateForm());
    form.prompt = "small owl";
    form.family = "flux";
    const wrapper = mount(MobilePromptTools, { props: { form, target } });

    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    expect(expandPrompt).toHaveBeenCalledWith(
      "small owl",
      { variations: 1, modelFamily: "flux" },
      target,
    );
    expect(form.prompt).toBe("a small owl under cinematic moonlight");
    expect(form.originalPrompt).toBe("small owl");

    await wrapper.get("[data-test='mobile-prompt-undo']").trigger("click");
    expect(form.prompt).toBe("small owl");
    expect(form.originalPrompt).toBeNull();
  });

  it("previews variations and recalls recent prompts without hidden state", async () => {
    vi.mocked(expandPrompt).mockResolvedValue({
      original: "owl",
      expanded: ["candidate one", "candidate two", "candidate three"],
    });
    const form = reactive(newGenerateForm());
    form.prompt = "owl";
    form.family = "sdxl";
    const wrapper = mount(MobilePromptTools, { props: { form, target } });

    await wrapper.get("[data-test='mobile-prompt-options']").trigger("click");
    await wrapper.get("[data-test='mobile-prompt-variations-3']").trigger("click");
    await wrapper.get("[data-test='mobile-prompt-preview']").trigger("click");
    await flushPromises();
    expect(expandPrompt).toHaveBeenCalledWith(
      "owl",
      { variations: 3, modelFamily: "sdxl" },
      target,
    );
    expect(wrapper.findAll("[data-test='mobile-prompt-candidate']")).toHaveLength(3);
    await wrapper.findAll("[data-test='mobile-prompt-candidate']")[1]?.trigger("click");
    expect(form.prompt).toBe("candidate two");
    expect(form.originalPrompt).toBe("owl");

    await wrapper.get("[data-test='mobile-prompt-recent']").trigger("click");
    await flushPromises();
    expect(fetchHistoryFrom).toHaveBeenCalledWith(target, "", 20);
    await wrapper.get("[data-test='mobile-prompt-history-item']").trigger("click");
    expect(form.prompt).toBe("older storm prompt");
    expect(form.originalPrompt).toBeNull();
  });

  it("shows directed errors while leaving the prompt unchanged", async () => {
    vi.mocked(expandPrompt).mockRejectedValue(new Error("expansion model unavailable"));
    const form = reactive(newGenerateForm());
    form.prompt = "owl";
    const wrapper = mount(MobilePromptTools, { props: { form, target } });

    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-prompt-tools-error']").text()).toContain(
      "expansion model unavailable",
    );
    expect(form.prompt).toBe("owl");
  });

  it("ignores late expansion results and clears candidates after the form changes", async () => {
    let resolveExpansion!: (value: { original: string; expanded: string[] }) => void;
    vi.mocked(expandPrompt).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveExpansion = resolve;
        }),
    );
    const form = reactive(newGenerateForm());
    form.prompt = "owl";
    form.family = "flux";
    const wrapper = mount(MobilePromptTools, { props: { form, target } });

    await wrapper.get("[data-test='mobile-prompt-expand']").trigger("click");
    form.prompt = "owl typed while waiting";
    await nextTick();
    resolveExpansion({ original: "owl", expanded: ["stale expanded owl"] });
    await flushPromises();

    expect(form.prompt).toBe("owl typed while waiting");
    expect(form.originalPrompt).toBeNull();
    expect(
      wrapper.get("[data-test='mobile-prompt-expand']").attributes("disabled"),
    ).toBeUndefined();

    vi.mocked(expandPrompt).mockResolvedValueOnce({
      original: "owl typed while waiting",
      expanded: ["candidate from old model"],
    });
    await wrapper.get("[data-test='mobile-prompt-options']").trigger("click");
    await wrapper.get("[data-test='mobile-prompt-preview']").trigger("click");
    await flushPromises();
    expect(wrapper.findAll("[data-test='mobile-prompt-candidate']")).toHaveLength(1);

    form.family = "sdxl";
    await nextTick();
    expect(wrapper.findAll("[data-test='mobile-prompt-candidate']")).toHaveLength(0);
  });

  it("refreshes an open recent-prompt panel when the selected host changes", async () => {
    const form = reactive(newGenerateForm());
    const wrapper = mount(MobilePromptTools, { props: { form, target } });

    await wrapper.get("[data-test='mobile-prompt-recent']").trigger("click");
    await flushPromises();

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
