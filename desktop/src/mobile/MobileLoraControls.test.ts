import { flushPromises, mount } from "@vue/test-utils";
import { reactive } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchLoras } from "../lib/api/loras";
import { newGenerateForm } from "../lib/generateForm";
import MobileLoraControls from "./MobileLoraControls.vue";

vi.mock("../lib/api/loras", () => ({ fetchLoras: vi.fn() }));

const target = { baseUrl: "https://studio.tailnet.ts.net", apiKey: "secret" };
const available = {
  id: "cinematic",
  name: "Cinematic light",
  family: "flux",
  path: "loras/cinematic.safetensors",
  trained_words: ["cinematic glow"],
  added_at: 1,
};

describe("MobileLoraControls", () => {
  beforeEach(() => {
    vi.mocked(fetchLoras).mockReset().mockResolvedValue([available]);
  });

  it("loads LoRAs from the selected remote host and edits the stack", async () => {
    const form = reactive(newGenerateForm());
    form.model = "flux:fp16";
    form.family = "flux";
    const wrapper = mount(MobileLoraControls, { props: { form, target } });

    await wrapper.get("[data-test='mobile-lora-disclosure']").trigger("click");
    await flushPromises();
    expect(fetchLoras).toHaveBeenCalledWith(form.model, target);
    expect(wrapper.get("[data-test='mobile-lora-option']").text()).toContain("Cinematic light");

    await wrapper.get("[data-test='mobile-lora-option']").trigger("click");
    expect(form.loras).toEqual([
      {
        path: available.path,
        name: available.name,
        scale: 1,
        trainedWords: available.trained_words,
      },
    ]);
    expect(wrapper.get("[data-test='mobile-lora-disclosure']").text()).toContain("1 active");

    await wrapper.get("[data-test='mobile-lora-scale-0']").setValue("0.65");
    expect(form.loras[0]?.scale).toBe(0.65);
    await wrapper.get("[data-test='mobile-lora-trigger-0-0']").trigger("click");
    expect(wrapper.emitted("append-word")).toEqual([["cinematic glow"]]);

    await wrapper.get("[data-test='mobile-lora-remove-0']").trigger("click");
    expect(form.loras).toEqual([]);
  });

  it("surfaces load failures without discarding an existing stack", async () => {
    vi.mocked(fetchLoras).mockRejectedValue(new Error("host unavailable"));
    const form = reactive(newGenerateForm());
    form.model = "flux:fp16";
    form.family = "flux";
    form.loras = [{ path: available.path, name: available.name, scale: 0.8, trainedWords: [] }];
    const wrapper = mount(MobileLoraControls, { props: { form, target } });

    await flushPromises();

    expect(wrapper.get("[data-test='mobile-lora-error']").text()).toContain("host unavailable");
    expect(form.loras).toHaveLength(1);
  });

  it("reloads the available adapters when generation moves to another host", async () => {
    const form = reactive(newGenerateForm());
    form.model = "flux:fp16";
    form.family = "flux";
    const wrapper = mount(MobileLoraControls, { props: { form, target } });

    await wrapper.get("[data-test='mobile-lora-disclosure']").trigger("click");
    await flushPromises();

    const otherTarget = { baseUrl: "https://render.tailnet.ts.net", apiKey: "other" };
    const other = { ...available, id: "film", name: "Film grain", path: "loras/film.safetensors" };
    vi.mocked(fetchLoras).mockResolvedValue([other]);
    await wrapper.setProps({ target: otherTarget });
    await flushPromises();

    expect(fetchLoras).toHaveBeenLastCalledWith(form.model, otherTarget);
    expect(wrapper.get("[data-test='mobile-lora-option']").text()).toContain("Film grain");
  });
});
