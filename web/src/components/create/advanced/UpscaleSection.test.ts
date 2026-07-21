import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import UpscaleSection from "./UpscaleSection.vue";
import type { ModelInfoExtended } from "../../../types";

const fetchModelsMock = vi.fn();
vi.mock("../../../api", () => ({
  fetchModels: (...args: unknown[]) => fetchModelsMock(...args),
}));

function model(
  name: string,
  overrides: Partial<ModelInfoExtended> = {},
): ModelInfoExtended {
  return {
    name,
    family: "upscaler",
    size_gb: 0.03,
    is_loaded: false,
    last_used: null,
    hf_repo: "",
    downloaded: true,
    default_steps: 0,
    default_guidance: 0,
    default_width: 0,
    default_height: 0,
    description: "",
    ...overrides,
  } as ModelInfoExtended;
}

const UPSCALERS = [
  model("real-esrgan-x4plus:fp16"),
  model("real-esrgan-x2plus:fp16"),
  model("real-esrgan-anime-v3:fp32", {
    downloaded: false,
    description: "fast anime 4x upscaler",
  }),
];

function factory(modelValue = "", models: ModelInfoExtended[] | null = null) {
  return mount(UpscaleSection, {
    props: { modelValue, models },
    global: { stubs: { RouterLink: { template: "<a><slot /></a>" } } },
  });
}

function lastEmit(wrapper: ReturnType<typeof factory>): string | undefined {
  const events = wrapper.emitted("update:modelValue");
  return events?.at(-1)?.[0] as string | undefined;
}

describe("UpscaleSection", () => {
  beforeEach(() => {
    fetchModelsMock.mockReset();
    fetchModelsMock.mockResolvedValue([
      model("flux-dev:q8", { family: "flux" }),
      ...UPSCALERS,
    ]);
  });

  it("starts off with no picker rendered", async () => {
    const wrapper = factory("", UPSCALERS);
    await flushPromises();
    expect(wrapper.find("[data-test='upscale-model']").exists()).toBe(false);
    expect(wrapper.text()).toContain("Off");
  });

  it("selects a real upscaler when the switch is flipped on", async () => {
    const wrapper = factory("", UPSCALERS);
    await flushPromises();
    await wrapper.get("[data-test='upscale-toggle']").trigger("click");

    // The toggle must produce a usable value — not leave the form empty.
    expect(lastEmit(wrapper)).toBe("real-esrgan-x2plus:fp16");

    await wrapper.setProps({ modelValue: "real-esrgan-x2plus:fp16" });
    expect(wrapper.find("[data-test='upscale-model']").exists()).toBe(true);
  });

  it("clears the model when the switch is flipped off", async () => {
    const wrapper = factory("real-esrgan-x4plus:fp16", UPSCALERS);
    await flushPromises();
    expect(wrapper.find("[data-test='upscale-model']").exists()).toBe(true);
    await wrapper.get("[data-test='upscale-toggle']").trigger("click");
    expect(lastEmit(wrapper)).toBe("");
  });

  it("round-trips a different upscaler from the picker", async () => {
    const wrapper = factory("real-esrgan-x4plus:fp16", UPSCALERS);
    await flushPromises();
    const select = wrapper.get("[data-test='upscale-model']");
    (select.element as HTMLSelectElement).value = "real-esrgan-x2plus:fp16";
    await select.trigger("change");
    expect(lastEmit(wrapper)).toBe("real-esrgan-x2plus:fp16");
  });

  it("labels upscalers that are not on disk yet", async () => {
    const wrapper = factory("real-esrgan-x4plus:fp16", UPSCALERS);
    await flushPromises();
    expect(wrapper.get("[data-test='upscale-model']").text()).toContain(
      "downloads on first use",
    );
  });

  it("shows the derived scale factor for the chosen upscaler", async () => {
    const wrapper = factory("real-esrgan-x2plus:fp16", UPSCALERS);
    await flushPromises();
    expect(wrapper.get("[data-test='upscale-factor']").text()).toContain("2");
  });

  it("explains itself instead of offering a dead toggle when the host has none", async () => {
    const wrapper = factory("", [model("flux-dev:q8", { family: "flux" })]);
    await flushPromises();

    const empty = wrapper.get("[data-test='upscale-empty']");
    expect(empty.text().toLowerCase()).toContain("no upscalers");
    expect(wrapper.find("[data-test='upscale-browse']").exists()).toBe(true);

    const toggle = wrapper.get("[data-test='upscale-toggle']");
    expect((toggle.element as HTMLButtonElement).disabled).toBe(true);
    await toggle.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });

  it("keeps an upscaler the host does not list so the request stays visible", async () => {
    // Reused metadata (or a host switch) can name an upscaler this listing
    // lacks — the request still carries it, so the picker must admit that
    // rather than showing an empty state while submitting an upscale.
    const wrapper = factory("real-esrgan-x4plus:fp16", []);
    await flushPromises();
    expect(wrapper.find("[data-test='upscale-empty']").exists()).toBe(false);
    const select = wrapper.get("[data-test='upscale-model']");
    expect(select.text()).toContain("not listed on this server");
    expect((select.element as HTMLSelectElement).value).toBe(
      "real-esrgan-x4plus:fp16",
    );
    await wrapper.get("[data-test='upscale-toggle']").trigger("click");
    expect(lastEmit(wrapper)).toBe("");
  });

  it("fetches the model list itself when the parent does not supply one", async () => {
    const wrapper = factory("", null);
    await flushPromises();
    expect(fetchModelsMock).toHaveBeenCalled();
    await wrapper.get("[data-test='upscale-toggle']").trigger("click");
    expect(lastEmit(wrapper)).toBe("real-esrgan-x2plus:fp16");
  });

  it("still enables once the model list arrives after the switch is flipped", async () => {
    let resolve: (v: ModelInfoExtended[]) => void = () => {};
    fetchModelsMock.mockReturnValue(
      new Promise<ModelInfoExtended[]>((r) => {
        resolve = r;
      }),
    );
    const wrapper = factory("", null);
    await wrapper.get("[data-test='upscale-toggle']").trigger("click");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();

    resolve(UPSCALERS);
    await flushPromises();
    expect(lastEmit(wrapper)).toBe("real-esrgan-x2plus:fp16");
  });

  it("reports an unreachable model list rather than pretending it is empty", async () => {
    fetchModelsMock.mockRejectedValue(new Error("offline"));
    const wrapper = factory("", null);
    await flushPromises();
    expect(wrapper.get("[data-test='upscale-empty']").text()).toContain(
      "could not read",
    );
  });
});
