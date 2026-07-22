import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ControlsAside from "./ControlsAside.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import Stepper from "@ui/components/Stepper.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import { dimsForMp } from "@ui/lib/resolution";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import {
  addHost,
  getGenerateTargetId,
  setGenerateTargetId,
} from "../../lib/hostRegistry";
import { __testing__ as routingTesting } from "../../composables/useHostRouting";
import { CAPABLE_TARGET_ID } from "../../lib/hostRouting";
import type { GenerateFormState } from "../../types";

const pushMock = vi.hoisted(() => vi.fn());
vi.mock("vue-router", () => ({
  useRouter: () => ({ push: pushMock }),
}));

// The rail's host picker polls every registered machine. Keep the unit under
// test off the network: no host answers, so rows render from the registry with
// their pre-poll status.
vi.mock("../machines/hostClient", () => ({
  hostStatus: () => Promise.reject(new Error("offline in tests")),
  hostModels: () => Promise.reject(new Error("offline in tests")),
}));

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, ...overrides };
}

function factory(overrides: Partial<GenerateFormState> = {}, family = "flux") {
  return mount(ControlsAside, {
    props: { modelValue: baseForm(overrides), family, advCount: 0 },
  });
}

describe("ControlsAside", () => {
  beforeEach(() => {
    localStorage.clear();
    pushMock.mockClear();
    routingTesting.reset();
  });
  afterEach(() => __testing__.resetForTest());

  it("projects the current pixels onto Shape and Resolution", () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    expect(wrapper.getComponent(ShapePicker).props("modelValue")).toBe(
      "square",
    );
    expect(wrapper.getComponent(ResolutionSelector).props("modelValue")).toBe(
      1,
    );
  });

  it("keeps the full SDXL guidance range reachable", () => {
    const wrapper = factory({}, "sdxl");
    const strength = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Prompt strength");
    expect(strength?.props("max")).toBe(20);
  });

  it("applies the projected dims when a shape is picked", async () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    wrapper
      .getComponent(ShapePicker)
      .vm.$emit("update:modelValue", "landscape");
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    const expected = dimsForMp(1, 4 / 3);
    expect(next.width).toBe(expected.width);
    expect(next.height).toBe(expected.height);
  });

  it("applies the projected dims when resolution changes", async () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    wrapper.getComponent(ResolutionSelector).vm.$emit("update:modelValue", 2);
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    const expected = dimsForMp(2, 1);
    expect(next.width).toBe(expected.width);
    expect(next.height).toBe(expected.height);
  });

  function seedButton(wrapper: ReturnType<typeof factory>, label: string) {
    return wrapper
      .findAll("[data-test='seed-seg'] button")
      .find((b) => b.text() === label)!;
  }

  it("maps the seed control to Random/Fixed and reveals the seed input when fixed", async () => {
    const wrapper = factory({ seedMode: "random", seed: null });
    expect(wrapper.find("[data-test='controls-seed']").exists()).toBe(false);

    await seedButton(wrapper, "Fixed").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.seedMode).toBe("static");
  });

  it("keeps an increment seed under the Fixed segment", () => {
    const wrapper = factory({ seedMode: "increment", seed: 100 });
    // Fixed covers both static and increment, so the seed input is shown.
    expect(wrapper.find("[data-test='controls-seed']").exists()).toBe(true);
  });

  it("steps the batch size", async () => {
    const wrapper = factory({ batchSize: 1 });
    wrapper.getComponent(Stepper).vm.$emit("update:modelValue", 3);
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.batchSize).toBe(3);
  });

  it("shows the advanced badge and opens the sheet on phones", async () => {
    // On tablet+ the Advanced sections render inline, so the sheet button is
    // phone-only (mobile: true).
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(),
        family: "flux",
        advCount: 3,
        mobile: true,
      },
    });
    expect(wrapper.get("[data-test='adv-badge']").text()).toContain("3");
    await wrapper.get("[data-test='open-advanced']").trigger("click");
    expect(wrapper.emitted("open-advanced")).toHaveLength(1);
  });

  it("hides the Advanced sheet button on tablet+ (inline advanced instead)", () => {
    const wrapper = mount(ControlsAside, {
      props: { modelValue: baseForm(), family: "flux", advCount: 3 },
    });
    expect(wrapper.find("[data-test='open-advanced']").exists()).toBe(false);
  });

  it("locks batch to 1 for edit families", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(),
        family: "qwen-image-edit",
        advCount: 0,
      },
    });
    expect(wrapper.find("[data-test='batch-locked']").exists()).toBe(true);
    expect(wrapper.getComponent(Stepper).props("max")).toBe(1);
  });

  it("reroll switches seed back to random", async () => {
    const wrapper = factory({ seedMode: "static", seed: 42 });
    await wrapper.get("[data-test='seed-reroll']").trigger("click");
    const events = wrapper.emitted("update:modelValue") ?? [];
    const last = events.at(-1)?.[0] as GenerateFormState;
    expect(last.seedMode).toBe("random");
    expect(last.seed).toBeNull();
  });

  it("collapses the run-on row to this server with no remote machines", () => {
    const wrapper = factory();
    expect(wrapper.get("[data-test='controls-host']").text()).toContain(
      "Run on this server",
    );
    expect(wrapper.find("[data-test='host-chip']").exists()).toBe(false);
  });

  it("opens the machines workspace when the collapsed host row is clicked", async () => {
    const wrapper = factory();
    await wrapper.get("[data-test='controls-host']").trigger("click");
    expect(pushMock).toHaveBeenCalledWith("/machines");
  });

  it("offers the routing menu once a remote machine is registered", async () => {
    const host = addHost({ url: "http://studio:7680", name: "Studio" });
    const wrapper = factory();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-option-auto']").exists()).toBe(true);
    expect(wrapper.find(`[data-test='host-option-${host.id}']`).exists()).toBe(
      true,
    );
  });

  it("persists a routing pick made from the rail", async () => {
    addHost({ url: "http://studio:7680", name: "Studio" });
    const wrapper = factory();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    await wrapper.get("[data-test='host-option-capable']").trigger("click");
    expect(getGenerateTargetId()).toBe(CAPABLE_TARGET_ID);
  });

  it("names an already-persisted sticky pick on the chip", async () => {
    const host = addHost({ url: "http://studio:7680", name: "Studio" });
    setGenerateTargetId(host.id);
    const wrapper = factory();
    await wrapper.vm.$nextTick();
    expect(wrapper.get("[data-test='host-chip']").text()).toContain(
      "Run on Studio",
    );
  });
});
