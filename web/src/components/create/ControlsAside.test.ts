import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ControlsAside from "./ControlsAside.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import Stepper from "@ui/components/Stepper.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
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
  hostCapabilities: () => Promise.reject(new Error("offline in tests")),
  hostQueue: () => Promise.resolve({ entries: [], plan: null }),
  hostDevices: () => Promise.reject(new Error("offline in tests")),
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

  it("fixes distilled LTX guidance while guided pipelines remain editable", async () => {
    const wrapper = factory(
      { model: "ltx-2.3-22b-distilled:fp8", modelFamily: "ltx2", guidance: 7 },
      "ltx2",
    );
    const guidance = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Prompt strength")!;
    expect(guidance.props("disabled")).toBe(true);
    expect(guidance.props("modelValue")).toBe(1);
    expect(wrapper.get("[data-test='fixed-guidance-hint']").text()).toContain(
      "fixes CFG at 1.0",
    );

    await wrapper.setProps({
      modelValue: {
        ...(wrapper.props("modelValue") as GenerateFormState),
        pipeline: "two-stage",
      },
    });
    expect(
      wrapper
        .findAllComponents(SliderRow)
        .find((row) => row.props("label") === "Prompt strength")!
        .props("disabled"),
    ).toBe(false);
  });

  it("projects the current pixels onto Shape and Resolution", () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    expect(wrapper.getComponent(ShapePicker).props("modelValue")).toBe(
      "square",
    );
    expect(wrapper.getComponent(ResolutionSelector).props("modelValue")).toBe(
      (1024 * 1024) / 1_000_000,
    );
  });

  it("renders a custom source canvas and restores it after a manual override", async () => {
    const sourceDimensions = { width: 896, height: 1152 };
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(sourceDimensions),
        family: "qwen-image-edit",
        advCount: 0,
        sourceDimensions,
      },
    });

    expect(wrapper.getComponent(ShapePicker).props("modelValue")).toBe(
      "source",
    );
    expect(wrapper.getComponent(ResolutionSelector).props()).toMatchObject({
      resolvedWidth: 896,
      resolvedHeight: 1152,
      customLabel: "Source",
      status: "Matches source · 896×1152",
    });
    expect(wrapper.find("[data-test='match-source-resolution']").exists()).toBe(
      false,
    );

    await wrapper.setProps({
      modelValue: baseForm({ width: 1024, height: 1024 }),
    });
    expect(wrapper.getComponent(ResolutionSelector).props("status")).toContain(
      "manual output",
    );
    await wrapper.get("[data-test='match-source-resolution']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next).toMatchObject(sourceDimensions);
  });

  it("renders the profile-compatible Detail admission range", () => {
    const wrapper = factory({}, "sdxl");
    const detail = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Detail");
    expect(detail?.props("min")).toBe(1);
    expect(detail?.props("max")).toBe(100);
  });

  it("renders the profile-compatible guidance admission range", () => {
    const wrapper = factory({}, "sdxl");
    const strength = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Prompt strength");
    expect(strength?.props("min")).toBe(0);
    expect(strength?.props("max")).toBe(100);
    expect(strength?.props("step")).toBe(0.1);
  });

  it("shows a per-model duration slider for one-shot video", async () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ frames: 97, fps: 24 }),
        family: "ltx2",
        model: {
          name: "ltx-2-19b-distilled:fp8",
          family: "ltx2",
          default_frames: 97,
          default_fps: 24,
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        } as never,
      },
    });
    const duration = wrapper.getComponent(VideoDurationSlider);
    expect(duration.text()).toContain("4.0s");
    duration.vm.$emit("update:frames", 241);
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      frames: 241,
    });
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
    expect(next.width).toBe(1024);
    expect(next.height).toBe(768);
  });

  it("hides aspect ratios the selected wan checkpoint does not support", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "wan22-i2v-a14b:q5",
          modelFamily: "wan",
          width: 480,
          height: 832,
        }),
        family: "wan",
        advCount: 0,
        model: {
          name: "wan22-i2v-a14b:q5",
          family: "wan",
          recommended_dimensions: [
            { width: 832, height: 480 },
            { width: 480, height: 832 },
          ],
          dimension_alignment: 16,
          max_pixels: 1280 * 720,
        } as never,
      },
    });

    expect(wrapper.getComponent(ShapePicker).props("options")).toEqual([
      expect.objectContaining({ id: "26:15", label: "26:15" }),
      expect.objectContaining({ id: "15:26", label: "15:26" }),
    ]);
  });

  it("exposes and applies Z-Image's exact 16:9 and 9:16 buckets", async () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "z-image-turbo:q4",
          modelFamily: "z-image",
          width: 1024,
          height: 1024,
        }),
        family: "z-image",
        model: {
          name: "z-image-turbo:q4",
          family: "z-image",
          recommended_dimensions: [
            { width: 1024, height: 1024 },
            { width: 1280, height: 720 },
            { width: 720, height: 1280 },
          ],
          dimension_alignment: 16,
          max_pixels: 1_800_000,
        } as never,
      },
    });
    const picker = wrapper.getComponent(ShapePicker);
    expect(picker.props("options")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "16:9", label: "16:9" }),
        expect.objectContaining({ id: "9:16", label: "9:16" }),
      ]),
    );
    picker.vm.$emit("update:modelValue", "16:9");
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      width: 1280,
      height: 720,
    });
    await wrapper.setProps({
      modelValue: baseForm({
        model: "z-image-turbo:q4",
        modelFamily: "z-image",
        width: 1280,
        height: 720,
      }),
    });
    picker.vm.$emit("update:modelValue", "9:16");
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      width: 720,
      height: 1280,
    });
  });

  it("applies the projected dims when resolution changes", async () => {
    const wrapper = factory({ width: 1024, height: 1024 }, "flux");
    wrapper
      .getComponent(ResolutionSelector)
      .vm.$emit("update:modelValue", (768 * 768) / 1_000_000);
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.width).toBe(768);
    expect(next.height).toBe(768);
  });

  it("labels a model's only runnable bucket as Native", () => {
    const wrapper = factory({ width: 1024, height: 1024 }, "wuerstchen");
    expect(wrapper.getComponent(ResolutionSelector).props("options")).toEqual([
      expect.objectContaining({ sub: "Native" }),
    ]);
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
    const stepper = wrapper
      .findAllComponents(Stepper)
      .find((candidate) => candidate.props("label") === "Batch size")!;
    expect(stepper.props("editable")).toBe(true);
    expect(stepper.props("max")).toBe(10_000);
    stepper.vm.$emit("update:modelValue", 300);
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.batchSize).toBe(300);
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

  it("offers a settings reset in the rail header", async () => {
    const wrapper = factory();
    const reset = wrapper.get("[data-test='settings-reset']");
    expect(reset.attributes("aria-label")).toBe(
      "Reset settings to model defaults",
    );
    await reset.trigger("click");
    expect(wrapper.emitted("reset-settings")).toHaveLength(1);
  });

  it("offers the same settings reset on phones", async () => {
    const wrapper = mount(ControlsAside, {
      props: { modelValue: baseForm(), family: "flux", mobile: true },
    });
    await wrapper.get("[data-test='settings-reset']").trigger("click");
    expect(wrapper.emitted("reset-settings")).toHaveLength(1);
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

  it("offers lock-last-seed while random once a run has completed", async () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ seedMode: "random", seed: null }),
        family: "flux",
        advCount: 0,
        lastSeed: 184023,
      },
    });
    const lock = wrapper.get("[data-test='lock-last-seed']");
    expect(lock.text()).toContain("lock last (184023)");

    await lock.trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.seedMode).toBe("static");
    expect(next.seed).toBe(184023);
  });

  it("hides the lock control before any run completes", () => {
    const wrapper = factory({ seedMode: "random", seed: null });
    expect(wrapper.find("[data-test='lock-last-seed']").exists()).toBe(false);
  });

  it("hides the lock control while the seed is already fixed", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ seedMode: "static", seed: 7 }),
        family: "flux",
        advCount: 0,
        lastSeed: 184023,
      },
    });
    expect(wrapper.find("[data-test='lock-last-seed']").exists()).toBe(false);
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

  // ── Output card (mockup 1c/3a: "mode is a setting, not a place") ─────
  it("renders the Output card ahead of Shape and emits the mode change", async () => {
    const wrapper = factory();
    const card = wrapper.get("[data-test='output-card']");
    // The Output card is the first group so it reads as part of the model
    // decision, above Shape.
    const firstGroup = wrapper.find(".controls__group");
    expect(firstGroup.element).toBe(card.element);
    const sequenceButton = card
      .findAll("button")
      .find((b) => b.text() === "Sequence")!;
    await sequenceButton.trigger("click");
    expect(wrapper.emitted("update:output")?.[0]).toEqual(["sequence"]);
  });

  it("captions Sequence mode with the parked-clips explanation", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(),
        family: "ltx2",
        advCount: 0,
        output: "sequence",
        clipCount: 3,
      },
    });
    expect(wrapper.get("[data-test='output-caption']").text()).toBe(
      "3 clips on the composer rail · one-shot and sequence prompts stay separate.",
    );
  });

  it("locks Batch to 1 in sequence mode with the one-timeline caption", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ batchSize: 4 }),
        family: "ltx2",
        advCount: 0,
        output: "sequence",
        clipCount: 2,
      },
    });
    const stepper = wrapper
      .findAllComponents(Stepper)
      .find((s) => s.props("label") === "Batch size")!;
    expect(stepper.props("modelValue")).toBe(1);
    expect(stepper.props("max")).toBe(1);
    expect(wrapper.get("[data-test='batch-locked']").text()).toContain(
      "a sequence renders one timeline",
    );
  });

  it("keeps Shape, Resolution, Detail, Prompt strength, and Seed live in sequence mode", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ width: 1024, height: 1024 }),
        family: "ltx2",
        advCount: 0,
        output: "sequence",
        clipCount: 2,
      },
    });
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(true);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(true);
    const labels = wrapper
      .findAllComponents(SliderRow)
      .map((row) => row.props("label"));
    expect(labels).toContain("Detail");
    expect(labels).toContain("Prompt strength");
    expect(wrapper.find("[data-test='seed-seg']").exists()).toBe(true);
  });
});


describe("ControlsAside — explicit shape pick vs source tie", () => {
  it("lets the user select the canonical aspect even when it equals the source", async () => {
    // A source whose size IS a canonical aspect describes the same canvas as
    // that aspect chip — the explicit pick, not derived matching, decides
    // which chip lights up.
    const sourceDimensions = { width: 1024, height: 1024 };
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(sourceDimensions),
        family: "flux",
        advCount: 0,
        sourceDimensions,
      },
    });
    const shape = wrapper.getComponent(ShapePicker);
    expect(shape.props("modelValue")).toBe("source");

    const canonical = (shape.props("options") as Array<{ id: string }>).find(
      (option) => option.id !== "source",
    )!.id;
    shape.vm.$emit("update:modelValue", canonical);
    await wrapper.vm.$nextTick();
    // The pick may or may not resize (nearest preset); when the canvas stays
    // identical the explicit pick must still win the highlight.
    const emitted = wrapper.emitted("update:modelValue");
    if (emitted) {
      await wrapper.setProps({
        modelValue: emitted.at(-1)![0] as GenerateFormState,
      });
    }
    expect(shape.props("modelValue")).toBe(canonical);
  });
});
