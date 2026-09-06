import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import StylePicker from "./StylePicker.vue";
import stylePickerSource from "./StylePicker.vue?raw";
import ModelPicker from "./ModelPicker.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useModelStore } from "../../stores/models";
import { useConnectionStore } from "../../stores/connection";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { ModelEntry } from "../../lib/api/types";
import { apiJsonTo } from "../../lib/api/client";

/*
 * The composer's Style chip IS the picker. These cover the rows it may offer
 * (moved here from InspectorPanel.test.ts, which used to mount the second,
 * inspector-side copy of this control) plus the chip's own behaviour.
 */

const { routerPush } = vi.hoisted(() => ({ routerPush: vi.fn() }));
vi.mock("vue-router", () => ({ useRouter: () => ({ push: routerPush }) }));
vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../../lib/ipc", () => ({ ipc: {}, inTauri: () => false }));

beforeEach(() => {
  setActivePinia(createPinia());
  routerPush.mockClear();
});
afterEach(() => (document.body.innerHTML = ""));

function formFor(family: string): GenerateForm {
  return reactive({ ...newGenerateForm(), family });
}

function mountPicker(form: GenerateForm) {
  return mount(StylePicker, { props: { form }, attachTo: document.body });
}

const model: ModelEntry = {
  name: "flux-dev:q8",
  family: "flux",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 20,
  default_guidance: 4.5,
} as ModelEntry;

const videoModel = {
  ...model,
  name: "ltx-video",
  family: "ltx-video",
  supports_sequence: true,
  supports_video: true,
} as ModelEntry;

describe("StylePicker — the chip is the picker", () => {
  it("opens the menu from the chip and closes it on a second click", async () => {
    useModelStore().all = [model];
    const wrapper = mountPicker(useGenerateFormStore().form);

    expect(wrapper.find('[data-test="model-picker-menu"]').exists()).toBe(false);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(wrapper.find('[data-test="model-picker-menu"]').exists()).toBe(true);
    expect(wrapper.get('[data-test="style-chip"]').attributes("aria-expanded")).toBe("true");

    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(wrapper.find('[data-test="model-picker-menu"]').exists()).toBe(false);
  });

  it("opens UPWARD — the composer sits on the bottom edge of the canvas", async () => {
    useModelStore().all = [model];
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");

    const menu = wrapper.get('[data-test="model-picker-menu"]');
    expect(menu.attributes("data-placement")).toBe("up");
    expect(menu.classes()).toContain("ms-model__menu--up");
    expect(stylePickerSource).toContain('placement="up"');
  });

  it("picks a style into the shared form and closes", async () => {
    useModelStore().all = [model];
    const form = useGenerateFormStore().form;
    const wrapper = mountPicker(form);

    await wrapper.get('[data-test="style-chip"]').trigger("click");
    await wrapper.get('[data-test="model-option-name"]').trigger("click");

    expect(form.model).toBe("flux-dev:q8");
    expect(wrapper.find('[data-test="model-picker-menu"]').exists()).toBe(false);
  });

  it("closes on Escape", async () => {
    useModelStore().all = [model];
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    await wrapper.get('[data-test="style-chip"]').trigger("keydown", { key: "Escape" });
    expect(wrapper.find('[data-test="model-picker-menu"]').exists()).toBe(false);
  });

  it("walks the rows with ↑/↓ and picks with Enter", async () => {
    const second = { ...model, name: "flux-schnell:q8" } as ModelEntry;
    useModelStore().all = [model, second];
    const form = useGenerateFormStore().form;
    const wrapper = mountPicker(form);

    const chip = wrapper.get('[data-test="style-chip"]');
    await chip.trigger("click");
    await chip.trigger("keydown", { key: "ArrowDown" });
    await chip.trigger("keydown", { key: "Enter" });

    expect(form.model).toBe("flux-schnell:q8");
  });

  it("marks the style the form already carries", async () => {
    const second = { ...model, name: "flux-schnell:q8" } as ModelEntry;
    useModelStore().all = [model, second];
    const form = useGenerateFormStore().form;
    form.model = second.name;
    const wrapper = mountPicker(form);

    await wrapper.get('[data-test="style-chip"]').trigger("click");
    const current = wrapper.findAll('[data-test="model-option-current"]');
    expect(current).toHaveLength(1);
    expect(current[0]!.element.closest("button")!.textContent).toContain("flux-schnell:q8");
  });
});

describe("StylePicker — finding a style in a long list", () => {
  function manyModels(count: number): ModelEntry[] {
    return Array.from({ length: count }, (_, i) => ({
      ...model,
      name: i === 0 ? "z-image-turbo:q6" : `flux-dev-${i}:q8`,
      family: i === 0 ? "zimage" : "flux",
    })) as ModelEntry[];
  }

  it("offers no filter for a list short enough to read", async () => {
    useModelStore().all = manyModels(4);
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(wrapper.find('[data-test="model-filter"]').exists()).toBe(false);
  });

  it("offers a type-to-filter field once the list is long, and narrows on it", async () => {
    useModelStore().all = manyModels(12);
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");

    const filter = wrapper.get('[data-test="model-filter"]');
    expect(wrapper.findAll('[data-test="model-option-name"]')).toHaveLength(12);

    // By id …
    await filter.setValue("z-image");
    expect(wrapper.findAll('[data-test="model-option-id"]').map((row) => row.text())).toEqual([
      "z-image-turbo:q6",
    ]);

    // … and by the family's friendly label.
    await filter.setValue("Z-Image");
    expect(wrapper.findAll('[data-test="model-option-name"]')).toHaveLength(1);

    await filter.setValue("nothing matches this");
    expect(wrapper.find('[data-test="model-picker-empty"]').exists()).toBe(true);
  });
});

describe("StylePicker — what each row says", () => {
  // Moved from InspectorPanel.test.ts: the rows are the composable's, and the
  // inspector no longer renders a picker of its own.
  it("narrows to clip styles while a clip style is selected", async () => {
    useModelStore().all = [model, videoModel];
    const form = useGenerateFormStore().form;
    form.model = videoModel.name;
    form.family = videoModel.family;
    const wrapper = mountPicker(form);

    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(wrapper.findAll('[data-test="model-option-name"]').map((o) => o.text())).toEqual([
      "ltx-video",
    ]);
  });

  it("shows a human-readable catalog name while preserving the runnable id", async () => {
    const catalogModel = {
      ...model,
      name: "cv:23423432",
      family: "sdxl",
      description: "RealVisXL V5.0 by SG161222",
    } as ModelEntry;
    useModelStore().all = [catalogModel];
    const form = useGenerateFormStore().form;
    form.model = catalogModel.name;
    const wrapper = mountPicker(form);

    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe(
      "RealVisXL V5.0 by SG161222",
    );
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    const option = wrapper.get('[data-test="model-option-name"]');
    expect(option.text()).toBe("RealVisXL V5.0 by SG161222");
    expect(wrapper.get('[data-test="model-option-id"]').text()).toBe("cv:23423432");
    await option.trigger("click");
    expect(form.model).toBe("cv:23423432");
  });

  it("carries the model's own description as the row's second line", async () => {
    const described = {
      ...model,
      description: "full quality, 20+ steps",
      disk_usage_bytes: 23_100_000_000,
      is_loaded: true,
    } as ModelEntry;
    useModelStore().all = [described];
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");

    // The hint that used to sit under the inspector's picker now lives on the
    // entry, split into the plain sentence and the mono facts.
    expect(wrapper.get('[data-test="model-option-description"]').text()).toBe(
      "full quality, 20+ steps",
    );
    expect(wrapper.get('[data-test="model-option-size"]').text()).toBe("23.1 GB");
    expect(wrapper.get('[data-test="model-option-loaded"]').text()).toBe("on GPU");
  });

  it("shows a remote H3 download-only install with readable labels and its refusal", async () => {
    // H3 is a clip style, so the Short clip section is where its row lives.
    // The runtime refusal below is unchanged — it outranks every other reason.
    useGenerateFormStore().form.family = "minimax-h3";
    const h3 = {
      ...model,
      name: "minimax-h3-fl2va:comfy-pruned-nvfp4",
      family: "minimax-h3",
      runtime_available: false,
      runtime_unavailable_reason: "This H3 weight layout has no executable loader.",
    } as ModelEntry;
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "HAL 9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    useHostModelsStore().byHost["hal9000-7680"] = {
      entries: [h3],
      fetchedAt: Date.now(),
      error: null,
    };
    vi.mocked(apiJsonTo).mockResolvedValueOnce([h3]);
    const wrapper = mountPicker(useGenerateFormStore().form);

    await wrapper.get('[data-test="style-chip"]').trigger("click");

    expect(wrapper.get(".ms-model__group").text()).toBe("MiniMax H3");
    expect(wrapper.get('[data-test="model-option-name"]').text()).toBe("MiniMax H3 FL2VA · NVFP4");
    expect(wrapper.get('[data-test="model-disabled-reason"]').text()).toBe(
      "Download only — This H3 weight layout has no executable loader.",
    );
    expect(wrapper.get(".ms-model__option").attributes("disabled")).toBeDefined();
  });

  it("keeps an unavailable model disabled on the pinned host when another host can run it", () => {
    const name = "shared-runtime-model";
    const runnable = { ...model, name, runtime_available: true } as ModelEntry;
    const unavailable = {
      ...runnable,
      runtime_available: false,
      runtime_unavailable_reason: "HAL cannot execute this H3 weight layout.",
    } as ModelEntry;
    const connection = useConnectionStore();
    connection.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
    connection.status = "ready";
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "HAL 9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = { entries: [runnable], fetchedAt: Date.now(), error: null };
    hostModels.byHost["hal9000-7680"] = {
      entries: [unavailable],
      fetchedAt: Date.now(),
      error: null,
    };
    useModelStore().all = [runnable];
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    const form = useGenerateFormStore().form;
    form.model = name;
    const wrapper = mountPicker(form);

    const disabledReason = wrapper.getComponent(ModelPicker).props("disabledReason");
    expect(disabledReason).toBeTypeOf("function");
    if (!disabledReason) throw new Error("ModelPicker disabledReason prop is required");
    expect(disabledReason(unavailable)).toBe(
      "Download only — HAL cannot execute this H3 weight layout.",
    );
  });

  it("goes to Styles from the menu's footer", async () => {
    useModelStore().all = [model];
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    await wrapper.get('[data-test="browse-catalog"]').trigger("click");
    // Filtered to the section the menu was showing — the per-section targets
    // are pinned below in "sends Browse more to the Styles view filtered to
    // this section's kind".
    expect(routerPush).toHaveBeenCalledWith("/models?type=image");
  });
});

describe("StylePicker — a restored model no machine has", () => {
  it("keeps the recorded model visible with a Not installed tag", async () => {
    useModelStore().all = [];
    const form = formFor("zimage");
    form.model = "z-image-turbo:q6";
    const wrapper = mountPicker(form);
    await flushPromises();

    // Plain name first, the technical id in mono beside it (the mock's chip),
    // and the fact the user needs before pressing Generate.
    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe("Z-Image");
    expect(wrapper.get('[data-test="style-chip"]').text()).toContain("z-image-turbo:q6");
    expect(wrapper.get('[data-test="style-not-installed"]').text()).toBe("Not on this machine");
  });

  it("offers the pull for that exact id when its menu row is chosen", async () => {
    useModelStore().all = [];
    const form = formFor("zimage");
    form.model = "z-image-turbo:q6";
    const wrapper = mountPicker(form);
    await flushPromises();

    await wrapper.get('[data-test="style-chip"]').trigger("click");
    await wrapper.get('[data-test="model-option-missing"]').trigger("click");

    expect(wrapper.emitted("pull-missing-model")).toEqual([["z-image-turbo:q6"]]);
    // The raw id is what the form and the request keep carrying.
    expect(form.model).toBe("z-image-turbo:q6");
  });

  it("shows Choose a style only when nothing is selected at all", () => {
    const wrapper = mountPicker(formFor("flux"));
    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe("Choose a style");
    expect(wrapper.find('[data-test="style-not-installed"]').exists()).toBe(false);
  });

  it("says which pinned machine will have to download the style", async () => {
    const connection = useConnectionStore();
    connection.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
    connection.status = "ready";
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "HAL 9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = { entries: [model], fetchedAt: Date.now(), error: null };
    hostModels.byHost["hal9000-7680"] = { entries: [], fetchedAt: Date.now(), error: null };
    useModelStore().all = [model];
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    const form = useGenerateFormStore().form;
    form.model = model.name;
    const wrapper = mountPicker(form);
    await flushPromises();

    expect(wrapper.get('[data-test="style-will-download"]').text()).toBe(
      "Not on HAL 9000 — will download there",
    );
  });
});

/*
 * The view toolbar's Still picture | Short clip | 3-D object control names
 * three sections, and the menu must only offer what the section can make.
 * `useCreateOutputKind` is the one authority for that partition; these pin
 * that the picker reads it, says which section it is showing, and sends
 * Browse more to the matching kind filter on Styles.
 */
describe("StylePicker — the menu holds one section", () => {
  const meshModel = { ...model, name: "hunyuan3d-mini:fp16", family: "hunyuan3d" } as ModelEntry;
  const h3Model = {
    ...model,
    name: "minimax-h3-fl2va:comfy-pruned-nvfp4",
    family: "minimax-h3",
    supports_sequence: false,
  } as ModelEntry;
  const wanModel = {
    ...model,
    name: "wan22-ti2v-5b:q8",
    family: "wan",
    supports_sequence: true,
  } as ModelEntry;

  function openedIds(wrapper: ReturnType<typeof mountPicker>) {
    return wrapper.findAll('[data-test="model-option-id"]').map((o) => o.text());
  }

  /** H3 is license-gated on the primary (`filterRestrictedModels`), so its row
   *  reaches the picker through a connected machine's inventory, exactly as it
   *  does in the download-only case above. */
  function installOnRemote(entries: ModelEntry[]) {
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "HAL 9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    useHostModelsStore().byHost["hal9000-7680"] = {
      entries,
      fetchedAt: Date.now(),
      error: null,
    };
    // Opening the menu force-refreshes per-host availability, so the mocked
    // `/api/models` must answer with the same inventory or it clears it.
    vi.mocked(apiJsonTo).mockResolvedValue(entries);
  }

  it("offers only picture styles under Still picture", async () => {
    useModelStore().all = [model, videoModel, wanModel, meshModel];
    installOnRemote([h3Model]);
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(openedIds(wrapper)).toEqual(["flux-dev:q8"]);
  });

  it("offers every clip style under Short clip, H3 included", async () => {
    useModelStore().all = [model, videoModel, wanModel, meshModel];
    installOnRemote([h3Model]);
    const form = useGenerateFormStore().form;
    form.model = videoModel.name;
    form.family = videoModel.family;
    const wrapper = mountPicker(form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(openedIds(wrapper).sort()).toEqual([
      "ltx-video",
      "minimax-h3-fl2va:comfy-pruned-nvfp4",
      "wan22-ti2v-5b:q8",
    ]);
  });

  /*
   * `supports_sequence` is not a client gate any more: a clip has ONE way of
   * being made, so H3 is pickable rather than rendered disabled with a
   * refusal about joining scenes.
   */
  it("never refuses a runnable clip style for supports_sequence", async () => {
    useModelStore().all = [videoModel];
    installOnRemote([h3Model]);
    const form = useGenerateFormStore().form;
    form.model = videoModel.name;
    form.family = videoModel.family;
    const wrapper = mountPicker(form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");

    const disabledReason = wrapper.getComponent(ModelPicker).props("disabledReason");
    if (!disabledReason) throw new Error("ModelPicker disabledReason prop is required");
    expect(disabledReason(videoModel)).toBeNull();
    expect(disabledReason(h3Model)).toBeNull();
    expect(wrapper.find('[data-test="model-disabled-reason"]').exists()).toBe(false);
  });

  it("offers only 3-D styles under 3-D object", async () => {
    useModelStore().all = [model, videoModel, meshModel];
    const form = useGenerateFormStore().form;
    form.model = meshModel.name;
    form.family = meshModel.family;
    const wrapper = mountPicker(form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(openedIds(wrapper)).toEqual(["hunyuan3d-mini:fp16"]);
  });

  it("keeps the selected style visible even when it is not this section's kind", async () => {
    // Reusing the settings of a one-shot clip print leaves a clip style on a
    // Still-picture form. Hiding it would read as "the style was dropped".
    useModelStore().all = [model, videoModel];
    const form = useGenerateFormStore().form;
    form.model = videoModel.name;
    form.family = videoModel.family;
    const wrapper = mountPicker(form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(openedIds(wrapper)).toContain("ltx-video");
    expect(wrapper.findAll('[data-test="model-option-current"]')).toHaveLength(1);
  });

  it("names the section above the rows", async () => {
    useModelStore().all = [model];
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    expect(wrapper.get('[data-test="model-picker-kicker"]').text()).toBe("still picture styles");

    useGenerateFormStore().form.family = videoModel.family;
    await flushPromises();
    expect(wrapper.get('[data-test="model-picker-kicker"]').text()).toBe("short clip styles");
  });

  it("names the empty section instead of the generic no-match line", async () => {
    useModelStore().all = [model];
    const form = useGenerateFormStore().form;
    form.model = "";
    form.family = videoModel.family;
    const wrapper = mountPicker(form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");

    expect(wrapper.get('[data-test="model-picker-empty"]').text()).toBe(
      "No short clip styles on this machine.",
    );
    expect(wrapper.get('[data-test="browse-catalog"]').text()).toBe("Browse more →");
  });

  it("keeps the filter's own no-match line when the section does hold styles", async () => {
    useModelStore().all = Array.from({ length: 12 }, (_, i) => ({
      ...model,
      name: `flux-dev-${i}:q8`,
    })) as ModelEntry[];
    const wrapper = mountPicker(useGenerateFormStore().form);
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    await wrapper.get('[data-test="model-filter"]').setValue("nothing matches this");
    expect(wrapper.get('[data-test="model-picker-empty"]').text()).toBe(
      "No style matches “nothing matches this”.",
    );
  });

  it("sends Browse more to the Styles view filtered to this section's kind", async () => {
    useModelStore().all = [model, videoModel, meshModel];
    const form = useGenerateFormStore().form;
    const wrapper = mountPicker(form);

    await wrapper.get('[data-test="style-chip"]').trigger("click");
    await wrapper.get('[data-test="browse-catalog"]').trigger("click");
    expect(routerPush).toHaveBeenLastCalledWith("/models?type=image");

    form.family = videoModel.family;
    await flushPromises();
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    await wrapper.get('[data-test="browse-catalog"]').trigger("click");
    expect(routerPush).toHaveBeenLastCalledWith("/models?type=video");

    form.model = meshModel.name;
    form.family = meshModel.family;
    await flushPromises();
    await wrapper.get('[data-test="style-chip"]').trigger("click");
    await wrapper.get('[data-test="browse-catalog"]').trigger("click");
    expect(routerPush).toHaveBeenLastCalledWith("/models?type=mesh");
  });
});
