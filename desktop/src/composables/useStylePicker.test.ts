import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import { useStylePicker } from "./useStylePicker";
import { newGenerateForm, type GenerateForm } from "../lib/generateForm";
import { useModelStore } from "../stores/models";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useAppPrefsStore } from "../stores/appPrefs";
import type { ModelEntry } from "../lib/api/types";

/*
 * The single authority for which rows the ONE style picker may offer, which
 * row answers for the checkpoint's contract, and why a row refuses. Every
 * assertion on it used to arrive through a mounted `StylePicker.vue`, so a
 * rule could only be read through a rendered menu.
 */

vi.mock("../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {}, inTauri: () => false }));

beforeEach(() => setActivePinia(createPinia()));

function entry(over: Partial<ModelEntry> = {}): ModelEntry {
  return {
    name: "flux-dev:q8",
    family: "flux",
    downloaded: true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 4.5,
    ...over,
  } as ModelEntry;
}

const flux = entry();
const wan = entry({ name: "wan22-ti2v-5b:fp16", family: "wan" });
const mesh = entry({ name: "hunyuan3d-2.1:fp16", family: "hunyuan3d" });

/** `hosts.all` is derived, so a fixture registers machines the way the app
 *  does: an extra per machine plus its friendly name. */
function registerHosts(rows: { id: string; label: string }[]) {
  const hosts = useHostsStore();
  hosts.extras = rows.map((row) => ({ id: row.id, url: `http://${row.id}:7680` })) as never;
  hosts.names = Object.fromEntries(rows.map((row) => [row.id, row.label]));
  return hosts;
}

/** One host's `/api/models` answer, as the store stores it. */
function hostList(entries: ModelEntry[]) {
  return { entries, fetchedAt: Date.now(), error: null };
}

function pickerFor(form: GenerateForm) {
  return useStylePicker(() => form);
}

function formFor(over: Partial<GenerateForm> = {}): GenerateForm {
  return reactive({ ...newGenerateForm(), ...over });
}

describe("useStylePicker — which rows are offered", () => {
  it("offers only the section the view is in", () => {
    useModelStore().all = [flux, wan, mesh];
    const picker = pickerFor(formFor({ family: "flux", model: flux.name }));

    expect(picker.outputKind.value).toBe("still");
    expect(picker.pickerModels.value.map((m) => m.name)).toEqual([flux.name]);
  });

  it("follows the form's family into the clip and 3-D sections", () => {
    useModelStore().all = [flux, wan, mesh];
    expect(
      pickerFor(formFor({ family: "wan", model: wan.name })).pickerModels.value.map((m) => m.name),
    ).toEqual([wan.name]);
    expect(
      pickerFor(formFor({ family: "hunyuan3d", model: mesh.name })).pickerModels.value.map(
        (m) => m.name,
      ),
    ).toEqual([mesh.name]);
  });

  /* Reuse settings can leave a clip style on a Still-picture form. Hiding it
   * there would read as the style having been silently dropped. */
  it("keeps the SELECTED row whatever section it belongs to", () => {
    useModelStore().all = [flux, wan];
    const picker = pickerFor(formFor({ family: "flux", model: wan.name }));

    expect(picker.pickerModels.value.map((m) => m.name)).toEqual([wan.name, flux.name]);
    expect(picker.selectedPickerModel.value?.name).toBe(wan.name);
  });

  it("names a style no machine has instead of reading as no style at all", () => {
    useModelStore().all = [flux];
    const picker = pickerFor(formFor({ family: "flux", model: "ltx-2-19b-dev:fp8" }));

    expect(picker.selectedPickerModel.value).toBeNull();
    expect(picker.missingModelId.value).toBe("ltx-2-19b-dev:fp8");
  });

  it("has no missing style when the form names none", () => {
    useModelStore().all = [flux];
    expect(pickerFor(formFor({ family: "flux", model: "" })).missingModelId.value).toBeNull();
  });
});

describe("useStylePicker — why a row refuses", () => {
  it("keeps a downloaded-but-unrunnable row visible and refuses it in the server's words", () => {
    const unrunnable = entry({
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
      runtime_available: false,
      runtime_unavailable_reason: "This build has no MiniMax H3 engine.",
    });
    useModelStore().all = [unrunnable];

    const picker = pickerFor(formFor({ family: "minimax-h3" }));
    expect(picker.pickerModels.value.map((m) => m.name)).toEqual([unrunnable.name]);
    expect(picker.pickerDisabledReason(unrunnable)).toBe(
      "Download only — This build has no MiniMax H3 engine.",
    );
    // It is deliberately NOT runnable, so nothing may route to it.
    expect(picker.installedModels.value.map((m) => m.name)).toEqual([]);
  });

  /* `supports_sequence` is NOT a gate: a clip has one way of being made, so
   * every clip style is pickable in the clip section. */
  it("refuses nothing for a runnable row, clip styles included", () => {
    const h3 = entry({ name: "minimax-h3-fl2va:official-bf16", family: "minimax-h3" });
    useModelStore().all = [wan, h3];
    const picker = pickerFor(formFor({ family: "wan", model: wan.name }));

    expect(picker.pickerDisabledReason(wan)).toBeNull();
    expect(picker.pickerDisabledReason(h3)).toBeNull();
  });

  it("refuses a row no selected machine can run", () => {
    useModelStore().all = [];
    const hostModels = useHostModelsStore();
    hostModels.byHost = { a: hostList([{ ...wan, runtime_available: false }]) };

    const picker = pickerFor(formFor({ family: "wan" }));
    expect(picker.pickerDisabledReason(wan)).toContain("Download only");
  });
});

describe("useStylePicker — the pinned machine", () => {
  it("is null under Auto and Most capable", () => {
    useModelStore().all = [flux];
    expect(pickerFor(formFor({ family: "flux" })).stickyTarget.value).toBeNull();

    useAppPrefsStore().settings = { generateTargetHost: "capable" } as never;
    expect(pickerFor(formFor({ family: "flux" })).stickyTarget.value).toBe("capable");
  });

  it("says nothing about a pinned machine that already has the style", () => {
    useModelStore().all = [flux];
    registerHosts([{ id: "a", label: "plato" }]);
    useAppPrefsStore().settings = { generateTargetHost: "a" } as never;
    useHostModelsStore().byHost = { a: hostList([flux]) };

    expect(
      pickerFor(formFor({ family: "flux", model: flux.name })).stickyHostMissingModel.value,
    ).toBeNull();
  });

  it("names the pinned machine that will have to download the style", () => {
    useModelStore().all = [flux];
    registerHosts([
      { id: "a", label: "plato" },
      { id: "b", label: "hal9000" },
    ]);
    useAppPrefsStore().settings = { generateTargetHost: "a" } as never;
    useHostModelsStore().byHost = { a: hostList([]), b: hostList([flux]) };

    expect(
      pickerFor(formFor({ family: "flux", model: flux.name })).stickyHostMissingModel.value,
    ).toBe("plato");
  });
});
