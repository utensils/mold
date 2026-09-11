import { beforeEach, describe, expect, it } from "vitest";
import { computed, effectScope, nextTick, ref } from "vue";
import { createPinia, setActivePinia } from "pinia";
import { useLastUsedStylesStore } from "@studio/stores/lastUsedStyles";
import {
  OUTPUT_KIND_MISSING,
  useCreateOutputKind,
} from "./useCreateOutputKind";
import { useGenerateForm, __testing__ } from "./useGenerateForm";
import type { ModelInfoExtended } from "../types";

function model(name: string, family: string): ModelInfoExtended {
  return {
    name,
    family,
    size_gb: 6,
    is_loaded: false,
    last_used: null,
    hf_repo: "",
    downloaded: true,
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: name,
  };
}
const still = model("still", "flux");
const clip = model("clip", "wan");
const mesh = model("mesh", "hunyuan3d");
function setup(entries = [still, clip, mesh]) {
  const form = useGenerateForm();
  const models = ref(entries);
  const family = computed(
    () =>
      models.value.find((m) => m.name === form.state.value.model)?.family ?? "",
  );
  const scope = effectScope();
  const output = scope.run(() => useCreateOutputKind(form, models, family))!;
  return { form, models, output, scope };
}
beforeEach(() => {
  localStorage.clear();
  __testing__.resetForTest();
  setActivePinia(createPinia());
});
describe("web output doors", () => {
  it("switches one existing form and restores the shared remembered style", async () => {
    const preferred = model("preferred-clip", "wan");
    const memory = useLastUsedStylesStore();
    memory.remember("clip", preferred.name);
    const { form, output, scope } = setup([still, clip, preferred, mesh]);
    form.applyModelDefaults(still);
    form.state.value.prompt = "A quiet shore";
    await nextTick();
    output.selectKind("clip");
    await nextTick();
    expect(form.state.value.model).toBe(preferred.name);
    expect(form.state.value.prompt).toBe("A quiet shore");
    expect(output.kind.value).toBe("clip");
    expect(output.pickerModels.value.map((m) => m.name)).toEqual([
      clip.name,
      preferred.name,
    ]);
    expect(memory.lastSection).toBe("clip");
    scope.stop();
  });
  it("preserves a remembered style when startup must use another host's fallback", async () => {
    const memory = useLastUsedStylesStore();
    memory.remember("clip", "absent-on-this-host");
    const { form, output, scope } = setup();
    form.applyModelDefaults(output.initialStyle()!);
    await nextTick();
    expect(form.state.value.model).toBe(clip.name);
    expect(memory.bySection.clip).toBe("absent-on-this-host");
    output.selectStyle(still);
    await nextTick();
    expect(memory.lastSection).toBe("still");
    scope.stop();
  });
  it("keeps the current request intact when the requested output is unavailable", async () => {
    const { form, output, scope } = setup([still]);
    form.applyModelDefaults(still);
    form.state.value.prompt = "Keep this draft";
    await nextTick();
    const before = form.toRequest();
    output.selectKind("mesh");
    await nextTick();
    expect(form.toRequest()).toEqual(before);
    // The sentence is the shared table's, so "3-D" keeps its capitals here
    // the way it does everywhere else the kind is named.
    expect(output.notice.value).toBe(OUTPUT_KIND_MISSING.mesh);
    expect(output.notice.value).toContain("3-D object");
    expect(output.browseTo.value).toBe("/models?type=mesh");
    output.selectKind("still");
    expect(output.notice.value).toBe("");
    expect(output.browseTo.value).toBe("/models?type=image");
    scope.stop();
  });
});
