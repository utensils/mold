import { computed, inject, ref } from "vue";
import type { AdvancedPlacement, DevicePlacement, DeviceRef } from "../types";
import { RESOURCES_INJECTION_KEY, type UseResources } from "./useResources";

// Families that support the Advanced (Tier 2) per-component disclosure.
// Matches spec §3.2 — update both in lock-step.
const TIER2_FAMILIES: ReadonlyArray<string> = [
  "flux",
  "flux2",
  "flux.2",
  "flux2-klein",
  "z-image",
  "qwen-image",
  "qwen_image",
  "sd3",
  "sd3.5",
  "stable-diffusion-3",
  "stable-diffusion-3.5",
];

export interface UsePlacement {
  placement: import("vue").Ref<DevicePlacement | null>;
  gpuList: import("vue").ComputedRef<Array<{ ordinal: number; name: string }>>;
  supportsAdvanced: (family: string) => boolean;
  setTextEncoders: (ref: DeviceRef) => void;
  setAdvancedField: (
    field: keyof AdvancedPlacement,
    ref: DeviceRef | null,
  ) => void;
  loadSaved: (p: DevicePlacement | null) => void;
  clear: () => void;
  saveAsDefault: (model: string) => Promise<void>;
}

function defaultAdvanced(): AdvancedPlacement {
  return {
    transformer: { kind: "auto" },
    vae: { kind: "auto" },
    clip_l: null,
    clip_g: null,
    t5: null,
    qwen: null,
  };
}

export function usePlacement(): UsePlacement {
  const placement = ref<DevicePlacement | null>(null);

  // Consume Agent B's resource telemetry via the provided singleton
  // (`RESOURCES_INJECTION_KEY`, mounted once in App.vue). Fall back to an
  // empty list when running outside the provider (e.g. unit tests that
  // instantiate `usePlacement()` directly). The mapped `{ordinal, name}`
  // shape matches `PlacementPanel.vue`'s consumer contract.
  const resources = inject<UseResources | null>(RESOURCES_INJECTION_KEY, null);
  const gpuList = computed<Array<{ ordinal: number; name: string }>>(() =>
    (resources?.snapshot.value?.gpus ?? []).map((g) => ({
      ordinal: g.ordinal,
      name: g.name,
    })),
  );

  const supportsAdvanced = (family: string) => TIER2_FAMILIES.includes(family);

  const setTextEncoders = (r: DeviceRef) => {
    placement.value = {
      text_encoders: r,
      advanced: placement.value?.advanced ?? null,
    };
  };

  const setAdvancedField = (
    field: keyof AdvancedPlacement,
    r: DeviceRef | null,
  ) => {
    const current = placement.value ?? {
      text_encoders: { kind: "auto" } as DeviceRef,
      advanced: null,
    };
    const adv = { ...(current.advanced ?? defaultAdvanced()) };
    if (field === "transformer" || field === "vae") {
      adv[field] = r ?? { kind: "auto" };
    } else {
      adv[field] = r;
    }
    placement.value = { ...current, advanced: adv };
  };

  const loadSaved = (p: DevicePlacement | null) => {
    placement.value = p;
  };

  const clear = () => {
    placement.value = null;
  };

  const saveAsDefault = async (model: string) => {
    if (!placement.value) return;
    const encoded = encodeURIComponent(model);
    const resp = await fetch(`/api/config/model/${encoded}/placement`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(placement.value),
    });
    if (!resp.ok) {
      throw new Error(
        `failed to save placement default: ${resp.status} ${resp.statusText}`,
      );
    }
  };

  return {
    placement,
    gpuList,
    supportsAdvanced,
    setTextEncoders,
    setAdvancedField,
    loadSaved,
    clear,
    saveAsDefault,
  };
}
