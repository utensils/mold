import { computed, type ComputedRef } from "vue";
import { filterRestrictedModels } from "@studio/lib/modelAccess";
import { modelsForOutput } from "@studio/lib/sequence";
import {
  isModelRuntimeUnavailable,
  modelRuntimeNotice,
  RUNTIME_UNAVAILABLE_BADGE,
} from "@studio/lib/modelRuntimeAvailability";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ModelEntry } from "../lib/api/types";
import type { GenerateForm } from "../lib/generateForm";
import { mergeInstalledModels } from "../lib/generateModels";
import { normalizeTargetHost } from "../lib/hosts";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";

/**
 * Everything the ONE style picker needs to decide what it may offer, and who
 * answers for the selected checkpoint's contract.
 *
 * It lives here rather than inside a component because two places ask these
 * questions: the composer's Style chip (which IS the picker) and the
 * inspector, whose Output switch, quality presets, canvas and mesh controls
 * all read the same rows. Duplicating the rules is how the two selectors
 * disagreed in the first place.
 */
export interface StylePicker {
  /** Runnable somewhere in the fleet, merged across hosts. */
  installedModels: ComputedRef<ModelEntry[]>;
  /** The pinned target host id, or null under Auto / Most capable. */
  stickyTarget: ComputedRef<string | null>;
  /** The form's model as the TARGET host has it, when it has it. */
  selectedModel: ComputedRef<ModelEntry | null>;
  /** The row the picker shows as selected — the target's, else any host's. */
  selectedPickerModel: ComputedRef<ModelEntry | null>;
  /** The row that answers for the CHECKPOINT'S CONTRACT (see below). */
  contractModel: ComputedRef<ModelEntry | null>;
  /** The form's model id when no machine has it installed. */
  missingModelId: ComputedRef<string | null>;
  /** Every row the picker may render, narrowed by target and output kind. */
  pickerModels: ComputedRef<ModelEntry[]>;
  /** Non-null marks a row unpickable and says why. */
  pickerDisabledReason: (model: ModelEntry) => string | null;
  /** The pinned machine's label when it lacks the selected model. */
  stickyHostMissingModel: ComputedRef<string | null>;
}

export function useStylePicker(form: () => GenerateForm): StylePicker {
  const draft = useSequenceDraftStore();
  const models = useModelStore();
  const hostModels = useHostModelsStore();
  const hosts = useHostsStore();
  const appPrefs = useAppPrefsStore();

  const installedModels = computed(() =>
    mergeInstalledModels(
      filterRestrictedModels(models.installed, hosts.capabilities.local),
      hostModels.unionInstalled,
    ),
  );

  /** Keep downloaded-but-unrunnable rows visible in Create. They remain outside
   * `installedModels`, so routing and submission cannot select them; the picker
   * only discloses what is already on disk and why generation is unavailable. */
  const downloadOnlyModels = computed(() =>
    mergeInstalledModels(
      models.installed.filter(isModelRuntimeUnavailable),
      hostModels.unionDownloaded.filter(isModelRuntimeUnavailable),
    ),
  );

  const stickyTarget = computed<string | null>(() =>
    normalizeTargetHost(appPrefs.settings?.generateTargetHost ?? null, hosts.all),
  );

  const selectedModel = computed<ModelEntry | null>(() =>
    hostModels.installedEntryForTarget(form().model, stickyTarget.value),
  );

  const pickerCandidates = computed<ModelEntry[]>(() => {
    const byName = new Map(installedModels.value.map((model) => [model.name, model]));
    for (const model of downloadOnlyModels.value) {
      if (!byName.has(model.name)) byName.set(model.name, model);
    }
    return [...byName.values()];
  });

  const selectedPickerModel = computed<ModelEntry | null>(
    () =>
      selectedModel.value ??
      pickerCandidates.value.find((model) => model.name === form().model) ??
      null,
  );

  /**
   * The row that answers for the CHECKPOINT'S CONTRACT — the advertised recipe
   * behind the canvas, the prompt mode, the source-image shape and the mesh
   * block.
   *
   * That contract belongs to the checkpoint, not to the machine holding the
   * file: `generation_profile::prompt_requirement_for_family` and the recipe's
   * own `mesh` / `resolution` blocks are the same wherever it runs, and the
   * target host will advertise exactly them once it has downloaded it. Reading
   * only the target's inventory therefore lost the whole contract the moment
   * Create was aimed at a machine that would have to pull the model: a
   * Hunyuan3D form silently fell back to raster controls with a Resolution
   * bound to the canvasless recipe's own 0 × 0, showing `NaN×NaN px` under an
   * uncorrectable "Width and height must be whole numbers".
   *
   * The target's own row still WINS wherever it exists — two machines may
   * advertise corrected or aliased metadata, and the one that will run the job
   * is the authority on it. This only decides who answers when it has none.
   * `selectedModel` remains the answer for questions that really are about the
   * holding machine (its runtime readiness, its on-disk size, its LoRAs).
   *
   * EVERY contract question reads this one row, including the ones the child
   * panels ask. Handing `SourceImageWell` and `AdvancedSettings` the target's
   * row instead is what left a Denoise slider, an Edit-mask control, a
   * Negative-prompt field and a `png`/`jpeg`/`webp` picker on a 3-D print after
   * the canvas itself had already been fixed.
   */
  const contractModel = computed<ModelEntry | null>(() =>
    hostModels.contractEntryForTarget(form().model, stickyTarget.value),
  );

  /**
   * The form's model when no machine has it installed. Restoring a print whose
   * checkpoint is gone must keep the id visible with a Not installed tag rather
   * than reading "Choose a model" — the raw id stays in `form.model` and in the
   * request either way.
   */
  const missingModelId = computed<string | null>(() =>
    form().model && !selectedPickerModel.value ? form().model : null,
  );

  const pickerModels = computed<ModelEntry[]>(() => {
    const target = stickyTarget.value;
    const fetched =
      target && target !== "capable" && (hostModels.byHost[target]?.fetchedAt ?? 0) > 0;
    const forTarget = fetched
      ? hostModels.downloadedOn(target).filter((model) => {
          const runnable = hostModels
            .installedOn(target)
            .some((candidate) => candidate.name === model.name);
          return runnable || isModelRuntimeUnavailable(model);
        })
      : pickerCandidates.value;
    // Sequence output narrows the picker to chain-capable video models.
    return modelsForOutput(forTarget, draft.output);
  });

  function pickerDisabledReason(model: ModelEntry): string | null {
    if (isModelRuntimeUnavailable(model)) {
      const reason =
        modelRuntimeNotice(model)?.message ?? "No selected machine can run this model.";
      return `${RUNTIME_UNAVAILABLE_BADGE} — ${reason}`;
    }
    if (installedModels.value.some((candidate) => candidate.name === model.name)) return null;
    const reason = modelRuntimeNotice(model)?.message ?? "No selected machine can run this model.";
    return `${RUNTIME_UNAVAILABLE_BADGE} — ${reason}`;
  }

  const stickyHostMissingModel = computed<string | null>(() => {
    const sel = stickyTarget.value;
    if (!sel || sel === "capable" || !form().model) return null;
    const host = hosts.all.find((h) => h.id === sel);
    if (!host) return null;
    const ids = hostModels.hostsFor(form().model);
    if (ids.length === 0 || ids.includes(sel)) return null;
    return host.label;
  });

  return {
    installedModels,
    stickyTarget,
    selectedModel,
    selectedPickerModel,
    contractModel,
    missingModelId,
    pickerModels,
    pickerDisabledReason,
    stickyHostMissingModel,
  };
}
