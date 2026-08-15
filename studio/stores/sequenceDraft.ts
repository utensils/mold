/**
 * The durable sequence draft — output mode, the clip list, audio, and the
 * edit session — shared by desktop, web, and iPhone Create.
 *
 * This store exists because every surface lost sequence state its own way:
 * desktop kept the chain form component-local (every navigation wiped all
 * clip prompts), web persisted a private script copy that drifted from the
 * inspector, and the iPhone composer's form died behind a `v-if` on tab
 * switch. Clips live HERE; shared params (model/shape/detail/seed/fps) stay
 * in each surface's generate form and are read at submit time.
 *
 * Persistence is localStorage on all three surfaces (the iPhone Tauri
 * webview already trusts localStorage for durable state). Base64 source
 * payloads are stripped before writing — a 2K PNG base64s to megabytes and
 * blows quota — the filename is kept for display. `editing` is never
 * persisted on purpose: an edit session reloads from the durable job.
 */

import { defineStore } from "pinia";
import { reactive, ref, watch } from "vue";
import {
  newSequenceClip,
  type SequenceClipForm,
  type SequenceClipSourceImage,
} from "../lib/sequenceForm";
import { chainScriptToClips } from "../lib/sequenceForm";
import {
  deleteDraftMedia,
  getDraftMedia,
  putDraftMedia,
} from "../lib/draftMediaStore";
import type { OutputMode, SequenceTransition } from "../lib/sequence";
import type { ChainScript } from "../lib/api/chainTypes";

export const SEQUENCE_DRAFT_KEY = "mold.sequence.draft.v1";
export const LEGACY_WEB_DRAFT_KEY = "mold.chain.draft.v2";
export const LEGACY_WEB_MODE_KEY = "mold.composer.mode";
export const LEGACY_MOBILE_MODE_KEY = "mold.mobile.create-mode.v1";

const PERSIST_DEBOUNCE_MS = 300;

/** Minimal storage surface — `localStorage` in browsers/webviews, an
 * injected stub in tests (this vitest environment exposes no global). */
export interface DraftStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

let storageOverride: DraftStorage | null = null;

/** Test hook: inject a storage stub (pass null to restore the browser default). */
export function setSequenceDraftStorage(storage: DraftStorage | null) {
  storageOverride = storage;
}

function draftStorage(): DraftStorage | null {
  if (storageOverride) return storageOverride;
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export interface SequenceEditSession {
  jobId: string;
  hostId: string;
  /** The loaded job's clips at load time — `stageInvalidation`'s baseline. */
  baseline: SequenceClipForm[];
  /** The durable job's opening frame, restored when edits are discarded. */
  baselineOpeningImage?: SequenceClipSourceImage | null;
  /** The durable job's audio choice, restored when edits are discarded. */
  baselineEnableAudio?: boolean;
  /** Leading stages the job has actually completed (cache ceiling). */
  completedStages: number;
}

interface PersistedClip extends Omit<SequenceClipForm, "sourceImage"> {
  sourceImage:
    (Omit<SequenceClipSourceImage, "base64"> & { base64: null }) | null;
}

interface PersistedDraftV1 {
  version: 1;
  output: OutputMode;
  clips: PersistedClip[];
  openingImage?:
    (Omit<SequenceClipSourceImage, "base64"> & { base64: null }) | null;
  enableAudio: boolean;
  lastSingleModel: string | null;
  sequenceModel?: string | null;
}

function persistableClips(clips: readonly SequenceClipForm[]): PersistedClip[] {
  return clips.map((clip) => ({
    ...clip,
    sourceImage: clip.sourceImage
      ? {
          filename: clip.sourceImage.filename,
          ...(clip.sourceImage.draftId
            ? { draftId: clip.sourceImage.draftId }
            : {}),
          ...(clip.sourceImage.width ? { width: clip.sourceImage.width } : {}),
          ...(clip.sourceImage.height
            ? { height: clip.sourceImage.height }
            : {}),
          base64: null,
        }
      : null,
  }));
}

function normalizePersistedClip(clip: PersistedClip): SequenceClipForm {
  return {
    ...clip,
    cameraControl: clip.cameraControl ?? null,
  };
}

const OPENING_MEDIA_ID = "sequence-opening-image";

function readJson<T>(key: string): T | null {
  try {
    const raw = draftStorage()?.getItem(key) ?? null;
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export const useSequenceDraftStore = defineStore("sequence-draft", () => {
  const output = ref<OutputMode>("single");
  const clips = reactive<SequenceClipForm[]>([]);
  const openingImage = ref<SequenceClipSourceImage | null>(null);
  const mediaRestoring = ref(false);
  const activeClipId = ref<string | null>(null);
  const enableAudio = ref(false);
  const editing = ref<SequenceEditSession | null>(null);
  const lastSingleModel = ref<string | null>(null);
  const sequenceModel = ref<string | null>(null);
  const hydrated = ref(false);

  let persistTimer: ReturnType<typeof setTimeout> | null = null;

  function persistNow() {
    if (openingImage.value?.base64) {
      const draftId = openingImage.value.draftId ?? OPENING_MEDIA_ID;
      openingImage.value.draftId = draftId;
      void putDraftMedia({
        ...openingImage.value,
        draftId,
        base64: openingImage.value.base64,
      });
    }
    for (const clip of clips) {
      if (!clip.sourceImage?.base64) continue;
      const draftId = clip.sourceImage.draftId ?? `sequence-clip-${clip.id}`;
      clip.sourceImage.draftId = draftId;
      void putDraftMedia({
        ...clip.sourceImage,
        draftId,
        base64: clip.sourceImage.base64,
      });
    }
    const draft: PersistedDraftV1 = {
      version: 1,
      output: output.value,
      clips: persistableClips(clips),
      openingImage: openingImage.value
        ? {
            filename: openingImage.value.filename,
            draftId: openingImage.value.draftId ?? OPENING_MEDIA_ID,
            ...(openingImage.value.width
              ? { width: openingImage.value.width }
              : {}),
            ...(openingImage.value.height
              ? { height: openingImage.value.height }
              : {}),
            base64: null,
          }
        : null,
      enableAudio: enableAudio.value,
      lastSingleModel: lastSingleModel.value,
      sequenceModel: sequenceModel.value,
    };
    try {
      draftStorage()?.setItem(SEQUENCE_DRAFT_KEY, JSON.stringify(draft));
    } catch {
      // Quota or private-mode failures must never break composing.
    }
  }

  function schedulePersist() {
    if (!hydrated.value) return;
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = setTimeout(persistNow, PERSIST_DEBOUNCE_MS);
  }

  function applyPersisted(saved: PersistedDraftV1) {
    output.value = saved.output;
    clips.splice(0, clips.length, ...saved.clips.map(normalizePersistedClip));
    openingImage.value = saved.openingImage ?? null;
    enableAudio.value = saved.enableAudio;
    lastSingleModel.value = saved.lastSingleModel ?? null;
    sequenceModel.value = saved.sequenceModel ?? null;
  }

  async function restorePersistedMedia() {
    const pending: Array<Promise<void>> = [];
    if (openingImage.value && !openingImage.value.base64) {
      const marker = openingImage.value;
      pending.push(
        getDraftMedia<SequenceClipSourceImage & { draftId: string }>(
          marker.draftId ?? OPENING_MEDIA_ID,
        ).then((stored) => {
          if (stored?.base64 && openingImage.value === marker) {
            openingImage.value = {
              filename: stored.filename,
              draftId: stored.draftId,
              ...((stored.width ?? marker.width)
                ? { width: stored.width ?? marker.width }
                : {}),
              ...((stored.height ?? marker.height)
                ? { height: stored.height ?? marker.height }
                : {}),
              base64: stored.base64,
            };
          }
        }),
      );
    }
    for (const clip of clips) {
      const marker = clip.sourceImage;
      if (!marker || marker.base64) continue;
      pending.push(
        getDraftMedia<SequenceClipSourceImage & { draftId: string }>(
          marker.draftId ?? `sequence-clip-${clip.id}`,
        ).then((stored) => {
          if (stored?.base64 && clip.sourceImage === marker) {
            clip.sourceImage = {
              filename: stored.filename,
              draftId: stored.draftId,
              ...((stored.width ?? marker.width)
                ? { width: stored.width ?? marker.width }
                : {}),
              ...((stored.height ?? marker.height)
                ? { height: stored.height ?? marker.height }
                : {}),
              base64: stored.base64,
            };
          }
        }),
      );
    }
    if (pending.length === 0) return;
    mediaRestoring.value = true;
    await Promise.all(pending);
    mediaRestoring.value = false;
  }

  /** One-shot migrations from the pre-unification keys. */
  function migrateLegacy() {
    // web's ScriptComposer persisted its TOML-shaped draft verbatim, which
    // keys stages as `stage` (mold.chain.v1 / Rust serde rename); accept
    // both spellings so real drafts migrate, not just idealized ones.
    const legacyDraft = readJson<
      ChainScript & { stage?: ChainScript["stages"] }
    >(LEGACY_WEB_DRAFT_KEY);
    const legacyStages = legacyDraft?.stages ?? legacyDraft?.stage;
    if (legacyDraft && legacyStages?.length) {
      const loaded = chainScriptToClips({
        ...legacyDraft,
        stages: legacyStages,
      });
      clips.splice(0, clips.length, ...loaded.clips);
      openingImage.value = loaded.openingImage;
      enableAudio.value = loaded.enableAudio;
      // Deliberately NOT importing the legacy chain-level width/steps/
      // guidance — those private copies were the stale-inspector bug.
    }
    const storage = draftStorage();
    const webMode = storage?.getItem(LEGACY_WEB_MODE_KEY) ?? null;
    if (webMode === "script") output.value = "sequence";
    const mobileMode = storage?.getItem(LEGACY_MOBILE_MODE_KEY) ?? null;
    if (mobileMode === "sequence") output.value = "sequence";
    storage?.removeItem(LEGACY_WEB_DRAFT_KEY);
    storage?.removeItem(LEGACY_WEB_MODE_KEY);
    storage?.removeItem(LEGACY_MOBILE_MODE_KEY);
  }

  function hydrate() {
    if (hydrated.value) return;
    const saved = readJson<PersistedDraftV1>(SEQUENCE_DRAFT_KEY);
    let migrated = false;
    if (saved?.version === 1) {
      applyPersisted(saved);
    } else {
      migrateLegacy();
      migrated = clips.length > 0 || output.value === "sequence";
    }
    if (clips.length > 0 && !activeClipId.value) {
      activeClipId.value = clips[clips.length - 1]?.id ?? null;
    }
    hydrated.value = true;
    void restorePersistedMedia();
    // migrateLegacy() consumed the legacy keys while `hydrated` was still
    // false (the watcher won't schedule persistence), so a reload before
    // any edit would lose the migrated draft — make it durable right away.
    if (migrated) persistNow();
  }

  /** A sequence always has at least two clips (repo invariant). */
  function ensureClips(defaultFrames: number) {
    while (clips.length < 2) clips.push(newSequenceClip(defaultFrames));
    if (!activeClipId.value) activeClipId.value = clips[0]?.id ?? null;
  }

  function addClip(defaultFrames: number) {
    const clip = newSequenceClip(defaultFrames);
    clips.push(clip);
    activeClipId.value = clip.id;
    return clip;
  }

  /** Reset the model-owned duration field without disturbing prompts, media,
   * transitions, seeds, or the active clip. Model selectors call this only
   * after the new model's authoritative chain limits have arrived. */
  function resetClipFrames(defaultFrames: number) {
    for (const clip of clips) clip.frames = defaultFrames;
  }

  /** Adopt one concrete model as the clip-duration authority. */
  function adoptSequenceModel(model: string, defaultFrames: number): boolean {
    if (sequenceModel.value === model) return false;
    sequenceModel.value = model;
    resetClipFrames(defaultFrames);
    return true;
  }

  /** Bind deliberately imported timings to their concrete model without
   * treating that import as a model switch. Reuse/edit loaders own the clip
   * lengths they just restored; later model changes still flow through
   * `adoptSequenceModel` and reset those model-owned values. */
  function bindSequenceModel(model: string) {
    sequenceModel.value = model;
  }

  function removeClip(id: string) {
    if (clips.length <= 2) return;
    const idx = clips.findIndex((clip) => clip.id === id);
    if (idx < 0) return;
    const [removed] = clips.splice(idx, 1);
    if (removed?.sourceImage?.draftId)
      void deleteDraftMedia(removed.sourceImage.draftId);
    if (activeClipId.value === id) {
      activeClipId.value = clips[Math.min(idx, clips.length - 1)]?.id ?? null;
    }
  }

  function moveClip(id: string, toIndex: number) {
    const from = clips.findIndex((clip) => clip.id === id);
    if (from < 0) return;
    const clamped = Math.max(0, Math.min(toIndex, clips.length - 1));
    const [clip] = clips.splice(from, 1);
    if (clip) clips.splice(clamped, 0, clip);
  }

  function setTransition(
    id: string,
    transition: SequenceTransition,
    fadeFrames?: number,
  ) {
    const clip = clips.find((c) => c.id === id);
    if (!clip) return;
    clip.transition = transition;
    if (fadeFrames != null) clip.fadeFrames = fadeFrames;
  }

  /** ⌥-click in the seam editor: apply one transition to every seam. */
  function applyTransitionToAllSeams(
    transition: SequenceTransition,
    fadeFrames?: number,
  ) {
    clips.forEach((clip, idx) => {
      if (idx === 0) return;
      clip.transition = transition;
      if (fadeFrames != null) clip.fadeFrames = fadeFrames;
    });
  }

  /**
   * Output is a setting, not a place. One-shot and sequence prompts are
   * independent authorities: switching modes parks both without copying
   * either prompt into the other mode.
   */
  function setOutput(
    mode: OutputMode,
    // Retained while every surface shares the existing output-switch callback
    // shape. Deliberately unused: prompts have separate authorities and must
    // never be bridged during an output switch.
    _bridge: { getPrompt: () => string; setPrompt: (value: string) => void },
    defaultFrames: number,
  ) {
    if (mode === output.value) return;
    if (mode === "sequence") {
      ensureClips(defaultFrames);
      activeClipId.value = clips[0]?.id ?? null;
    }
    output.value = mode;
  }

  /** Load a durable job's effective script into an edit session. */
  function loadFromJob(
    session: SequenceEditSession,
    loadedClips: SequenceClipForm[],
    loadedEnableAudio: boolean,
    loadedOpeningImage: SequenceClipSourceImage | null = null,
  ) {
    clips.splice(0, clips.length, ...loadedClips);
    enableAudio.value = loadedEnableAudio;
    openingImage.value = loadedOpeningImage;
    editing.value = {
      ...session,
      baselineOpeningImage: loadedOpeningImage
        ? { ...loadedOpeningImage }
        : null,
      baselineEnableAudio: loadedEnableAudio,
    };
    output.value = "sequence";
    activeClipId.value = clips[0]?.id ?? null;
  }

  function stopEditing() {
    editing.value = null;
  }

  /**
   * Clear the whole sequence: back to two fresh clips, no audio, and no
   * edit session. Stays in Sequence — clearing means "start this story
   * over", not "leave sequence mode" (that's the Output control's job).
   */
  function clearSequence(defaultFrames: number) {
    if (openingImage.value?.draftId)
      void deleteDraftMedia(openingImage.value.draftId);
    for (const clip of clips) {
      if (clip.sourceImage?.draftId)
        void deleteDraftMedia(clip.sourceImage.draftId);
    }
    clips.splice(
      0,
      clips.length,
      newSequenceClip(defaultFrames),
      newSequenceClip(defaultFrames),
    );
    activeClipId.value = clips[0]?.id ?? null;
    enableAudio.value = false;
    openingImage.value = null;
    editing.value = null;
  }

  function reset() {
    if (openingImage.value?.draftId)
      void deleteDraftMedia(openingImage.value.draftId);
    for (const clip of clips) {
      if (clip.sourceImage?.draftId)
        void deleteDraftMedia(clip.sourceImage.draftId);
    }
    clips.splice(0, clips.length);
    activeClipId.value = null;
    enableAudio.value = false;
    openingImage.value = null;
    editing.value = null;
    output.value = "single";
  }

  // Sync flush so the debounce timer arms on the mutation itself (the
  // callback only re-arms a setTimeout — cheap enough for every keystroke).
  watch(
    [output, clips, openingImage, enableAudio, lastSingleModel, sequenceModel],
    () => schedulePersist(),
    { deep: true, flush: "sync" },
  );

  return {
    output,
    clips,
    openingImage,
    mediaRestoring,
    activeClipId,
    enableAudio,
    editing,
    lastSingleModel,
    sequenceModel,
    hydrated,
    hydrate,
    ensureClips,
    addClip,
    resetClipFrames,
    adoptSequenceModel,
    bindSequenceModel,
    removeClip,
    moveClip,
    setTransition,
    applyTransitionToAllSeams,
    setOutput,
    loadFromJob,
    stopEditing,
    clearSequence,
    reset,
  };
});
