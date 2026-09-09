import { defineStore } from "pinia";
import { computed, ref, shallowRef } from "vue";

/**
 * The 3-D Studio's draft, held outside the view.
 *
 * `MeshWorkflowStudio` unmounts on every route change — the router lazy-loads
 * it and nothing wraps it in `<KeepAlive>` — so a component-local `ref` took
 * the prompt, both styles, the attached files, the stage settings and the
 * selected workflow with it. Leaving for the Queue and coming back landed on
 * an empty form. This is the same reason `desktop/src/stores/generateForm.ts`
 * exists for New image, and the same reason `parkedStillModel` had to move out
 * of the Create toolbar and into the ui store.
 *
 * Two lifetimes on purpose:
 *
 * - The SCALARS persist to localStorage, like `lastUsedStyles`, so a restart
 *   opens on the workflow you were writing.
 * - The attached `File`s live in memory only. A browser cannot re-open a file
 *   handle from a previous session, so persisting a name would promise a file
 *   the next launch cannot read; the wells simply come back empty and say so.
 *   Within a session they survive navigation, which is the reported bug.
 */
export type MeshWorkflowMode =
  "text_to_mesh" | "mesh_roundtrip" | "mesh_texture";

export type MeshUpAxis = "y" | "z";

export const MESH_WORKFLOW_DRAFT_KEY = "mold.create.meshWorkflowDraft.v1";

/** Minimal storage surface — `localStorage`, or an injected stub in tests. */
export interface MeshWorkflowDraftStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

let storageOverride: MeshWorkflowDraftStorage | null = null;

/** Test hook: inject a storage stub (pass null to restore the browser default). */
export function setMeshWorkflowDraftStorage(
  storage: MeshWorkflowDraftStorage | null,
): void {
  storageOverride = storage;
}

function storage(): MeshWorkflowDraftStorage | null {
  if (storageOverride) return storageOverride;
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

const MODES: readonly MeshWorkflowMode[] = [
  "text_to_mesh",
  "mesh_roundtrip",
  "mesh_texture",
];

const TEXTURE_RESOLUTIONS: readonly number[] = [1024, 2048, 4096];

interface PersistedV1 {
  version: 1;
  mode: MeshWorkflowMode;
  meshModelName: string;
  imageModelName: string;
  prompt: string;
  texture: boolean;
  textureResolution: number;
  delight: boolean;
  upAxis: MeshUpAxis;
  metersPerUnit: number;
  /** The machine the workflow is pinned to, or `auto` / `capable`. */
  routing: string | null;
  browseHostId: string;
}

function defaults(): PersistedV1 {
  return {
    version: 1,
    mode: "text_to_mesh",
    meshModelName: "",
    imageModelName: "",
    prompt: "",
    texture: true,
    textureResolution: 2048,
    delight: false,
    upAxis: "y",
    metersPerUnit: 1,
    routing: null,
    browseHostId: "",
  };
}

/**
 * A stored record is data this build did not write — an older or newer
 * version, a hand-edited value, a truncated write. Every field falls back on
 * its own so one unreadable value never discards the rest of the draft.
 */
function load(): PersistedV1 {
  const record = defaults();
  try {
    const raw = storage()?.getItem(MESH_WORKFLOW_DRAFT_KEY);
    if (!raw) return record;
    const parsed = JSON.parse(raw) as Partial<PersistedV1> | null;
    if (!parsed || typeof parsed !== "object" || parsed.version !== 1)
      return record;
    if (MODES.includes(parsed.mode as MeshWorkflowMode))
      record.mode = parsed.mode as MeshWorkflowMode;
    if (typeof parsed.meshModelName === "string")
      record.meshModelName = parsed.meshModelName;
    if (typeof parsed.imageModelName === "string")
      record.imageModelName = parsed.imageModelName;
    if (typeof parsed.prompt === "string") record.prompt = parsed.prompt;
    if (typeof parsed.texture === "boolean") record.texture = parsed.texture;
    if (TEXTURE_RESOLUTIONS.includes(parsed.textureResolution as number))
      record.textureResolution = parsed.textureResolution as number;
    if (typeof parsed.delight === "boolean") record.delight = parsed.delight;
    if (parsed.upAxis === "y" || parsed.upAxis === "z")
      record.upAxis = parsed.upAxis;
    if (
      typeof parsed.metersPerUnit === "number" &&
      Number.isFinite(parsed.metersPerUnit) &&
      parsed.metersPerUnit > 0
    )
      record.metersPerUnit = parsed.metersPerUnit;
    if (typeof parsed.routing === "string" || parsed.routing === null)
      record.routing = parsed.routing;
    if (typeof parsed.browseHostId === "string")
      record.browseHostId = parsed.browseHostId;
  } catch {
    // Unreadable storage or a corrupt record opens on the defaults.
  }
  return record;
}

export const useMeshWorkflowDraftStore = defineStore(
  "meshWorkflowDraft",
  () => {
    const initial = load();

    const mode = ref<MeshWorkflowMode>(initial.mode);
    const meshModelName = ref(initial.meshModelName);
    const imageModelName = ref(initial.imageModelName);
    const prompt = ref(initial.prompt);
    const texture = ref(initial.texture);
    const textureResolution = ref(initial.textureResolution);
    const delight = ref(initial.delight);
    const upAxis = ref<MeshUpAxis>(initial.upAxis);
    const metersPerUnit = ref(initial.metersPerUnit);
    const routing = ref<string | null>(initial.routing);
    const browseHostId = ref(initial.browseHostId);

    /*
     * `shallowRef`, not `ref`: a `File` is host-object state with no useful
     * reactive interior, and Vue's proxy over one breaks the identity the
     * upload path compares against.
     */
    const meshFile = shallowRef<File | null>(null);
    const appearanceFile = shallowRef<File | null>(null);

    /**
     * The workflow whose result is on the canvas. Session-scoped: a restart
     * opens on a new workflow rather than re-fetching a job that may have been
     * deleted, and the Recent workflows list is the way back to it.
     */
    const selectedId = ref("");

    function persist(): void {
      const record: PersistedV1 = {
        version: 1,
        mode: mode.value,
        meshModelName: meshModelName.value,
        imageModelName: imageModelName.value,
        prompt: prompt.value,
        texture: texture.value,
        textureResolution: textureResolution.value,
        delight: delight.value,
        upAxis: upAxis.value,
        metersPerUnit: metersPerUnit.value,
        routing: routing.value,
        browseHostId: browseHostId.value,
      };
      try {
        storage()?.setItem(MESH_WORKFLOW_DRAFT_KEY, JSON.stringify(record));
      } catch {
        // Storage refused (quota, private mode): the session still remembers.
      }
    }

    /** Whether anything the person typed or attached would be lost. */
    const dirty = computed(
      () =>
        prompt.value.trim().length > 0 ||
        meshFile.value !== null ||
        appearanceFile.value !== null,
    );

    /** Start over, keeping the machine and the styles this session is using. */
    function clear(): void {
      prompt.value = "";
      meshFile.value = null;
      appearanceFile.value = null;
      selectedId.value = "";
      persist();
    }

    return {
      mode,
      meshModelName,
      imageModelName,
      prompt,
      texture,
      textureResolution,
      delight,
      upAxis,
      metersPerUnit,
      routing,
      browseHostId,
      meshFile,
      appearanceFile,
      selectedId,
      dirty,
      clear,
      persist,
    };
  },
);
