import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it } from "vitest";

import {
  MESH_WORKFLOW_DRAFT_KEY,
  setMeshWorkflowDraftStorage,
  useMeshWorkflowDraftStore,
  type MeshWorkflowDraftStorage,
} from "./meshWorkflowDraft";

function memoryStorage(seed: Record<string, string> = {}) {
  const map = new Map(Object.entries(seed));
  const storage: MeshWorkflowDraftStorage = {
    getItem: (key) => map.get(key) ?? null,
    setItem: (key, value) => void map.set(key, value),
    removeItem: (key) => void map.delete(key),
  };
  return { storage, map };
}

describe("the 3-D Studio draft", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    setMeshWorkflowDraftStorage(null);
  });

  /*
   * The reported bug: the view unmounts on every route change, so a
   * component-local ref lost the whole form on a trip to the Queue. The store
   * IS the fix, so the thing to pin is that a second instantiation — what a
   * remount does — sees what the first one wrote.
   */
  it("keeps the draft across a remount of the view", () => {
    const first = useMeshWorkflowDraftStore();
    first.prompt = "a hand-carved wooden fox";
    first.mode = "mesh_texture";
    first.meshModelName = "hunyuan3d-2.1:fp16";
    first.meshFile = new File(["glb"], "fox.glb");

    // Same Pinia, second `useStore()` — exactly what a remounted view does.
    const second = useMeshWorkflowDraftStore();
    expect(second.prompt).toBe("a hand-carved wooden fox");
    expect(second.mode).toBe("mesh_texture");
    expect(second.meshModelName).toBe("hunyuan3d-2.1:fp16");
    expect(second.meshFile?.name).toBe("fox.glb");
  });

  it("restores the scalars from storage on a fresh launch", () => {
    const { storage, map } = memoryStorage();
    setMeshWorkflowDraftStorage(storage);
    const first = useMeshWorkflowDraftStore();
    first.prompt = "a brass teapot";
    first.textureResolution = 4096;
    first.upAxis = "z";
    first.metersPerUnit = 0.001;
    first.routing = "plato";
    first.persist();
    expect(map.get(MESH_WORKFLOW_DRAFT_KEY)).toBeTruthy();

    setActivePinia(createPinia());
    const relaunched = useMeshWorkflowDraftStore();
    expect(relaunched.prompt).toBe("a brass teapot");
    expect(relaunched.textureResolution).toBe(4096);
    expect(relaunched.upAxis).toBe("z");
    expect(relaunched.metersPerUnit).toBe(0.001);
    expect(relaunched.routing).toBe("plato");
  });

  /*
   * A browser cannot re-open a file handle from a previous session, so a
   * persisted name would promise bytes the next launch cannot read. The well
   * comes back empty instead.
   */
  it("never promises an attached file survived a restart", () => {
    const { storage, map } = memoryStorage();
    setMeshWorkflowDraftStorage(storage);
    const first = useMeshWorkflowDraftStore();
    first.meshFile = new File(["glb"], "fox.glb");
    first.appearanceFile = new File(["png"], "fox.png");
    first.persist();
    expect(map.get(MESH_WORKFLOW_DRAFT_KEY)).not.toContain("fox.glb");

    setActivePinia(createPinia());
    const relaunched = useMeshWorkflowDraftStore();
    expect(relaunched.meshFile).toBeNull();
    expect(relaunched.appearanceFile).toBeNull();
  });

  /*
   * A stored record is data this build did not write. One unreadable value
   * must never discard the rest of the draft — the per-key recovery rule
   * `settings::load` learned when a stale dev build moved a Nightly install
   * back to Stable and forgot every machine.
   */
  it("recovers per key from a record it cannot fully read", () => {
    const { storage } = memoryStorage({
      [MESH_WORKFLOW_DRAFT_KEY]: JSON.stringify({
        version: 1,
        prompt: "a brass teapot",
        mode: "sculpt_from_vibes",
        textureResolution: 7777,
        metersPerUnit: -3,
        upAxis: "q",
      }),
    });
    setMeshWorkflowDraftStorage(storage);
    const draft = useMeshWorkflowDraftStore();
    expect(draft.prompt).toBe("a brass teapot");
    expect(draft.mode).toBe("text_to_mesh");
    expect(draft.textureResolution).toBe(2048);
    expect(draft.metersPerUnit).toBe(1);
    expect(draft.upAxis).toBe("y");
  });

  it("opens on the defaults when storage is unreadable or empty", () => {
    setMeshWorkflowDraftStorage({
      getItem: () => {
        throw new Error("private mode");
      },
      setItem: () => {
        throw new Error("private mode");
      },
      removeItem: () => {},
    });
    const draft = useMeshWorkflowDraftStore();
    expect(draft.mode).toBe("text_to_mesh");
    expect(draft.texture).toBe(true);
    // A refused write must not throw out of the store.
    expect(() => draft.persist()).not.toThrow();
  });

  it("reports what would be lost, and clears only the authored work", () => {
    const draft = useMeshWorkflowDraftStore();
    expect(draft.dirty).toBe(false);
    draft.prompt = "a brass teapot";
    draft.meshModelName = "hunyuan3d-2.1:fp16";
    draft.selectedId = "workflow-1";
    expect(draft.dirty).toBe(true);
    draft.clear();
    expect(draft.dirty).toBe(false);
    expect(draft.selectedId).toBe("");
    expect(draft.meshModelName).toBe("hunyuan3d-2.1:fp16");
  });
});
