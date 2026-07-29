import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  LEGACY_MOBILE_MODE_KEY,
  LEGACY_WEB_DRAFT_KEY,
  LEGACY_WEB_MODE_KEY,
  SEQUENCE_DRAFT_KEY,
  setSequenceDraftStorage,
  useSequenceDraftStore,
} from "./sequenceDraft";

function memoryStorage() {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => void values.set(key, value),
    removeItem: (key: string) => void values.delete(key),
  };
}

let localStorage: ReturnType<typeof memoryStorage>;

function freshStore() {
  setActivePinia(createPinia());
  return useSequenceDraftStore();
}

describe("sequence draft store", () => {
  beforeEach(() => {
    localStorage = memoryStorage();
    setSequenceDraftStorage(localStorage);
    vi.useFakeTimers();
  });
  afterEach(() => {
    setSequenceDraftStorage(null);
    vi.useRealTimers();
  });

  it("hydrates two blank clips by default and persists edits", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    expect(store.clips).toHaveLength(2);

    const first = store.clips[0];
    if (first) first.prompt = "a kingfisher waits";
    vi.advanceTimersByTime(1000);

    const raw = localStorage.getItem(SEQUENCE_DRAFT_KEY);
    expect(raw).toBeTruthy();
    const saved = JSON.parse(raw!);
    expect(saved.version).toBe(1);
    expect(saved.clips[0].prompt).toBe("a kingfisher waits");

    // A brand-new store (new session) restores the prompt — the desktop
    // prompt-loss regression: clip prompts must survive unmount/reload.
    const next = freshStore();
    next.hydrate();
    expect(next.clips[0]?.prompt).toBe("a kingfisher waits");
  });

  it("strips source-image payloads from persistence but keeps filenames", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    const first = store.clips[0];
    if (first) first.sourceImage = { filename: "open.png", base64: "QUJD" };
    vi.advanceTimersByTime(1000);

    const saved = JSON.parse(localStorage.getItem(SEQUENCE_DRAFT_KEY)!);
    expect(saved.clips[0].sourceImage.filename).toBe("open.png");
    expect(saved.clips[0].sourceImage.base64).toBeNull();
    // In-memory payload stays intact.
    expect(store.clips[0]?.sourceImage?.base64).toBe("QUJD");
  });

  it("parks clips when switching output and bridges the prompt both ways", () => {
    const store = freshStore();
    store.hydrate();
    let singlePrompt = "a lone lighthouse";
    const bridge = {
      getPrompt: () => singlePrompt,
      setPrompt: (v: string) => {
        singlePrompt = v;
      },
    };

    store.setOutput("sequence", bridge, 97);
    expect(store.output).toBe("sequence");
    expect(store.clips[0]?.prompt).toBe("a lone lighthouse");
    expect(store.activeClipId).toBe(store.clips[0]?.id);

    const second = store.clips[1];
    if (second) second.prompt = "waves crash closer";
    store.setOutput("single", bridge, 97);
    expect(singlePrompt).toBe("a lone lighthouse");
    // Parked, never erased.
    expect(store.clips).toHaveLength(2);
    expect(store.clips[1]?.prompt).toBe("waves crash closer");
  });

  it("keeps a two-clip floor and reorders with stable ids", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    const [a, b] = store.clips;
    store.removeClip(a!.id);
    expect(store.clips).toHaveLength(2);

    store.addClip(97);
    const c = store.clips[2];
    store.moveClip(c!.id, 0);
    expect(store.clips[0]?.id).toBe(c!.id);
    expect(store.clips[1]?.id).toBe(a!.id);
    expect(store.clips[2]?.id).toBe(b!.id);
  });

  it("applies a transition to one seam or every seam", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.addClip(97);
    const second = store.clips[1];
    store.setTransition(second!.id, "fade", 12);
    expect(store.clips[1]?.transition).toBe("fade");
    expect(store.clips[1]?.fadeFrames).toBe(12);
    expect(store.clips[2]?.transition).toBe("smooth");

    store.applyTransitionToAllSeams("cut");
    expect(store.clips[1]?.transition).toBe("cut");
    expect(store.clips[2]?.transition).toBe("cut");
    // Clip 0 has no incoming seam; its value is irrelevant but harmless.
  });

  it("migrates the legacy web draft, web mode, and mobile mode keys", () => {
    localStorage.setItem(
      LEGACY_WEB_DRAFT_KEY,
      JSON.stringify({
        schema: "mold.chain.v1",
        chain: { model: "ltx-2-19b-distilled:fp8", enable_audio: true },
        stages: [
          { prompt: "opening", frames: 97 },
          { prompt: "landing", frames: 33, transition: "fade", fade_frames: 8 },
        ],
      }),
    );
    localStorage.setItem(LEGACY_WEB_MODE_KEY, "script");

    const store = freshStore();
    store.hydrate();
    expect(store.output).toBe("sequence");
    expect(store.clips).toHaveLength(2);
    expect(store.clips[0]?.prompt).toBe("opening");
    expect(store.clips[1]?.fadeFrames).toBe(8);
    expect(store.enableAudio).toBe(true);
    // Legacy keys are consumed.
    expect(localStorage.getItem(LEGACY_WEB_DRAFT_KEY)).toBeNull();
    expect(localStorage.getItem(LEGACY_WEB_MODE_KEY)).toBeNull();

    localStorage = memoryStorage();
    setSequenceDraftStorage(localStorage);
    localStorage.setItem(LEGACY_MOBILE_MODE_KEY, "sequence");
    const mobile = freshStore();
    mobile.hydrate();
    expect(mobile.output).toBe("sequence");
    expect(localStorage.getItem(LEGACY_MOBILE_MODE_KEY)).toBeNull();
  });

  it("persists a migrated legacy draft immediately (no-edit session survives)", () => {
    // migrateLegacy() deletes the legacy keys while `hydrated` is still
    // false, so the deep watcher alone would never write the new key — a
    // reload before any edit would lose the migrated draft entirely.
    localStorage.setItem(
      LEGACY_WEB_DRAFT_KEY,
      JSON.stringify({
        schema: "mold.chain.v1",
        chain: { model: "ltx-2-19b-distilled:fp8" },
        stages: [
          { prompt: "opening", frames: 97 },
          { prompt: "landing", frames: 33 },
        ],
      }),
    );

    const store = freshStore();
    store.hydrate();
    // No edits, no timers — the migrated draft must already be durable.
    const saved = JSON.parse(localStorage.getItem(SEQUENCE_DRAFT_KEY)!);
    expect(saved.clips).toHaveLength(2);
    expect(saved.clips[0].prompt).toBe("opening");

    const reloaded = freshStore();
    reloaded.hydrate();
    expect(reloaded.clips[0]?.prompt).toBe("opening");
  });

  it("migrates a real legacy web draft whose stages persist under `stage`", () => {
    // web's ScriptComposer persisted its ChainScriptToml verbatim — the
    // mold.chain.v1 TOML shape keys stages as `stage`, not `stages`.
    localStorage.setItem(
      LEGACY_WEB_DRAFT_KEY,
      JSON.stringify({
        schema: "mold.chain.v1",
        chain: { model: "ltx-2-19b-distilled:fp8", enable_audio: true },
        stage: [
          { prompt: "opening", frames: 97 },
          { prompt: "landing", frames: 33, transition: "cut" },
        ],
      }),
    );

    const store = freshStore();
    store.hydrate();
    expect(store.clips).toHaveLength(2);
    expect(store.clips[0]?.prompt).toBe("opening");
    expect(store.clips[1]?.transition).toBe("cut");
    expect(store.enableAudio).toBe(true);
    expect(localStorage.getItem(LEGACY_WEB_DRAFT_KEY)).toBeNull();

    // Composes with the persist-on-migrate fix: a `stage`-keyed draft must
    // also survive a reload taken before the user edits anything.
    const reloaded = freshStore();
    reloaded.hydrate();
    expect(reloaded.clips.map((clip) => clip.prompt)).toEqual(["opening", "landing"]);
  });

  it("tracks an edit session without persisting it", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.loadFromJob(
      {
        jobId: "c1",
        hostId: "plato",
        baseline: store.clips.map((c) => ({ ...c })),
        completedStages: 2,
      },
      store.clips.map((c) => ({ ...c })),
      false,
    );
    expect(store.editing?.jobId).toBe("c1");
    vi.advanceTimersByTime(1000);
    const saved = JSON.parse(localStorage.getItem(SEQUENCE_DRAFT_KEY)!);
    expect(saved.editing).toBeUndefined();

    store.stopEditing();
    expect(store.editing).toBeNull();
  });
});
