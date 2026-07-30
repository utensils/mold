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
import { clearMemoryDraftsForTest } from "../lib/draftMediaStore";

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
    clearMemoryDraftsForTest();
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

  it("restores the sequence-level opening image from quota-safe media storage", async () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.openingImage = { filename: "open.png", base64: "QUJD" };
    vi.advanceTimersByTime(1000);

    const saved = JSON.parse(localStorage.getItem(SEQUENCE_DRAFT_KEY)!);
    expect(saved.openingImage.filename).toBe("open.png");
    expect(saved.openingImage.base64).toBeNull();
    // In-memory payload stays intact.
    expect(store.openingImage?.base64).toBe("QUJD");

    const reloaded = freshStore();
    reloaded.hydrate();
    await vi.waitFor(() => expect(reloaded.openingImage?.base64).toBe("QUJD"));
  });

  it("parks clips while keeping one-shot and sequence prompts independent", () => {
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
    expect(store.clips[0]?.prompt).toBe("");
    expect(store.activeClipId).toBe(store.clips[0]?.id);

    const second = store.clips[1];
    if (store.clips[0]) store.clips[0].prompt = "the sequence opening";
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
    store.openingImage = { filename: "opening.png", base64: "AAAA" };
    const [a, b] = store.clips;
    store.removeClip(a!.id);
    expect(store.clips).toHaveLength(2);

    store.addClip(97);
    const c = store.clips[2];
    store.moveClip(c!.id, 0);
    expect(store.clips[0]?.id).toBe(c!.id);
    expect(store.clips[1]?.id).toBe(a!.id);
    expect(store.clips[2]?.id).toBe(b!.id);
    expect(store.openingImage).toEqual({
      filename: "opening.png",
      base64: "AAAA",
    });
  });

  it("snapshots the opening image and audio with an edit session", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.loadFromJob(
      {
        jobId: "job-1",
        hostId: "host-1",
        baseline: store.clips.map((clip) => ({ ...clip })),
        completedStages: 2,
      },
      store.clips,
      true,
      { filename: "original.png", base64: "ORIGINAL" },
    );

    expect(store.editing?.baselineOpeningImage).toEqual({
      filename: "original.png",
      base64: "ORIGINAL",
    });
    expect(store.editing?.baselineEnableAudio).toBe(true);
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
    expect(reloaded.clips.map((clip) => clip.prompt)).toEqual([
      "opening",
      "landing",
    ]);
  });

  it("clears the whole sequence back to two fresh clips, staying in Sequence", () => {
    const store = freshStore();
    store.hydrate();
    store.output = "sequence";
    store.ensureClips(25);
    store.addClip(25);
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));
    store.enableAudio = true;
    store.loadFromJob(
      { jobId: "job-1", hostId: "h1", baseline: [], completedStages: 1 },
      store.clips.map((clip) => ({ ...clip })),
      true,
    );

    store.clearSequence(25);

    expect(store.clips).toHaveLength(2);
    expect(store.clips.every((clip) => clip.prompt === "")).toBe(true);
    expect(store.clips.every((clip) => clip.frames === 25)).toBe(true);
    expect(store.activeClipId).toBe(store.clips[0]!.id);
    expect(store.enableAudio).toBe(false);
    expect(store.editing).toBeNull();
    // Clearing starts the story over; it does not leave sequence mode.
    expect(store.output).toBe("sequence");
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
