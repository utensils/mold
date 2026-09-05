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
    if (first) {
      first.prompt = "a kingfisher waits";
      first.cameraControl = "dolly-in";
    }
    vi.advanceTimersByTime(1000);

    const raw = localStorage.getItem(SEQUENCE_DRAFT_KEY);
    expect(raw).toBeTruthy();
    const saved = JSON.parse(raw!);
    expect(saved.version).toBe(1);
    expect(saved.clips[0].prompt).toBe("a kingfisher waits");
    expect(saved.clips[0].cameraControl).toBe("dolly-in");

    // A brand-new store (new session) restores the prompt — the desktop
    // prompt-loss regression: clip prompts must survive unmount/reload.
    const next = freshStore();
    next.hydrate();
    expect(next.clips[0]?.prompt).toBe("a kingfisher waits");
    expect(next.clips[0]?.cameraControl).toBe("dolly-in");
  });

  it("normalizes pre-camera drafts to an explicit empty selection", () => {
    localStorage.setItem(
      SEQUENCE_DRAFT_KEY,
      JSON.stringify({
        version: 1,
        output: "sequence",
        clips: [
          {
            id: "old",
            prompt: "legacy",
            frames: 97,
            transition: "smooth",
            fadeFrames: 8,
            negativePrompt: "",
            sourceImage: null,
          },
        ],
        enableAudio: false,
        lastSingleModel: null,
      }),
    );
    const store = freshStore();
    store.hydrate();
    expect(store.clips[0]?.cameraControl).toBeNull();
  });

  it("restores the sequence-level opening image from quota-safe media storage", async () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.openingImage = {
      filename: "open.png",
      width: 1170,
      height: 2532,
      base64: "QUJD",
    };
    store.clips[1]!.sourceImage = {
      filename: "second.jpg",
      width: 896,
      height: 1152,
      base64: "REVG",
    };
    vi.advanceTimersByTime(1000);

    const saved = JSON.parse(localStorage.getItem(SEQUENCE_DRAFT_KEY)!);
    expect(saved.openingImage.filename).toBe("open.png");
    expect(saved.openingImage.width).toBe(1170);
    expect(saved.openingImage.height).toBe(2532);
    expect(saved.openingImage.base64).toBeNull();
    expect(saved.clips[1].sourceImage).toMatchObject({
      filename: "second.jpg",
      width: 896,
      height: 1152,
      base64: null,
    });
    // In-memory payload stays intact.
    expect(store.openingImage?.base64).toBe("QUJD");

    const reloaded = freshStore();
    reloaded.hydrate();
    await vi.waitFor(() => expect(reloaded.openingImage?.base64).toBe("QUJD"));
    expect(reloaded.openingImage).toMatchObject({
      width: 1170,
      height: 2532,
    });
    await vi.waitFor(() =>
      expect(reloaded.clips[1]?.sourceImage).toMatchObject({
        width: 896,
        height: 1152,
        base64: "REVG",
      }),
    );
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

  it("clearOpeningImage drops only the opening image and its stored media", async () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.clips[0]!.prompt = "a river";
    store.enableAudio = true;
    store.openingImage = { filename: "open.png", base64: "QUJD" };
    vi.advanceTimersByTime(1000);
    expect(store.openingImage?.draftId).toBeTruthy();

    store.clearOpeningImage();
    vi.advanceTimersByTime(1000);

    expect(store.openingImage).toBeNull();
    expect(store.clips).toHaveLength(2);
    expect(store.clips[0]?.prompt).toBe("a river");
    expect(store.enableAudio).toBe(true);
    const saved = JSON.parse(localStorage.getItem(SEQUENCE_DRAFT_KEY)!);
    expect(saved.openingImage).toBeNull();
    // The blob is gone too: a reload must not resurrect it.
    const reloaded = freshStore();
    reloaded.hydrate();
    await vi.advanceTimersByTimeAsync(50);
    expect(reloaded.openingImage).toBeNull();
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

    // Starting over restores the MODEL's audio answer, not a flat off — the
    // caller passes the chain limits' `supports_audio`, and omitting it (as
    // here) leaves whatever the draft already had.
    store.clearSequence(25, false);

    expect(store.clips).toHaveLength(2);
    expect(store.clips.every((clip) => clip.prompt === "")).toBe(true);
    expect(store.clips.every((clip) => clip.frames === 25)).toBe(true);
    expect(store.activeClipId).toBe(store.clips[0]!.id);
    expect(store.enableAudio).toBe(false);
    expect(store.editing).toBeNull();
    // Clearing starts the story over; it does not leave sequence mode.
    expect(store.output).toBe("sequence");
  });

  it("resets only model-owned clip lengths when the selected model changes", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.clips[0]!.prompt = "opening";
    store.clips[1]!.prompt = "ending";
    store.clips[1]!.transition = "cut";

    store.resetClipFrames(53);

    expect(store.clips.map((clip) => clip.frames)).toEqual([53, 53]);
    expect(store.clips.map((clip) => clip.prompt)).toEqual([
      "opening",
      "ending",
    ]);
    expect(store.clips[1]!.transition).toBe("cut");
  });

  it("adopts the new model's audio answer, and only on a real switch", () => {
    // A sequence renders with sound wherever the model delivers it, the same
    // default a one-shot of that model gets. Switching to an audio model
    // turns it on; switching to a silent one turns it off; a re-fetch for the
    // SAME model (fps change, host refresh) must leave the user's own choice.
    const store = freshStore();
    store.hydrate();
    store.ensureClips(53);

    expect(store.adoptSequenceModel("ltx-2.3-22b-dev:fp8", 97, true)).toBe(
      true,
    );
    expect(store.enableAudio).toBe(true);

    store.enableAudio = false;
    expect(store.adoptSequenceModel("ltx-2.3-22b-dev:fp8", 97, true)).toBe(
      false,
    );
    expect(store.enableAudio).toBe(false);

    expect(store.adoptSequenceModel("wan22-i2v-a14b:q5", 53, false)).toBe(true);
    expect(store.enableAudio).toBe(false);
  });

  it("clears back to the model's audio answer rather than to silence", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.enableAudio = false;

    store.clearSequence(97, true);

    expect(store.enableAudio).toBe(true);
  });

  it("persists the model that owns clip lengths and resets only on a change", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(53);

    expect(store.adoptSequenceModel("wan22-i2v-a14b:q5", 53)).toBe(true);
    store.clips[0]!.frames = 57;
    expect(store.adoptSequenceModel("wan22-i2v-a14b:q5", 53)).toBe(false);
    expect(store.clips[0]!.frames).toBe(57);
    expect(store.adoptSequenceModel("ltx-2.3-22b-dev:fp8", 97)).toBe(true);
    expect(store.clips.map((clip) => clip.frames)).toEqual([97, 97]);
  });

  it("binds deliberately imported timings without resetting them", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    store.clips[0]!.frames = 481;

    store.bindSequenceModel("ltx-2.3-22b-dev:fp8");

    expect(store.sequenceModel).toBe("ltx-2.3-22b-dev:fp8");
    expect(store.clips[0]!.frames).toBe(481);
    expect(store.adoptSequenceModel("ltx-2.3-22b-dev:fp8", 97)).toBe(false);
    expect(store.clips[0]!.frames).toBe(481);
  });

  it("duplicates a clip after its source with its own id and media", async () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    const source = store.clips[0]!;
    source.prompt = "a kingfisher waits";
    source.negativePrompt = "blurry";
    source.cameraControl = "dolly-in";
    source.frames = 49;
    source.transition = "fade";
    source.fadeFrames = 12;
    source.sourceImage = { filename: "open.png", base64: "QUJD" };
    vi.advanceTimersByTime(1000);
    expect(source.sourceImage?.draftId).toBeTruthy();

    const copy = store.duplicateClip(source.id);

    expect(copy).not.toBeNull();
    expect(store.clips).toHaveLength(3);
    expect(store.clips[1]!.id).toBe(copy!.id);
    expect(copy!.id).not.toBe(source.id);
    expect(store.activeClipId).toBe(copy!.id);
    expect(copy!.prompt).toBe("a kingfisher waits");
    expect(copy!.negativePrompt).toBe("blurry");
    expect(copy!.cameraControl).toBe("dolly-in");
    expect(copy!.frames).toBe(49);
    expect(copy!.transition).toBe("fade");
    expect(copy!.fadeFrames).toBe(12);
    expect(copy!.sourceImage?.base64).toBe("QUJD");

    // The copy owns its own persisted blob: removing the original must not
    // delete the duplicate's media.
    vi.advanceTimersByTime(1000);
    expect(copy!.sourceImage?.draftId).toBeTruthy();
    expect(copy!.sourceImage?.draftId).not.toBe(source.sourceImage?.draftId);
    store.removeClip(source.id);
    await vi.advanceTimersByTimeAsync(50);
    expect(store.clips[0]!.sourceImage?.base64).toBe("QUJD");

    expect(store.duplicateClip("nope")).toBeNull();
  });

  it("inserts a fresh clip at a clamped index and activates it", () => {
    const store = freshStore();
    store.hydrate();
    store.ensureClips(97);
    const [a, b] = store.clips;

    const inserted = store.insertClip(1, 49);
    expect(store.clips.map((clip) => clip.id)).toEqual([
      a!.id,
      inserted.id,
      b!.id,
    ]);
    expect(inserted.frames).toBe(49);
    expect(inserted.prompt).toBe("");
    expect(store.activeClipId).toBe(inserted.id);

    const head = store.insertClip(-5, 97);
    expect(store.clips[0]!.id).toBe(head.id);
    const tail = store.insertClip(99, 97);
    expect(store.clips[store.clips.length - 1]!.id).toBe(tail.id);
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
