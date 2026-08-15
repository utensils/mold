import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { IDBFactory } from "fake-indexeddb";
import {
  applyMetadataToForm,
  normalizeLegacyNegativeFormState,
  cloneTemplateForm,
  promptWithStyle,
  sanitizePersistedForm,
  useGenerateForm,
  __testing__,
} from "./useGenerateForm";
import type {
  GenerateFormState,
  ModelInfoExtended,
  OutputMetadata,
} from "../types";
import { WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT } from "@studio/lib/negativePrompt";

const STORAGE_KEY = "mold.generate.form";

function makeModel(
  overrides: Partial<ModelInfoExtended> = {},
): ModelInfoExtended {
  return {
    name: "flux2-klein:q4",
    family: "flux2",
    size_gb: 6,
    is_loaded: false,
    last_used: null,
    hf_repo: "black-forest-labs/FLUX.2-Klein",
    downloaded: true,
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    ...overrides,
  };
}

async function flushDraftPersistence() {
  await vi.advanceTimersByTimeAsync(300);
  // fake-indexeddb schedules request/transaction events as tasks; flush those
  // deliberately while this suite owns fake timers.
  await vi.runAllTimersAsync();
  await __testing__.flushDraftWrites();
}

async function flushDraftHydration() {
  const hydration = __testing__.flushHydration();
  await vi.runAllTimersAsync();
  await hydration;
}

describe("useGenerateForm", () => {
  beforeEach(() => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: new IDBFactory(),
      writable: true,
      configurable: true,
    });
    localStorage.clear();
    __testing__.resetForTest();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("hydrates defaults when localStorage is empty", () => {
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
    expect(form.state.value.width).toBe(1024);
    expect(form.state.value.height).toBe(1024);
    expect(form.state.value.steps).toBe(20);
    expect(form.state.value.batchSize).toBe(1);
    expect(form.state.value.outputFormat).toBe("png");
    expect(form.state.value.imageAttachments).toEqual([]);
    expect(form.state.value.icLoraControl).toBeNull();
    expect(form.state.value.sourceFitPolicy).toEqual({ mode: "pad-repaint" });
  });

  it("serializes a host-provided IC-LoRA control without replacing custom LoRAs", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx-2.3-22b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.icLoraControl = "motion-track";
    form.state.value.sourceVideoPath = "/guides/trajectory.mp4";
    form.state.value.loras = [{ path: "/loras/style.safetensors", scale: 0.8 }];
    expect(form.toRequest()).toMatchObject({
      pipeline: "ic-lora",
      ic_lora_control: "motion-track",
      source_video_path: "/guides/trajectory.mp4",
      loras: [{ path: "/loras/style.safetensors", scale: 0.8 }],
    });
  });

  /**
   * Continuation is not LTX-2-only (#783). The extend fields rode inside the
   * LTX-2 request spread, so a wan continuation the surface now offers would
   * have gone out as a plain text-to-video request — silently, with the
   * source clip dropped.
   */
  it("sends a wan continuation's source clip and overlap", () => {
    const form = useGenerateForm();
    form.state.value.model = "wan22-i2v-a14b:q5";
    form.state.value.modelFamily = "wan";
    form.state.value.frames = 53;
    form.state.value.extendVideoPath = "/srv/mold/clip.mp4";
    form.state.value.extendOverlapFrames = 1;
    expect(form.toRequest()).toMatchObject({
      extend_video_path: "/srv/mold/clip.mp4",
      extend_overlap_frames: 1,
    });
  });

  /**
   * The overlap the control shows must be the overlap the request carries.
   * Wan offers exactly one choice, so `@change` never fires and the form field
   * stays null; leaving the wire field absent handed the host its own
   * family-wide default of 17, which `wan/pipeline.rs`'s `extend_inner`
   * refuses — every untouched wan continuation failed (#783 review).
   */
  it("submits wan's single carried frame from an untouched overlap control", () => {
    const form = useGenerateForm();
    form.state.value.model = "wan22-i2v-a14b:q5";
    form.state.value.modelFamily = "wan";
    form.state.value.frames = 53;
    form.state.value.extendVideoPath = "/srv/mold/clip.mp4";
    form.state.value.extendOverlapFrames = null;
    const request = form.toRequest(
      makeModel({
        name: "wan22-i2v-a14b:q5",
        family: "wan",
        supports_extend: true,
        // The host advertises the family-wide LTX-2 value; the clamp is the
        // client's, so it has to survive all the way onto the wire.
        extend_default_overlap_frames: 17,
      }),
    );
    expect(request.extend_overlap_frames).toBe(1);
  });

  it("submits the host's advertised default for an untouched LTX-2 continuation", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx2-13b-distilled";
    form.state.value.modelFamily = "ltx2";
    form.state.value.frames = 97;
    form.state.value.extendVideoPath = "/srv/mold/clip.mp4";
    form.state.value.extendOverlapFrames = null;
    const request = form.toRequest(
      makeModel({
        name: "ltx2-13b-distilled",
        family: "ltx2",
        supports_extend: true,
        extend_default_overlap_frames: 25,
      }),
    );
    expect(request.extend_overlap_frames).toBe(25);
  });

  it("keeps the overlap home when there is no clip to continue", () => {
    const form = useGenerateForm();
    form.state.value.model = "wan22-i2v-a14b:q5";
    form.state.value.modelFamily = "wan";
    form.state.value.frames = 53;
    expect(form.toRequest().extend_overlap_frames).toBeUndefined();
  });

  /**
   * Continuation is per family, not part of the LTX-2 suite (#783), so the
   * staged clip is cleared on leaving a family that can continue at all — not
   * on leaving LTX-2. Desktop and iPhone apply the same rule.
   */
  it("keeps a staged continuation across a switch within a continuing family", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(
      makeModel({ name: "wan22-i2v-a14b:q5", family: "wan" }),
    );
    form.state.value.extendVideoPath = "/srv/mold/clip.mp4";
    form.state.value.extendOverlapFrames = 1;

    form.applyModelDefaults(
      makeModel({ name: "wan22-i2v-a14b:q8", family: "wan" }),
    );
    expect(form.state.value.extendVideoPath).toBe("/srv/mold/clip.mp4");
    expect(form.state.value.extendOverlapFrames).toBe(1);
  });

  it("drops a staged continuation on a switch to a family that cannot continue", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(
      makeModel({ name: "wan22-i2v-a14b:q5", family: "wan" }),
    );
    form.state.value.extendVideoPath = "/srv/mold/clip.mp4";
    form.state.value.extendOverlapFrames = 1;

    form.applyModelDefaults(makeModel());
    expect(form.state.value.extendVideoPath).toBe("");
    expect(form.state.value.extendOverlapFrames).toBeNull();
  });

  it("keeps the LTX-2 suite out of a wan request", () => {
    const form = useGenerateForm();
    form.state.value.model = "wan22-i2v-a14b:q5";
    form.state.value.modelFamily = "wan";
    form.state.value.extendVideoPath = "/srv/mold/clip.mp4";
    form.state.value.sourceVideoPath = "/srv/mold/guide.mp4";
    const request = form.toRequest();
    expect(request.extend_video_path).toBe("/srv/mold/clip.mp4");
    expect(request.source_video_path).toBeUndefined();
    expect(request.pipeline).toBeUndefined();
  });

  it("projects Ref2VA through the shared H3 request authority", () => {
    const form = useGenerateForm();
    form.state.value.model = "minimax-h3-ref2va:official-bf16";
    form.state.value.modelFamily = "";
    form.state.value.prompt = "resynthesize the performance";
    form.state.value.frames = 130;
    form.state.value.fps = 30;
    form.state.value.guidance = 7;
    form.state.value.outputFormat = "gif";
    form.state.value.h3Authoring = {
      firstFrame: null,
      lastFrame: null,
      references: [
        {
          reference: {
            kind: "image",
            media: { authority: "inline", data: "IMAGE" },
            provenance: { name: "subject.png", sha256: "a".repeat(64) },
            mime_type: "image/png",
            width: 1024,
            height: 768,
          },
        },
      ],
    };

    const request = form.toRequest();
    expect(request).toMatchObject({
      frames: 124,
      fps: 24,
      guidance: 0,
      strength: 1,
      batch_size: 1,
      output_format: "mp4",
      references: [
        {
          kind: "image",
          media: { authority: "inline", data: "IMAGE" },
        },
      ],
    });
    expect(request.negative_prompt).toBeUndefined();
    expect(request.enable_audio).toBeUndefined();
  });

  it("shares a singleton state across route remounts", () => {
    const first = useGenerateForm();
    first.state.value.prompt = "persistent cat";
    first.state.value.width = 768;
    first.state.value.upscaleModel = "real-esrgan-x4plus:fp16";
    first.state.value.imageAttachments = [
      {
        kind: "upload",
        filename: "source.png",
        base64: "SOURCE_BYTES",
        draftId: "draft-source",
        width: 640,
        height: 480,
        mime: "image/png",
      },
    ];
    first.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK_BYTES",
      draftId: "draft-mask",
      width: 640,
      height: 480,
      mime: "image/png",
    };

    const second = useGenerateForm();

    expect(second.state).toBe(first.state);
    expect(second.state.value.prompt).toBe("persistent cat");
    expect(second.state.value.width).toBe(768);
    expect(second.state.value.upscaleModel).toBe("real-esrgan-x4plus:fp16");
    expect(second.state.value.imageAttachments[0]?.base64).toBe("SOURCE_BYTES");
    expect(second.state.value.maskImage?.base64).toBe("MASK_BYTES");
  });

  it("discards persisted snapshots from pre-current schemas", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 1,
        prompt: "a cat",
        model: "flux-dev:q4",
        width: 512,
        height: 768,
        // sourceImage should never be read back from storage even if someone
        // injects it — base64 lives in memory only.
        sourceImage: { kind: "upload", filename: "x.png", base64: "AAAA" },
        imageAttachments: [
          { kind: "upload", filename: "y.png", base64: "BBBB" },
        ],
      }),
    );

    const form = useGenerateForm();
    expect(form.state.value).toMatchObject({ version: 3, prompt: "" });
  });

  it("loads a version 3 snapshot preserving a saved stylePreset", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: 3, prompt: "a cat", stylePreset: "cinematic" }),
    );
    const form = useGenerateForm();
    expect(form.state.value.stylePreset).toBe("cinematic");
  });

  it("upgrades a version 3 camera picker value into the visible LoRA stack", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 3,
        model: "ltx-2-19b-distilled:fp8",
        modelFamily: "ltx2",
        cameraControl: "dolly-in",
        loras: [],
      }),
    );

    const form = useGenerateForm();
    expect(form.state.value.cameraControl).toBe("dolly-in");
    expect(form.state.value.loras).toEqual([
      {
        path: "camera-control:dolly-in",
        scale: 1,
        trainedWords: [],
      },
    ]);
  });

  it("clears a legacy picker-only camera value when all LoRA slots are occupied", () => {
    const loras = ["one", "two", "three", "four"].map((path) => ({
      path,
      scale: 1,
      trainedWords: [],
    }));
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 3,
        model: "ltx-2-19b-distilled:fp8",
        modelFamily: "ltx2",
        cameraControl: "dolly-in",
        loras,
      }),
    );

    const form = useGenerateForm();
    expect(form.state.value.cameraControl).toBeNull();
    expect(form.state.value.loras).toEqual(loras);
  });

  it("toRequest bakes the shared kit's style template without mutating the prompt", () => {
    const form = useGenerateForm();
    form.state.value.model = "flux2-klein:q4";
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.stylePreset = "cinematic";
    expect(form.toRequest().prompt).toBe(
      "cinematic film still of a lighthouse in a storm, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
    // The textarea content itself is never rewritten by the style row.
    expect(form.state.value.prompt).toBe("a lighthouse in a storm");
  });

  it("toRequest merges the preset's curated negative for families that take one", () => {
    const form = useGenerateForm();
    form.state.value.model = "sdxl-base:fp16";
    form.state.value.modelFamily = "sdxl";
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.negativePrompt = "text";
    form.state.value.stylePreset = "cinematic";
    // User fragments first, preset fragments appended.
    expect(form.toRequest().negative_prompt).toBe(
      "text, anime, cartoon, graphic, washed out",
    );
    // The visible negative field is untouched — composition happens on the way out.
    expect(form.state.value.negativePrompt).toBe("text");
  });

  it("toRequest never ships a preset negative to a family that rejects one", () => {
    const form = useGenerateForm();
    form.state.value.model = "flux2-klein:q4";
    form.state.value.modelFamily = "flux2";
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.negativePrompt = "text";
    form.state.value.stylePreset = "cinematic";
    expect(form.toRequest().negative_prompt).toBeNull();
  });

  describe("advertised default negative prompt (#787)", () => {
    const WAN_DEFAULT = "色调艳丽，过曝，静态，细节模糊不清";

    function wanModel(): ModelInfoExtended {
      return makeModel({
        name: "wan22-t2v-a14b:q5",
        family: "wan",
        default_negative_prompt: WAN_DEFAULT,
        guidance_capabilities: {
          adjustable: true,
          supports_negative_prompt: true,
        },
      });
    }

    it("prefills the advertised default when a wan model is applied", () => {
      const form = useGenerateForm();
      form.applyModelDefaults(wanModel());
      expect(form.state.value.negativePrompt).toBe(WAN_DEFAULT);
      expect(form.state.value.negativePromptDefault).toBe(WAN_DEFAULT);
    });

    it("keeps an untouched default absent on the wire", () => {
      const form = useGenerateForm();
      form.applyModelDefaults(wanModel());
      form.state.value.prompt = "a cat";
      expect(form.toRequest().negative_prompt).toBeNull();
    });

    it("ships the explicit empty opt-out when the user clears the field", () => {
      const form = useGenerateForm();
      form.applyModelDefaults(wanModel());
      form.state.value.prompt = "a cat";
      form.state.value.negativePrompt = "";
      expect(form.toRequest().negative_prompt).toBe("");
    });

    it("ships typed text verbatim", () => {
      const form = useGenerateForm();
      form.applyModelDefaults(wanModel());
      form.state.value.prompt = "a cat";
      form.state.value.negativePrompt = "hands";
      expect(form.toRequest().negative_prompt).toBe("hands");
    });

    it("withdraws an untouched default on leaving the family but keeps typed text", () => {
      const form = useGenerateForm();
      form.applyModelDefaults(wanModel());
      form.applyModelDefaults(makeModel());
      expect(form.state.value.negativePrompt).toBe("");
      expect(form.state.value.negativePromptDefault).toBe("");

      form.applyModelDefaults(wanModel());
      form.state.value.negativePrompt = "hands";
      form.applyModelDefaults(makeModel());
      expect(form.state.value.negativePrompt).toBe("hands");
    });

    it("reconciles a pre-#787 v3 snapshot once the wan row arrives", () => {
      // The stored form predates `negativePromptDefault`; the saved model is
      // still installed, so the CreatePage watcher never reapplies defaults.
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          version: 3,
          prompt: "a cat",
          model: "wan22-t2v-a14b:q5",
          modelFamily: "wan",
          negativePrompt: "",
        }),
      );
      const form = useGenerateForm();
      expect(form.state.value.negativePromptDefault).toBe("");

      form.reconcileNegativeDefault(wanModel());

      expect(form.state.value.negativePrompt).toBe(WAN_DEFAULT);
      expect(form.state.value.negativePromptDefault).toBe(WAN_DEFAULT);
    });

    it("reconcile preserves typed text and an explicit clear against a known default", () => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          version: 3,
          model: "wan22-t2v-a14b:q5",
          modelFamily: "wan",
          negativePrompt: "",
          negativePromptDefault: WAN_DEFAULT,
        }),
      );
      const form = useGenerateForm();
      form.reconcileNegativeDefault(wanModel());
      expect(form.state.value.negativePrompt).toBe("");
      expect(form.toRequest().negative_prompt).toBe("");

      form.state.value.negativePrompt = "hands";
      form.reconcileNegativeDefault(wanModel());
      expect(form.state.value.negativePrompt).toBe("hands");
    });

    it("reconcile ignores a row for a model that is no longer selected", () => {
      const form = useGenerateForm();
      form.applyModelDefaults(makeModel());
      form.reconcileNegativeDefault(wanModel());
      expect(form.state.value.negativePrompt).toBe("");
      expect(form.state.value.negativePromptDefault).toBe("");
    });

    it('keeps the family default (and the "" opt-out) against an older server', () => {
      // An older server omits the additive field; the known wan default must
      // not decay to absence, or a cleared field stops serializing "".
      const form = useGenerateForm();
      form.applyModelDefaults(
        makeModel({
          name: "wan22-t2v-a14b:q5",
          family: "wan",
          guidance_capabilities: {
            adjustable: true,
            supports_negative_prompt: true,
          },
        }),
      );
      expect(form.state.value.negativePromptDefault).toBe(
        WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT,
      );
      expect(form.state.value.negativePrompt).toBe(
        WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT,
      );
      form.state.value.prompt = "a cat";
      form.state.value.negativePrompt = "";
      expect(form.toRequest().negative_prompt).toBe("");
    });

    it('restores absent metadata negatives as the model\'s default, and "" as empty', () => {
      const metadata = {
        prompt: "a cat",
        model: "wan22-t2v-a14b:q5",
        seed: 7,
        steps: 30,
        guidance: 6,
        width: 832,
        height: 480,
        version: "1",
      } as OutputMetadata;
      const base = JSON.parse(
        JSON.stringify(useGenerateForm().state.value),
      ) as GenerateFormState;
      const restored = applyMetadataToForm(base, metadata, {
        models: [wanModel()],
      });
      expect(restored.negativePrompt).toBe(WAN_DEFAULT);

      const optedOut = applyMetadataToForm(
        base,
        { ...metadata, negative_prompt: "" },
        { models: [wanModel()] },
      );
      expect(optedOut.negativePrompt).toBe("");
    });

    it("an explicit clear restored before rows load survives the wan row arriving", () => {
      // Reuse settings with the inventory not yet resolved: the model row is
      // unknown, the default restores as "", and the saved `""` opt-out is
      // visually identical to "untouched". The marker keeps the print's
      // authority so the later reconcile does not prefill over it — and the
      // wire still ships the explicit "" (#787 round 3).
      const metadata = {
        prompt: "a cat",
        model: "wan22-t2v-a14b:q5",
        negative_prompt: "",
        seed: 7,
        steps: 30,
        guidance: 6,
        width: 832,
        height: 480,
        version: "1",
      } as OutputMetadata;
      const base = JSON.parse(
        JSON.stringify(useGenerateForm().state.value),
      ) as GenerateFormState;

      const restored = applyMetadataToForm(base, metadata, { models: [] });
      expect(restored.negativePrompt).toBe("");
      expect(restored.negativeExplicitClear).toBe(true);

      const form = useGenerateForm();
      form.state.value = restored;
      form.reconcileNegativeDefault(wanModel());
      expect(form.state.value.negativePrompt).toBe("");
      expect(form.state.value.negativePromptDefault).toBe(WAN_DEFAULT);
      expect(form.toRequest().negative_prompt).toBe("");
    });

    it("absence restored before rows load still takes the prefill on reconcile", () => {
      const metadata = {
        prompt: "a cat",
        model: "wan22-t2v-a14b:q5",
        seed: 7,
        steps: 30,
        guidance: 6,
        width: 832,
        height: 480,
        version: "1",
      } as OutputMetadata;
      const base = JSON.parse(
        JSON.stringify(useGenerateForm().state.value),
      ) as GenerateFormState;

      const restored = applyMetadataToForm(base, metadata, { models: [] });
      expect(restored.negativeExplicitClear).toBe(false);

      const form = useGenerateForm();
      form.state.value = restored;
      form.reconcileNegativeDefault(wanModel());
      expect(form.state.value.negativePrompt).toBe(WAN_DEFAULT);
    });

    it("selecting a model resolves and resets the deferred clear marker", () => {
      const form = useGenerateForm();
      form.state.value.negativeExplicitClear = true;
      form.applyModelDefaults(wanModel());
      expect(form.state.value.negativeExplicitClear).toBe(false);
    });

    it("normalizeLegacyNegativeFormState resolves a pre-#787 template snapshot", () => {
      const legacy = JSON.parse(
        JSON.stringify(useGenerateForm().state.value),
      ) as GenerateFormState;
      legacy.model = "wan22-t2v-a14b:q5";
      legacy.modelFamily = "wan";
      legacy.negativePrompt = "";
      delete legacy.negativePromptDefault;
      delete legacy.negativeExplicitClear;

      const normalized = normalizeLegacyNegativeFormState(legacy, [wanModel()]);
      expect(normalized.negativePrompt).toBe(WAN_DEFAULT);
      expect(normalized.negativePromptDefault).toBe(WAN_DEFAULT);
      expect(normalized.negativeExplicitClear).toBe(false);

      // Post-#787 snapshots are authority and pass through untouched.
      const cleared = JSON.parse(
        JSON.stringify(normalized),
      ) as GenerateFormState;
      cleared.negativePrompt = "";
      expect(
        normalizeLegacyNegativeFormState(cleared, [wanModel()]).negativePrompt,
      ).toBe("");
    });
  });

  it("toRequest sends the bare prompt when no style preset is active", () => {
    const form = useGenerateForm();
    form.state.value.model = "flux2-klein:q4";
    form.state.value.prompt = "a lighthouse in a storm";
    form.state.value.stylePreset = null;
    expect(form.toRequest().prompt).toBe("a lighthouse in a storm");
  });

  it("promptWithStyle leaves an empty prompt empty (a template has nothing to wrap)", () => {
    const form = useGenerateForm();
    form.state.value.prompt = "";
    form.state.value.stylePreset = "anime";
    expect(promptWithStyle(form.state.value)).toBe("");
  });

  it("discards a snapshot with a mismatched version to avoid stale schemas", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: 99, prompt: "stale" }),
    );
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
  });

  it("swallows malformed JSON without throwing", () => {
    localStorage.setItem(STORAGE_KEY, "{not json");
    const form = useGenerateForm();
    expect(form.state.value.prompt).toBe("");
  });

  it("debounces persistence and strips attachment bytes from the written snapshot", async () => {
    const form = useGenerateForm();
    form.state.value.prompt = "a fox";
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "x.png", base64: "SECRETBYTES" },
    ];
    await nextTick();

    // Watch fires but persist is debounced by 300ms.
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();

    await flushDraftPersistence();

    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!);
    expect(parsed.prompt).toBe("a fox");
    expect(parsed.imageAttachments[0]).toMatchObject({
      filename: "x.png",
      draftId: expect.any(String),
    });
    expect(parsed.imageAttachments[0].base64).toBeUndefined();
    expect(raw).not.toContain("SECRETBYTES");
  });

  it("hydrates media drafts from the draft store after refresh", async () => {
    const first = useGenerateForm();
    first.state.value.imageAttachments = [
      { kind: "upload", filename: "source.png", base64: "SOURCE_BYTES" },
    ];
    first.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "MASK_BYTES",
    };
    first.state.value.controlImage = {
      kind: "upload",
      filename: "control.png",
      base64: "CONTROL_BYTES",
    };
    await nextTick();
    await flushDraftPersistence();

    __testing__.clearDraftsForTest();
    __testing__.resetForTest();
    const second = useGenerateForm();
    await flushDraftHydration();

    expect(second.state.value.imageAttachments[0]?.base64).toBe("SOURCE_BYTES");
    expect(second.state.value.maskImage?.base64).toBe("MASK_BYTES");
    expect(second.state.value.controlImage?.base64).toBe("CONTROL_BYTES");
    expect(localStorage.getItem(STORAGE_KEY)).not.toContain("_BYTES");
  });

  it("persists and hydrates H3 endpoint and ordered-reference bytes outside localStorage", async () => {
    const first = useGenerateForm();
    first.state.value.model = "minimax-h3-ref2va:official-bf16";
    first.state.value.modelFamily = "minimax-h3";
    first.state.value.h3Authoring = {
      firstFrame: {
        filename: "opening.png",
        mimeType: "image/png",
        width: 768,
        height: 512,
        data: "H3_FIRST_BYTES",
        sha256: "a".repeat(64),
      },
      lastFrame: {
        filename: "closing.png",
        mimeType: "image/png",
        width: 768,
        height: 512,
        data: "H3_LAST_BYTES",
        sha256: "b".repeat(64),
      },
      references: [
        {
          reference: {
            kind: "image",
            media: { authority: "inline", data: "H3_REFERENCE_ONE_BYTES" },
            provenance: { name: "subject.png", sha256: "c".repeat(64) },
            mime_type: "image/png",
            width: 512,
            height: 512,
          },
        },
        {
          reference: {
            kind: "audio",
            media: { authority: "inline", data: "H3_REFERENCE_TWO_BYTES" },
            provenance: { name: "voice.wav", sha256: "d".repeat(64) },
            mime_type: "audio/wav",
            duration_ms: 2_000,
            sample_rate: 32_000,
            channels: 2,
            sample_count: 64_000,
          },
        },
      ],
    };
    await nextTick();
    await flushDraftPersistence();

    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).not.toBeNull();
    expect(raw).not.toContain("H3_FIRST_BYTES");
    expect(raw).not.toContain("H3_LAST_BYTES");
    expect(raw).not.toContain("H3_REFERENCE_ONE_BYTES");
    expect(raw).not.toContain("H3_REFERENCE_TWO_BYTES");
    const persisted = JSON.parse(raw!);
    expect(persisted.h3Authoring.firstFrame).toMatchObject({
      data: "",
      draftId: expect.any(String),
    });
    expect(persisted.h3Authoring.lastFrame).toMatchObject({
      data: "",
      draftId: expect.any(String),
    });
    expect(persisted.h3Authoring.references).toEqual([
      expect.objectContaining({
        draftId: expect.any(String),
        reference: expect.objectContaining({
          media: { authority: "descriptor" },
        }),
      }),
      expect.objectContaining({
        draftId: expect.any(String),
        reference: expect.objectContaining({
          media: { authority: "descriptor" },
        }),
      }),
    ]);

    // Metadata-only edits must reuse the already durable H3 media instead of
    // rewriting hundreds of MiB. Making IndexedDB unavailable proves this
    // second save does not attempt another media transaction.
    const durableFactory = globalThis.indexedDB;
    Object.defineProperty(globalThis, "indexedDB", {
      value: undefined,
      writable: true,
      configurable: true,
    });
    first.state.value.prompt = "metadata-only edit";
    await nextTick();
    await flushDraftPersistence();
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).prompt).toBe(
      "metadata-only edit",
    );
    Object.defineProperty(globalThis, "indexedDB", {
      value: durableFactory,
      writable: true,
      configurable: true,
    });

    // Prove the reload comes from IndexedDB, not the module's memory cache.
    __testing__.clearDraftsForTest();
    __testing__.resetForTest();
    const second = useGenerateForm();
    await flushDraftHydration();

    expect(second.state.value.h3Authoring?.firstFrame?.data).toBe(
      "H3_FIRST_BYTES",
    );
    expect(second.state.value.h3Authoring?.lastFrame?.data).toBe(
      "H3_LAST_BYTES",
    );
    expect(
      second.state.value.h3Authoring?.references.map(
        (draft) => draft.reference.media,
      ),
    ).toEqual([
      { authority: "inline", data: "H3_REFERENCE_ONE_BYTES" },
      { authority: "inline", data: "H3_REFERENCE_TWO_BYTES" },
    ]);
  });

  it("projects descriptors without JSON-cloning media payloads", () => {
    const forbiddenPayload = {
      toJSON() {
        throw new Error("media payload reached JSON cloning");
      },
    } as unknown as string;
    const form = useGenerateForm();
    form.state.value.imageAttachments = [
      {
        kind: "upload",
        filename: "ordinary.png",
        base64: forbiddenPayload,
      },
    ];
    form.state.value.h3Authoring = {
      firstFrame: {
        filename: "opening.png",
        mimeType: "image/png",
        width: 768,
        height: 512,
        data: forbiddenPayload,
      },
      lastFrame: null,
      references: [
        {
          reference: {
            kind: "image",
            media: { authority: "inline", data: forbiddenPayload },
            provenance: { name: "reference.png", sha256: "a".repeat(64) },
            mime_type: "image/png",
            width: 512,
            height: 512,
          },
        },
      ],
    };

    expect(() => sanitizePersistedForm(form.state.value)).not.toThrow();
    const sanitized = sanitizePersistedForm(form.state.value);
    expect(sanitized.imageAttachments[0]?.base64).toBeUndefined();
    expect(sanitized.h3Authoring?.firstFrame?.data).toBe("");
    expect(sanitized.h3Authoring?.references[0]?.reference.media).toEqual({
      authority: "descriptor",
    });
  });

  it("fails closed without IndexedDB and retries the complete H3 draft on the next change", async () => {
    const durableFactory = globalThis.indexedDB;
    Object.defineProperty(globalThis, "indexedDB", {
      value: undefined,
      writable: true,
      configurable: true,
    });

    const form = useGenerateForm();
    form.state.value.model = "minimax-h3-fl2va:official-bf16";
    form.state.value.modelFamily = "minimax-h3";
    form.state.value.h3Authoring = {
      firstFrame: {
        filename: "opening.png",
        mimeType: "image/png",
        width: 768,
        height: 512,
        data: "H3_RETRY_BYTES",
      },
      lastFrame: null,
      references: [],
    };
    await nextTick();
    await flushDraftPersistence();

    // A durable descriptor must never point at bytes that existed only in the
    // form singleton. Keep the previous valid snapshot (none here) untouched.
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();

    Object.defineProperty(globalThis, "indexedDB", {
      value: durableFactory,
      writable: true,
      configurable: true,
    });
    form.state.value.prompt = "retry after storage returns";
    await nextTick();
    await flushDraftPersistence();

    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).not.toBeNull();
    expect(raw).not.toContain("H3_RETRY_BYTES");
    expect(JSON.parse(raw!).h3Authoring.firstFrame).toMatchObject({
      draftId: expect.any(String),
      data: "",
    });

    __testing__.clearDraftsForTest();
    __testing__.resetForTest();
    const reloaded = useGenerateForm();
    await flushDraftHydration();
    expect(reloaded.state.value.h3Authoring?.firstFrame?.data).toBe(
      "H3_RETRY_BYTES",
    );
  });

  it("applyModelDefaults copies model defaults and clears video fields for non-video families", () => {
    const form = useGenerateForm();
    form.state.value.frames = 25;
    form.state.value.fps = 24;

    form.applyModelDefaults(
      makeModel({
        name: "sdxl:fp16",
        family: "sdxl",
        default_width: 1024,
        default_height: 1024,
        default_steps: 30,
        default_guidance: 7.5,
      }),
    );

    expect(form.state.value.model).toBe("sdxl:fp16");
    expect(form.state.value.steps).toBe(30);
    expect(form.state.value.guidance).toBe(7.5);
    expect(form.state.value.frames).toBeNull();
    expect(form.state.value.fps).toBeNull();
  });

  it("applyModelDefaults seeds frame/fps for video families when absent", () => {
    const form = useGenerateForm();
    form.state.value.frames = null;
    form.state.value.fps = null;

    form.applyModelDefaults(
      makeModel({ name: "ltx-video:fp16", family: "ltx-video" }),
    );

    expect(form.state.value.frames).toBe(25);
    expect(form.state.value.fps).toBe(24);
  });

  it("applyModelDefaults takes the model's advertised fps, like steps and guidance", () => {
    const form = useGenerateForm();
    form.state.value.fps = 24;

    form.applyModelDefaults(
      makeModel({
        name: "ltx-video-0.9.6-distilled:bf16",
        family: "ltx-video",
        default_fps: 30,
      }),
    );

    expect(form.state.value.fps).toBe(30);
  });

  it("applyModelDefaults preserves user-chosen frame/fps for video families", () => {
    const form = useGenerateForm();
    form.state.value.frames = 49;
    form.state.value.fps = 30;

    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));

    expect(form.state.value.frames).toBe(49);
    expect(form.state.value.fps).toBe(30);
  });

  it("applyModelDefaults clears a scheduler the new family's options reject", () => {
    const form = useGenerateForm();
    // A UNet scheduler carried into wan is a server-side rejection, not a
    // no-op: both families support "a scheduler", but their option sets are
    // disjoint apart from uni-pc.
    form.state.value.scheduler = "ddim";
    form.applyModelDefaults(
      makeModel({ name: "wan22-t2v-a14b:q4", family: "wan" }),
    );
    expect(form.state.value.scheduler).toBeNull();

    // A wan solver carried back into an SD family is cleared the same way…
    form.state.value.scheduler = "dpm-pp";
    form.applyModelDefaults(makeModel({ name: "sdxl:fp16", family: "sdxl" }));
    expect(form.state.value.scheduler).toBeNull();

    // …while a value both families accept survives the switch.
    form.state.value.scheduler = "uni-pc";
    form.applyModelDefaults(
      makeModel({ name: "wan22-t2v-a14b:q4", family: "wan" }),
    );
    expect(form.state.value.scheduler).toBe("uni-pc");
  });

  it("never serializes the picker's default scheduler sentinel", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "wan22-t2v-a14b:q4",
      modelFamily: "wan",
      prompt: "a cat",
      // The select's omit-sentinel: the wire enum has no such variant, so a
      // stored "default" must serialize as an absent field, not a 422.
      scheduler: "default",
    });
    expect(form.toRequest().scheduler).toBeUndefined();
  });

  it("applyModelDefaults enters H3 on its valid minimum frame and fixed-fps contract", () => {
    const form = useGenerateForm();
    form.state.value.frames = 25;

    form.applyModelDefaults(
      makeModel({
        name: "minimax-h3-fl2va:official-bf16",
        family: "minimax-h3",
        default_frames: 124,
        default_fps: 24,
      }),
    );

    expect(form.state.value.frames).toBe(124);
    expect(form.state.value.fps).toBe(24);
    expect(form.state.value.outputFormat).toBe("mp4");
  });

  it("resetSettings restores the selected model's defaults", () => {
    const form = useGenerateForm();
    const model = makeModel({
      name: "sdxl:fp16",
      family: "sdxl",
      default_width: 1024,
      default_height: 1024,
      default_steps: 30,
      default_guidance: 7.5,
    });
    form.applyModelDefaults(model);
    Object.assign(form.state.value, {
      width: 512,
      height: 512,
      steps: 4,
      guidance: 11,
      seedMode: "static",
      seed: 42,
      strength: 0.3,
      negativePrompt: "blurry",
      loras: [{ path: "a.safetensors", scale: 0.8 }],
      upscaleModel: "real-esrgan-x4plus:fp16",
      scheduler: "ddim",
      cfgPlus: true,
      outputFormat: "webp",
      sourceFitPolicy: { mode: "crop-fill" },
      imageAttachments: [
        { kind: "upload", filename: "src.png", base64: "AAAA" },
      ],
      gifPreview: true,
    });

    form.resetSettings(model);

    expect(form.state.value.width).toBe(1024);
    expect(form.state.value.height).toBe(1024);
    expect(form.state.value.steps).toBe(30);
    expect(form.state.value.guidance).toBe(7.5);
    expect(form.state.value.seedMode).toBe("random");
    expect(form.state.value.seed).toBeNull();
    expect(form.state.value.strength).toBe(0.75);
    expect(form.state.value.negativePrompt).toBe("");
    expect(form.state.value.loras).toEqual([]);
    expect(form.state.value.upscaleModel).toBe("");
    expect(form.state.value.scheduler).toBeNull();
    expect(form.state.value.cfgPlus).toBe(false);
    expect(form.state.value.outputFormat).toBe("png");
    expect(form.state.value.sourceFitPolicy).toEqual({ mode: "pad-repaint" });
    expect(form.state.value.imageAttachments).toEqual([]);
    expect(form.state.value.gifPreview).toBe(false);
  });

  it("resetSettings preserves the prompt, style, model and batch size", () => {
    const form = useGenerateForm();
    const model = makeModel({ name: "sdxl:fp16", family: "sdxl" });
    form.applyModelDefaults(model);
    Object.assign(form.state.value, {
      prompt: "a lighthouse in a storm",
      stylePreset: "cinematic",
      // Prepared batch work is never silently resized (CLAUDE.md), so the
      // batch stepper survives a settings reset.
      batchSize: 4,
      steps: 3,
    });

    form.resetSettings(model);

    expect(form.state.value.prompt).toBe("a lighthouse in a storm");
    expect(form.state.value.stylePreset).toBe("cinematic");
    expect(form.state.value.model).toBe("sdxl:fp16");
    expect(form.state.value.modelFamily).toBe("sdxl");
    expect(form.state.value.batchSize).toBe(4);
    expect(form.state.value.steps).toBe(20);
  });

  it("resetSettings falls back to plain defaults with no resolved model row", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "mystery:fp16",
      modelFamily: "flux",
      prompt: "a cat",
      steps: 3,
      guidance: 11,
    });

    form.resetSettings(null);

    expect(form.state.value.model).toBe("mystery:fp16");
    expect(form.state.value.modelFamily).toBe("flux");
    expect(form.state.value.prompt).toBe("a cat");
    expect(form.state.value.steps).toBe(20);
    expect(form.state.value.guidance).toBe(3.5);
  });

  it("resetSettings restores video defaults for video families", () => {
    const form = useGenerateForm();
    const model = makeModel({ name: "ltx2:fp8", family: "ltx2" });
    form.applyModelDefaults(model);
    Object.assign(form.state.value, {
      frames: 97,
      fps: 30,
      pipeline: "two-stage",
      spatialUpscale: "x2",
      keyframes: [
        {
          frame: 8,
          image: { kind: "upload", filename: "k.png", base64: "AA" },
        },
      ],
    });

    form.resetSettings(model);

    expect(form.state.value.frames).toBe(25);
    expect(form.state.value.fps).toBe(24);
    expect(form.state.value.pipeline).toBeNull();
    expect(form.state.value.spatialUpscale).toBeNull();
    expect(form.state.value.keyframes).toEqual([]);
    expect(form.state.value.outputFormat).toBe("mp4");
  });

  it("toRequest maps camelCase state to snake_case wire payload", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      prompt: "a cat",
      negativePrompt: "blurry",
      model: "sdxl:fp16",
      width: 1024,
      height: 1024,
      steps: 30,
      guidance: 7.5,
      seedMode: "static",
      seed: 42,
      batchSize: 2,
      outputFormat: "png",
      scheduler: "ddim",
      strength: 0.8,
      frames: null,
      fps: null,
      expand: { enabled: true, variations: 3, familyOverride: null },
      imageAttachments: [
        { kind: "upload", filename: "src.png", base64: "AAAA" },
      ],
    });

    const wire = form.toRequest();
    expect(wire).toMatchObject({
      prompt: "a cat",
      negative_prompt: "blurry",
      model: "sdxl:fp16",
      width: 1024,
      height: 1024,
      steps: 30,
      guidance: 7.5,
      seed: 42,
      batch_size: 2,
      output_format: "png",
      scheduler: "ddim",
      strength: 0.8,
      source_image: "AAAA",
      expand: true,
    });
    expect(wire.edit_images).toBeUndefined();
  });

  it("serializes ordered Qwen edit attachments as edit_images and omits source_image", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "qwen-image-edit:q4",
      modelFamily: "qwen-image-edit",
      batchSize: 4,
      imageAttachments: [
        { kind: "upload", filename: "target.png", base64: "TARGET" },
        { kind: "upload", filename: "ref-a.png", base64: "REF_A" },
        { kind: "gallery", filename: "ref-b.png", base64: "REF_B" },
      ],
    });

    const wire = form.toRequest();
    expect(wire.edit_images).toEqual(["TARGET", "REF_A", "REF_B"]);
    expect(wire.source_image).toBeUndefined();
    expect(wire.batch_size).toBe(1);
    expect(wire.strength).toBeUndefined();
  });

  it("omits stale mask state for Qwen edit requests", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "qwen-image-edit:q4",
      modelFamily: "qwen-image-edit",
      imageAttachments: [
        { kind: "upload", filename: "target.png", base64: "TARGET" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });

    const wire = form.toRequest();

    expect(wire.edit_images).toEqual(["TARGET"]);
    expect(wire.mask_image).toBeUndefined();
    expect(wire.source_image).toBeUndefined();
  });

  it("serializes FLUX.2 Dev attachments as ordered references", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "flux2-dev:bf16",
      modelFamily: "flux2",
      batchSize: 3,
      imageAttachments: [
        { kind: "upload", filename: "one.png", base64: "REF_ONE" },
        { kind: "upload", filename: "two.png", base64: "REF_TWO" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });

    const wire = form.toRequest();
    expect(wire.edit_images).toEqual(["REF_ONE", "REF_TWO"]);
    expect(wire.source_image).toBeUndefined();
    expect(wire.mask_image).toBeUndefined();
    expect(wire.strength).toBeUndefined();
    expect(wire.batch_size).toBe(1);
  });

  it("omits stale LoRA state from FLUX.2 Dev requests", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "flux2-dev:bf16",
      modelFamily: "flux2",
      loras: [{ path: "/loras/stale.safetensors", scale: 0.8 }],
    });

    expect(form.toRequest().loras).toBeUndefined();
  });

  it("preserves text-only FLUX.2 Dev batches", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "flux2-dev:bf16",
      modelFamily: "flux2",
      batchSize: 3,
      imageAttachments: [],
    });

    expect(form.toRequest().batch_size).toBe(3);
  });

  it("serializes an uploaded mask image for non-edit img2img requests", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sdxl:fp16",
      modelFamily: "sdxl",
      imageAttachments: [
        { kind: "upload", filename: "source.png", base64: "SOURCE" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
    });

    const wire = form.toRequest();

    expect(wire.source_image).toBe("SOURCE");
    expect(wire.source_image_name).toBe("source.png");
    expect(wire.mask_image).toBe("MASK");
    expect(wire.edit_images).toBeUndefined();
  });

  it("serializes LoRAs in the visible stack order", () => {
    const form = useGenerateForm();
    form.state.value.model = "flux-dev:q4";
    form.state.value.modelFamily = "flux";
    form.state.value.loras = [
      { path: "/loras/second.safetensors", scale: 0.9 },
      { path: "/loras/first.safetensors", scale: 1.2 },
      { path: "/loras/third.safetensors", scale: 0.5 },
    ];

    expect(form.toRequest().loras).toEqual([
      { path: "/loras/second.safetensors", scale: 0.9 },
      { path: "/loras/first.safetensors", scale: 1.2 },
      { path: "/loras/third.safetensors", scale: 0.5 },
    ]);
  });

  it("serializes an LTX-2 camera preset as the camera-control LoRA alias", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.cameraControl = "dolly-in";
    expect(form.toRequest().loras).toContainEqual({
      path: "camera-control:dolly-in",
      scale: 1,
    });
  });

  it("uses the camera strength edited in the visible LoRA stack", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.cameraControl = "dolly-in";
    form.state.value.loras = [{ path: "camera-control:dolly-in", scale: 0.45 }];
    expect(form.toRequest().loras).toEqual([
      { path: "camera-control:dolly-in", scale: 0.45 },
    ]);
  });

  it("never evicts a user LoRA from a full stack to serialize camera control", () => {
    const form = useGenerateForm();
    form.state.value.model = "ltx-2-19b-distilled:fp8";
    form.state.value.modelFamily = "ltx2";
    form.state.value.cameraControl = "dolly-in";
    form.state.value.loras = ["one", "two", "three", "four"].map((path) => ({
      path,
      scale: 1,
    }));

    expect(form.toRequest().loras).toEqual([
      { path: "one", scale: 1 },
      { path: "two", scale: 1 },
      { path: "three", scale: 1 },
      { path: "four", scale: 1 },
    ]);
  });

  it("serializes only the first attachment as source_image for non-edit families", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sdxl:fp16",
      modelFamily: "sdxl",
      imageAttachments: [
        { kind: "upload", filename: "target.png", base64: "TARGET" },
        { kind: "upload", filename: "ignored.png", base64: "IGNORED" },
      ],
    });

    const wire = form.toRequest();
    expect(wire.source_image).toBe("TARGET");
    expect(wire.edit_images).toBeUndefined();
  });

  it("prunes source and video fields that are irrelevant to the selected family", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sdxl:fp16",
      modelFamily: "sdxl",
      imageAttachments: [],
      strength: 0.42,
      frames: 97,
      fps: 30,
      gifPreview: true,
    });

    const wire = form.toRequest();
    expect(wire.strength).toBeUndefined();
    expect(wire.frames).toBeUndefined();
    expect(wire.fps).toBeUndefined();
    expect(wire.gif_preview).toBeUndefined();
  });

  it("keeps video fields for a wan model whose stored family is empty", () => {
    // `selectedFamily` falls back to the model name only when `modelFamily` is
    // absent — a restored draft that predates the field. Without the wan arm
    // the family resolves to "", `supportsVideo` is false, and `toRequest`
    // prunes exactly the fields that make it a video request: the same request
    // is then submitted as a still.
    const form = useGenerateForm();
    for (const model of [
      "wan22-t2v-a14b:q5",
      "wan22-i2v-a14b:q8",
      "wan21-t2v-1.3b:bf16",
      "wan22-ti2v-5b:fp16",
    ]) {
      Object.assign(form.state.value, {
        model,
        modelFamily: "",
        frames: 81,
        fps: 16,
        gifPreview: true,
      });
      const wire = form.toRequest();
      expect(wire.frames, model).toBe(81);
      expect(wire.fps, model).toBe(16);
    }

    // A server-supplied family still wins over the name heuristic.
    Object.assign(form.state.value, {
      model: "wan22-t2v-a14b:q5",
      modelFamily: "sdxl",
      frames: 81,
      fps: 16,
    });
    expect(form.toRequest().frames).toBeUndefined();
  });

  it("serializes backend-supported advanced generation knobs", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "ltx-2.3-22b-distilled:fp8",
      modelFamily: "ltx2",
      seedMode: "static",
      seed: 123,
      cfgPlus: true,
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK" },
      controlImage: {
        kind: "upload",
        filename: "control.png",
        base64: "CONTROL",
      },
      controlModel: "controlnet-canny-sd15",
      controlScale: 0.8,
      upscaleModel: "real-esrgan-x4plus:fp16",
      gifPreview: true,
      audioFile: { kind: "upload", filename: "voice.wav", base64: "VOICE" },
      audioFilePath: "",
      sourceVideo: { kind: "upload", filename: "clip.mp4", base64: "VIDEO" },
      sourceVideoPath: "",
      keyframes: [
        {
          frame: 0,
          image: { kind: "upload", filename: "first.png", base64: "FIRST" },
        },
        {
          frame: 24,
          image: { kind: "upload", filename: "last.png", base64: "LAST" },
        },
      ],
      pipeline: "keyframe",
      retakeRange: { start_seconds: 1.25, end_seconds: 3.5 },
      spatialUpscale: "x1-5",
      temporalUpscale: "x2",
    });

    const wire = form.toRequest();

    expect(wire.seed).toBe(123);
    expect(wire.cfg_plus).toBeUndefined();
    expect(wire.mask_image).toBe("MASK");
    expect(wire.control_image).toBeUndefined();
    expect(wire.control_model).toBeUndefined();
    expect(wire.control_scale).toBeUndefined();
    expect(wire.upscale_model).toBe("real-esrgan-x4plus:fp16");
    expect(wire.gif_preview).toBe(true);
    expect(wire.audio_file).toBe("VOICE");
    expect(wire.audio_file_path).toBeUndefined();
    expect(wire.source_video).toBe("VIDEO");
    expect(wire.source_video_path).toBeUndefined();
    expect(wire.keyframes).toEqual([
      { frame: 0, image: "FIRST", name: "first.png" },
      { frame: 24, image: "LAST", name: "last.png" },
    ]);
    expect(wire.pipeline).toBe("keyframe");
    expect(wire.retake_range).toEqual({
      start_seconds: 1.25,
      end_seconds: 3.5,
    });
    expect(wire.spatial_upscale).toBe("x1-5");
    expect(wire.temporal_upscale).toBe("x2");
  });

  it("omits guidance overrides until one is set", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "ltx-2-19b-distilled:fp8",
      modelFamily: "ltx2",
    });

    expect(form.toRequest().guidance_overrides).toBeUndefined();

    form.state.value.guidanceOverrides = {
      stgScale: 1.5,
      stgBlocks: "28, 29",
      rescaleScale: null,
      modalityScale: null,
      skipStep: 2,
    };

    expect(form.toRequest().guidance_overrides).toEqual({
      stg_scale: 1.5,
      stg_blocks: [28, 29],
      skip_step: 2,
    });
  });

  it("serializes CFG++ only for SD3-family models", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sd3.5-large:fp16",
      modelFamily: "sd3.5",
      cfgPlus: true,
    });

    expect(form.toRequest().cfg_plus).toBe(true);
  });

  it("serializes ControlNet inputs for SD1.5 models", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "sd15:fp16",
      modelFamily: "sd15",
      controlImage: {
        kind: "upload",
        filename: "control.png",
        base64: "CONTROL",
      },
      controlModel: "controlnet-canny-sd15:fp16",
      controlScale: 0.8,
    });

    const wire = form.toRequest();

    expect(wire.control_image).toBe("CONTROL");
    expect(wire.control_model).toBe("controlnet-canny-sd15:fp16");
    expect(wire.control_scale).toBe(0.8);
  });

  it("omits seed when seed mode is random even if a numeric seed is present", () => {
    const form = useGenerateForm();
    form.state.value.seedMode = "random";
    form.state.value.seed = 123;

    expect(form.toRequest().seed).toBeNull();
  });

  it("model switching forces Qwen edit batch to 1 and trims multi-image attachments for non-edit families", () => {
    const form = useGenerateForm();
    form.state.value.batchSize = 3;
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "target.png", base64: "TARGET" },
      { kind: "upload", filename: "ref.png", base64: "REF" },
    ];

    form.applyModelDefaults(
      makeModel({ name: "qwen-image-edit:q4", family: "qwen-image-edit" }),
    );
    expect(form.state.value.modelFamily).toBe("qwen-image-edit");
    expect(form.state.value.batchSize).toBe(1);
    expect(form.state.value.imageAttachments).toHaveLength(2);

    form.applyModelDefaults(makeModel({ name: "sdxl:fp16", family: "sdxl" }));
    expect(form.state.value.modelFamily).toBe("sdxl");
    expect(form.state.value.batchSize).toBe(1);
    expect(form.state.value.imageAttachments).toHaveLength(1);
    expect(form.state.value.imageAttachments[0]?.base64).toBe("TARGET");
  });

  it("model switching preserves a text-only batch for FLUX.2 Dev", () => {
    const form = useGenerateForm();
    form.state.value.batchSize = 4;

    form.applyModelDefaults(
      makeModel({ name: "flux2-dev:bf16", family: "flux2" }),
    );

    expect(form.state.value.batchSize).toBe(4);
    expect(form.state.value.imageAttachments).toEqual([]);
  });

  it("toRequest omits expand entirely when disabled (server treats missing/false the same, but this keeps payload minimal)", () => {
    const form = useGenerateForm();
    form.state.value.expand.enabled = false;
    const wire = form.toRequest();
    expect(wire.expand).toBeUndefined();
  });

  it("toRequest maps empty negativePrompt to null so server skips CFG", () => {
    const form = useGenerateForm();
    form.state.value.negativePrompt = "";
    expect(form.toRequest().negative_prompt).toBeNull();
  });

  it("toRequest omits stale scheduler and negative prompt for families that ignore them", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "flux-dev:q4",
      modelFamily: "flux",
      negativePrompt: "blurry",
      scheduler: "ddim",
    });

    const wire = form.toRequest();
    expect(wire.negative_prompt).toBeNull();
    expect(wire.scheduler).toBeUndefined();
  });

  it("preserves a distilled LTX negative in form state but omits it from the request", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "hf:opaque/distilled-checkpoint",
      modelFamily: "ltx2",
      pipeline: null,
      negativePrompt: "flicker",
      guidance: 7,
      guidanceCapabilities: {
        adjustable: false,
        supports_negative_prompt: false,
        fixed_scale: 1,
      },
    });
    expect(form.toRequest().guidance).toBe(1);
    expect(form.toRequest().negative_prompt).toBeNull();
    expect(form.state.value.negativePrompt).toBe("flicker");
    expect(form.state.value.guidance).toBe(7);
  });

  it("family-capability helpers match the documented allow-lists", () => {
    const form = useGenerateForm();
    // Video families.
    expect(form.isVideoFamily("ltx-video")).toBe(true);
    expect(form.isVideoFamily("ltx2")).toBe(true);
    expect(form.isVideoFamily("flux")).toBe(false);

    // CFG (negative prompt) support — flow-matching families skip CFG.
    expect(form.supportsNegativePrompt("sdxl")).toBe(true);
    expect(form.supportsNegativePrompt("sd15")).toBe(true);
    expect(form.supportsNegativePrompt("flux")).toBe(false);
    expect(form.supportsNegativePrompt("z-image")).toBe(false);

    // Scheduler override — UNet families only.
    expect(form.supportsScheduler("sdxl")).toBe(true);
    expect(form.supportsScheduler("sd15")).toBe(true);
    expect(form.supportsScheduler("flux")).toBe(false);

    // LoRA picker visibility — mirrors the server-side validation gate.
    for (const family of [
      "flux",
      "flux2",
      "ltx2",
      "sd15",
      "sd3",
      "sdxl",
      "qwen-image",
      "qwen-image-edit",
      "z-image",
    ]) {
      expect(form.supportsLora(family)).toBe(true);
    }
    expect(form.supportsLora("wuerstchen")).toBe(false);
  });

  it("reset() restores defaults", () => {
    const form = useGenerateForm();
    form.state.value.prompt = "dirty";
    form.state.value.steps = 99;
    form.reset();
    expect(form.state.value.prompt).toBe("");
    expect(form.state.value.steps).toBe(20);
  });
});

describe("useGenerateForm placement", () => {
  it("carries placement into the outgoing request wire", () => {
    const form = useGenerateForm();
    form.state.value.placement = {
      text_encoders: { kind: "cpu" },
      advanced: {
        transformer: { kind: "gpu", ordinal: 1 },
        vae: { kind: "auto" },
        t5: { kind: "cpu" },
      },
    };
    const wire = form.toRequest();
    expect(wire.placement).toEqual({
      text_encoders: { kind: "cpu" },
      advanced: {
        transformer: { kind: "gpu", ordinal: 1 },
        vae: { kind: "auto" },
        t5: { kind: "cpu" },
      },
    });
  });

  it("omits placement from the request when null", () => {
    const form = useGenerateForm();
    form.state.value.placement = null;
    const wire = form.toRequest();
    expect(wire.placement).toBeUndefined();
  });
});

describe("useGenerateForm — enableAudio (LTX-2 / LTX-2.3)", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("defaults enableAudio to null on hydrate so the wire stays clean for image families", () => {
    const form = useGenerateForm();
    expect(form.state.value.enableAudio).toBeNull();
    expect(form.toRequest().enable_audio).toBeUndefined();
  });

  it("auto-enables audio when an LTX-2 model is selected so the AV path is on by default", () => {
    // Mirrors the server-side `family_supports_audio("ltx2")` truth table.
    // Switching to LTX-2 should turn audio on so the user gets the AV
    // capability they expect from the model — they can still uncheck it.
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));
    expect(form.state.value.enableAudio).toBe(true);
    expect(form.toRequest().enable_audio).toBe(true);
  });

  it("keeps audio off for an LTX-2 catalog checkpoint whose assets are video-only", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(
      makeModel({
        name: "cv:3143864",
        family: "ltx2",
        supports_audio: false,
      }),
    );
    expect(form.state.value.enableAudio).toBe(false);
    expect(form.toRequest().enable_audio).toBe(false);
  });

  it("forces audio off at request time when persisted form state is stale", () => {
    const form = useGenerateForm();
    form.state.value.model = "cv:3143864";
    form.state.value.modelFamily = "ltx2";
    form.state.value.enableAudio = true;

    const wire = form.toRequest(
      makeModel({
        name: "cv:3143864",
        family: "ltx2",
        supports_audio: false,
      }),
    );

    expect(wire.enable_audio).toBe(false);
  });

  it("clears enableAudio back to null when switching from an AV family to an image family", () => {
    // The server rejects `enable_audio: true` for non-AV families; the
    // form must drop the toggle on family change so the user doesn't get
    // a 400 they didn't ask for.
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));
    expect(form.state.value.enableAudio).toBe(true);
    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    expect(form.state.value.enableAudio).toBeNull();
    expect(form.toRequest().enable_audio).toBeUndefined();
  });

  it("clears enableAudio when switching to LTX-Video (video but no audio path)", () => {
    // LTX-Video is a video family but has no audio decode path. The toggle
    // must NOT auto-on here even though the family is video — only LTX-2 /
    // LTX-2.3 (`family === "ltx2"`) advertises audio support.
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "ltx2:fp8", family: "ltx2" }));
    form.applyModelDefaults(
      makeModel({ name: "ltx-video:fp16", family: "ltx-video" }),
    );
    expect(form.state.value.enableAudio).toBeNull();
  });

  it("forwards a user-set enableAudio onto the wire as enable_audio", () => {
    const form = useGenerateForm();
    form.state.value.enableAudio = false; // user explicitly disabled audio
    expect(form.toRequest().enable_audio).toBe(false);
    form.state.value.enableAudio = true;
    expect(form.toRequest().enable_audio).toBe(true);
  });

  it("omits every wan recipe field while the controls are untouched", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "wan22-t2v-a14b:q5",
      modelFamily: "wan",
    });

    const wire = form.toRequest();
    expect("sample_shift" in wire).toBe(false);
    expect("distill_strength_high" in wire).toBe(false);
    expect("distill_strength_low" in wire).toBe(false);
    expect(wire.scheduler).toBeNull();
  });

  it("sends the wan recipe fields the user touched", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "wan22-t2v-a14b:q5",
      modelFamily: "wan",
      scheduler: "dpm-pp",
      wanRecipe: {
        sampleShift: 12,
        distillStrengthHigh: 1.8,
        distillStrengthLow: null,
      },
    });

    const wire = form.toRequest();
    expect(wire).toMatchObject({
      scheduler: "dpm-pp",
      sample_shift: 12,
      distill_strength_high: 1.8,
    });
    expect("distill_strength_low" in wire).toBe(false);
  });

  it("never ships wan recipe fields for another family", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "ltx-2-19b-dev:fp8",
      modelFamily: "ltx2",
      wanRecipe: {
        sampleShift: 12,
        distillStrengthHigh: 1.8,
        distillStrengthLow: 1,
      },
    });

    const wire = form.toRequest();
    expect("sample_shift" in wire).toBe(false);
    expect("distill_strength_high" in wire).toBe(false);
  });

  it("drops distill strengths for a wan tier that ships no distill", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "wan22-t2v-a14b:q8",
      modelFamily: "wan",
      wanRecipe: {
        sampleShift: 8,
        distillStrengthHigh: 1.8,
        distillStrengthLow: 1,
      },
    });

    const wire = form.toRequest();
    expect(wire.sample_shift).toBe(8);
    expect("distill_strength_high" in wire).toBe(false);
    expect("distill_strength_low" in wire).toBe(false);
  });

  it("clears the wan recipe when the selected model leaves the family", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "wan22-t2v-a14b:q5",
      modelFamily: "wan",
      wanRecipe: {
        sampleShift: 12,
        distillStrengthHigh: 1.8,
        distillStrengthLow: 1,
      },
    });

    form.applyModelDefaults(makeModel({ name: "sdxl:fp16", family: "sdxl" }));
    expect(form.state.value.wanRecipe).toEqual({
      sampleShift: null,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    });
  });

  it("keeps the shift but drops the strengths when moving to a tier without a distill", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "wan22-t2v-a14b:q5",
      modelFamily: "wan",
      wanRecipe: {
        sampleShift: 12,
        distillStrengthHigh: 1.8,
        distillStrengthLow: 1,
      },
    });

    form.applyModelDefaults(
      makeModel({ name: "wan22-t2v-a14b:q8", family: "wan" }),
    );
    expect(form.state.value.wanRecipe).toEqual({
      sampleShift: 12,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    });
  });
});

/**
 * Per-model source-image conditioning (#772) and the wan first/last-frame
 * layout that rides the existing `keyframes` field (#779).
 */
describe("useGenerateForm — source image & first/last frames", () => {
  beforeEach(() => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: new IDBFactory(),
      writable: true,
      configurable: true,
    });
    localStorage.clear();
    __testing__.resetForTest();
  });

  const wanEndFrameModel = (
    overrides: Partial<ModelInfoExtended> = {},
  ): ModelInfoExtended =>
    makeModel({
      name: "wan22-ti2v-5b:fp16",
      family: "wan",
      default_frames: 81,
      default_fps: 24,
      source_image: "optional",
      ...overrides,
    });

  function attachEnds(form: ReturnType<typeof useGenerateForm>) {
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "open.png", base64: "FIRST" },
    ];
    form.state.value.endFrame = {
      kind: "upload",
      filename: "close.png",
      base64: "LAST",
    };
  }

  it("defaults both new fields to absent so nothing new reaches an older server", () => {
    const form = useGenerateForm();
    expect(form.state.value.sourceImageCapability).toBeNull();
    expect(form.state.value.endFrame).toBeNull();
  });

  it("snapshots the model's advertised contract on model change", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel({ source_image: "required" }));
    expect(form.state.value.sourceImageCapability).toBe("required");
    // A row without the field clears the snapshot rather than inheriting the
    // previous model's answer.
    form.applyModelDefaults(
      makeModel({ name: "flux-dev:q4", family: "flux", source_image: null }),
    );
    expect(form.state.value.sourceImageCapability).toBeNull();
  });

  it("retains staged wells across a text-to-video-only switch but ships neither", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel());
    attachEnds(form);

    // Still a wan checkpoint, but text-to-video only: retention keeps the
    // authored media staged while the wire prune keeps it off the request.
    form.applyModelDefaults(
      wanEndFrameModel({
        name: "wan22-t2v-a14b:q5",
        source_image: "unsupported",
      }),
    );
    expect(form.state.value.endFrame?.base64).toBe("LAST");
    expect(form.state.value.imageAttachments).toHaveLength(1);
    const req = form.toRequest();
    expect(req.source_image ?? null).toBeNull();
    expect(req.keyframes).toBeUndefined();
  });

  it("keeps the source and parks the end frame on a family with no first/last layout", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel());
    attachEnds(form);

    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    expect(form.state.value.endFrame?.base64).toBe("LAST");
    expect(form.state.value.imageAttachments).toHaveLength(1);
    const req = form.toRequest();
    expect(req.keyframes).toBeUndefined();
    expect(req.source_image).toBe("FIRST");
  });

  it("bridges the staged source into the H3 first frame and back", () => {
    const h3Model = makeModel({
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
      source_image: "required",
      default_frames: 124,
      default_fps: 24,
    });
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    form.state.value.imageAttachments = [
      {
        kind: "upload",
        filename: "pic.png",
        base64: "QUJD",
        width: 1024,
        height: 576,
        mime: "image/png",
      },
    ];

    form.applyModelDefaults(h3Model);
    expect(form.state.value.h3Authoring?.firstFrame).toMatchObject({
      filename: "pic.png",
      data: "QUJD",
      width: 1024,
      height: 576,
    });
    expect(form.state.value.imageAttachments).toEqual([]);

    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    expect(form.state.value.imageAttachments[0]).toMatchObject({
      filename: "pic.png",
      base64: "QUJD",
    });
    expect(form.state.value.h3Authoring?.firstFrame ?? null).toBeNull();
  });

  it("never seeds a last frame into a first-frame-only H3 checkpoint", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel());
    attachEnds(form);
    form.applyModelDefaults(
      makeModel({
        name: "minimax-h3-fl2va:official-bf16",
        family: "minimax-h3",
        source_image: "required",
      }),
    );
    expect(form.state.value.h3Authoring?.firstFrame?.data).toBe("FIRST");
    expect(form.state.value.h3Authoring?.lastFrame ?? null).toBeNull();
    // The closing frame stays parked for the next first/last-capable model.
    expect(form.state.value.endFrame?.base64).toBe("LAST");
  });

  it("keeps a bytes-less reattach descriptor out of the source well when leaving H3", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(
      makeModel({
        name: "minimax-h3-fl2va:official-bf16",
        family: "minimax-h3",
        source_image: "required",
      }),
    );
    form.state.value.h3Authoring = {
      firstFrame: {
        filename: "provenance.png",
        mimeType: "image/*",
        width: 0,
        height: 0,
        data: "",
        sha256: "a".repeat(64),
      },
      lastFrame: null,
      references: [],
    };
    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    expect(form.state.value.imageAttachments).toEqual([]);
    // The descriptor survives for a later H3 return.
    expect(form.state.value.h3Authoring?.firstFrame?.filename).toBe(
      "provenance.png",
    );
  });

  it("retains LTX-2 staged media across a switch away and back", () => {
    const ltx2 = makeModel({ name: "ltx-2-19b-distilled:fp8", family: "ltx2" });
    const form = useGenerateForm();
    form.applyModelDefaults(ltx2);
    form.state.value.sourceVideo = {
      kind: "upload",
      filename: "clip.mp4",
      base64: "VklE",
    };
    form.state.value.audioFile = {
      kind: "upload",
      filename: "voice.wav",
      base64: "QVVE",
    };
    form.state.value.keyframes = [
      {
        frame: 9,
        image: { kind: "upload", filename: "k.png", base64: "S0VZ" },
      },
    ];

    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    expect(form.state.value.sourceVideo?.filename).toBe("clip.mp4");
    expect(form.state.value.audioFile?.filename).toBe("voice.wav");
    expect(form.state.value.keyframes).toHaveLength(1);
    // Settings knobs still clear with the suite, and nothing ships.
    expect(form.state.value.pipeline).toBeNull();
    const req = form.toRequest();
    expect(req.source_video).toBeUndefined();
    expect(req.audio_file).toBeUndefined();
    expect(req.keyframes).toBeUndefined();

    form.applyModelDefaults(ltx2);
    expect(form.state.value.sourceVideo?.filename).toBe("clip.mp4");
  });

  it("snaps stale values onto fixed recipe controls instead of stranding Generate", () => {
    // A distilled LTX recipe fixes guidance at 1. A form carrying an older
    // model's 3.5 must normalize on switch (shared with desktop) — the
    // disabled slider displays 1, so 3.5 was never user authority.
    const distilled = makeModel({
      name: "ltx-2.3-22b-distilled:fp8",
      family: "ltx2",
      default_guidance: 1,
      default_steps: 8,
      generation_profile: {
        schema_version: 1,
        profile_id: "ltx2.v1",
        profile_hash: "hash",
        default_recipe_id: "distilled",
        recipes: [
          {
            id: "distilled",
            label: "Distilled",
            request_selector: {},
            defaults: { width: 1216, height: 704, steps: 8, guidance: 1 },
            resolution: {
              domain: "dynamic",
              alignment: 32,
              min_width: 64,
              min_height: 64,
              max_pixels: 4_000_000,
              aspect_groups: [
                {
                  id: "19:11",
                  label: "19:11",
                  presets: [
                    {
                      id: "1216x704",
                      width: 1216,
                      height: 704,
                      tier: "recommended",
                    },
                  ],
                },
              ],
            },
            steps: {
              default: 8,
              min: 1,
              max: 100,
              step: 1,
              mode: "adjustable",
            },
            guidance: { default: 1, min: 1, max: 1, step: 0.1, mode: "fixed" },
            capabilities: {
              guidance: {
                adjustable: false,
                supports_negative_prompt: false,
                fixed_scale: 1,
              },
              negative_prompt: { mode: "hidden", required: false },
              supports_lora: true,
              supports_controlnet: false,
              supports_sequence: true,
              supports_extend: true,
              supports_audio: true,
              source_video: { mode: "adjustable", required: false },
              mask: { mode: "hidden", required: false },
              keyframes: { mode: "adjustable", required: false },
              audio: { mode: "adjustable", required: false },
              lora: { mode: "adjustable", max_count: 4 },
              controlnet: { mode: "hidden", max_count: 0 },
              output: {
                default_format: "mp4",
                formats: ["mp4"],
                audio_requires_mp4: false,
              },
              wan_recipe: {
                mode: "hidden",
                supports_distill_strength: false,
                supports_first_last_frame: false,
              },
              schedulers: [],
            },
            provenance: [],
          },
        ],
      },
    } as never);
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    form.state.value.guidance = 3.5;
    form.applyModelDefaults(distilled);
    expect(form.state.value.guidance).toBe(1);
  });

  it("ships and restores the source-fit crop provenance", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "pic.png", base64: "QUJD" },
    ];
    form.state.value.sourceFitPolicy = { mode: "crop-fill", alignX: "left" };
    expect(form.toRequest().source_fit).toEqual({
      mode: "crop-fill",
      alignX: "left",
    });

    // No staged media → no provenance field at all.
    form.state.value.imageAttachments = [];
    expect(form.toRequest().source_fit).toBeUndefined();

    // Restore round-trips through metadata; malformed values are ignored.
    const restored = applyMetadataToForm(form.state.value, {
      prompt: "p",
      model: "flux-dev:q4",
      seed: 1,
      steps: 4,
      guidance: 3.5,
      width: 1024,
      height: 1024,
      source_fit: { mode: "lanczos-resize" },
    } as never);
    expect(restored.sourceFitPolicy).toEqual({ mode: "lanczos-resize" });
    const poisoned = applyMetadataToForm(form.state.value, {
      prompt: "p",
      model: "flux-dev:q4",
      seed: 1,
      steps: 4,
      guidance: 3.5,
      width: 1024,
      height: 1024,
      source_fit: { mode: "teleport" },
    } as never);
    expect(poisoned.sourceFitPolicy).toEqual(form.state.value.sourceFitPolicy);
  });

  it("gates retained media off the wire for a model with no source support", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(makeModel({ name: "flux-dev:q4", family: "flux" }));
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "pic.png", base64: "QUJD" },
    ];
    form.state.value.maskImage = {
      kind: "upload",
      filename: "mask.png",
      base64: "TUFTSw==",
    };
    form.applyModelDefaults(
      makeModel({ name: "ltx-video:q8", family: "ltx-video" }),
    );
    expect(form.state.value.imageAttachments).toHaveLength(1);
    expect(form.state.value.maskImage?.base64).toBe("TUFTSw==");
    const req = form.toRequest();
    expect(req.source_image ?? null).toBeNull();
    expect(req.mask_image).toBeUndefined();
  });

  it("persists the end frame as a descriptor and hydrates its bytes from the draft store", async () => {
    vi.useFakeTimers();
    try {
      const first = useGenerateForm();
      first.state.value.endFrame = {
        kind: "upload",
        filename: "close.png",
        base64: "END_FRAME_BYTES",
      };
      await nextTick();
      await flushDraftPersistence();

      const raw = localStorage.getItem(STORAGE_KEY);
      expect(raw).not.toContain("END_FRAME_BYTES");
      expect(JSON.parse(raw!).endFrame).toMatchObject({
        filename: "close.png",
        draftId: expect.any(String),
      });

      __testing__.clearDraftsForTest();
      __testing__.resetForTest();
      const second = useGenerateForm();
      await flushDraftHydration();
      expect(second.state.value.endFrame?.base64).toBe("END_FRAME_BYTES");
    } finally {
      vi.useRealTimers();
    }
  });

  it("serializes both ends as exactly two keyframes and keeps source_image home", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel());
    attachEnds(form);
    form.state.value.frames = 81;

    const wire = form.toRequest();
    // The engine refuses a request carrying both `source_image` and
    // `keyframes` ("not both"); admission counts keyframes as source
    // presence, so the pair travels only once.
    expect(wire.source_image).toBeNull();
    expect(wire.source_image_name).toBeUndefined();
    expect(wire.strength).toBeUndefined();
    expect(wire.keyframes).toEqual([
      { frame: 0, image: "FIRST", name: "open.png" },
      { frame: 80, image: "LAST", name: "close.png" },
    ]);
  });

  it("recomputes the closing index from the frame count held at submit time", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel());
    attachEnds(form);
    form.state.value.frames = 81;
    expect(form.toRequest().keyframes?.[1]?.frame).toBe(80);

    form.state.value.frames = 49;
    expect(form.toRequest().keyframes?.[1]?.frame).toBe(48);
  });

  it("sends no keyframes for a lone source image — that request is unchanged", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel());
    form.state.value.imageAttachments = [
      { kind: "upload", filename: "open.png", base64: "FIRST" },
    ];
    form.state.value.frames = 81;

    const wire = form.toRequest();
    expect(wire.source_image).toBe("FIRST");
    expect(wire.keyframes).toBeUndefined();
  });

  it("sends no keyframes for an end frame with no first frame", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel());
    form.state.value.endFrame = {
      kind: "upload",
      filename: "close.png",
      base64: "LAST",
    };
    form.state.value.frames = 81;
    expect(form.toRequest().keyframes).toBeUndefined();
  });

  it("sends no keyframes when the server advertised nothing for this checkpoint", () => {
    // A host old enough to omit `source_image` rejects wan keyframes outright,
    // so the pair must not be optimistically serialized.
    const form = useGenerateForm();
    form.applyModelDefaults(
      wanEndFrameModel({ source_image: null }) as ModelInfoExtended,
    );
    attachEnds(form);
    form.state.value.frames = 81;
    expect(form.toRequest().keyframes).toBeUndefined();
  });

  it("leaves the LTX-2 keyframe control alone", () => {
    const form = useGenerateForm();
    Object.assign(form.state.value, {
      model: "ltx-2-19b:fp8",
      modelFamily: "ltx2",
      sourceImageCapability: "optional",
      frames: 97,
      keyframes: [
        {
          frame: 48,
          image: { kind: "upload", filename: "kf.png", base64: "KF" },
        },
      ],
      endFrame: { kind: "upload", filename: "close.png", base64: "LAST" },
      imageAttachments: [
        { kind: "upload", filename: "open.png", base64: "FIRST" },
      ],
    });
    expect(form.toRequest().keyframes).toEqual([
      { frame: 48, image: "KF", name: "kf.png" },
    ]);
  });

  it("applyMetadataToForm clears the end frame — saved keyframes carry no bytes", () => {
    const form = useGenerateForm();
    form.applyModelDefaults(wanEndFrameModel());
    attachEnds(form);

    const restored = applyMetadataToForm(
      form.state.value,
      {
        prompt: "a heron",
        model: "wan22-ti2v-5b:fp16",
        seed: 7,
        steps: 20,
        guidance: 3.5,
        width: 1280,
        height: 704,
        version: "test",
        keyframes: [
          { frame: 0, name: "open.png", sha256: "a".repeat(64) },
          { frame: 80, name: "close.png", sha256: "b".repeat(64) },
        ],
      } as OutputMetadata,
      { models: [wanEndFrameModel()] },
    );
    expect(restored.endFrame).toBeNull();
  });
});

describe("generate form serialization helpers", () => {
  function makeForm(
    overrides: Partial<GenerateFormState> = {},
  ): GenerateFormState {
    return {
      version: 3,
      stylePreset: null,
      prompt: "a cat",
      negativePrompt: "",
      model: "flux-dev:q4",
      modelFamily: "flux",
      width: 1024,
      height: 1024,
      steps: 28,
      guidance: 3.5,
      sourceImageCapability: null,
      endFrame: null,
      seedMode: "random",
      seed: null,
      batchSize: 1,
      strength: 0.75,
      frames: null,
      fps: null,
      scheduler: null,
      cfgPlus: false,
      outputFormat: "png",
      expand: { enabled: false, variations: 1, familyOverride: null },
      imageAttachments: [],
      maskImage: null,
      controlImage: null,
      controlModel: "",
      controlScale: 1,
      upscaleModel: "",
      gifPreview: false,
      audioFile: null,
      audioFilePath: "",
      sourceVideo: null,
      extendVideo: null,
      extendVideoPath: "",
      extendOverlapFrames: null,
      sourceVideoPath: "",
      keyframes: [],
      pipeline: null,
      retakeRange: null,
      spatialUpscale: null,
      temporalUpscale: null,
      cameraControl: null,
      placement: null,
      loras: [],
      enableAudio: null,
      sourceFitPolicy: { mode: "pad-repaint" },
      ...overrides,
    };
  }

  it("sanitizePersistedForm strips binary media while preserving safe path references", () => {
    const form = makeForm({
      imageAttachments: [
        { kind: "gallery", filename: "source.png", base64: "SOURCE_BYTES" },
      ],
      maskImage: { kind: "upload", filename: "mask.png", base64: "MASK_BYTES" },
      controlImage: {
        kind: "upload",
        filename: "control.png",
        base64: "CONTROL_BYTES",
      },
      audioFile: {
        kind: "upload",
        filename: "voice.wav",
        base64: "VOICE_BYTES",
      },
      audioFilePath: "/srv/voice.wav",
      sourceVideo: {
        kind: "upload",
        filename: "clip.mp4",
        base64: "VIDEO_BYTES",
      },
      sourceVideoPath: "/srv/clip.mp4",
      keyframes: [
        {
          frame: 24,
          image: { kind: "upload", filename: "kf.png", base64: "KF_BYTES" },
        },
      ],
    });

    const sanitized = sanitizePersistedForm(form);

    expect(sanitized.imageAttachments).toEqual([
      { kind: "gallery", filename: "source.png" },
    ]);
    expect(sanitized.maskImage).toEqual({
      kind: "upload",
      filename: "mask.png",
    });
    expect(sanitized.controlImage).toEqual({
      kind: "upload",
      filename: "control.png",
    });
    expect(sanitized.audioFile).toEqual({
      kind: "upload",
      filename: "voice.wav",
    });
    expect(sanitized.sourceVideo).toEqual({
      kind: "upload",
      filename: "clip.mp4",
    });
    expect(sanitized.keyframes).toEqual([
      { frame: 24, image: { kind: "upload", filename: "kf.png" } },
    ]);
    expect(sanitized.audioFilePath).toBe("/srv/voice.wav");
    expect(sanitized.sourceVideoPath).toBe("/srv/clip.mp4");
    expect(JSON.stringify(sanitized)).not.toContain("_BYTES");
  });

  it("cloneTemplateForm preserves generation config including static seed and ordered LoRAs", () => {
    const form = makeForm({
      seedMode: "static",
      seed: 777,
      scheduler: "ddim",
      loras: [
        { path: "/loras/a.safetensors", scale: 0.5, trainedWords: ["a"] },
        { path: "/loras/b.safetensors", scale: 1.2, trainedWords: ["b"] },
      ],
      placement: { text_encoders: { kind: "cpu" } },
    });

    const cloned = cloneTemplateForm(form);
    cloned.loras[0].scale = 2;

    expect(cloned.seedMode).toBe("static");
    expect(cloned.seed).toBe(777);
    expect(cloned.scheduler).toBe("ddim");
    expect(cloned.loras).toEqual([
      { path: "/loras/a.safetensors", scale: 2, trainedWords: ["a"] },
      { path: "/loras/b.safetensors", scale: 1.2, trainedWords: ["b"] },
    ]);
    expect(form.loras[0].scale).toBe(0.5);
  });

  it("applyMetadataToForm restores recreate-safe metadata without carrying stale binary inputs", () => {
    const current = makeForm({
      prompt: "old prompt",
      imageAttachments: [
        { kind: "upload", filename: "old.png", base64: "OLD_BYTES" },
      ],
    });
    const metadata: OutputMetadata = {
      prompt: "new prompt",
      negative_prompt: "blurry",
      model: "sdxl:fp16",
      seed: 42,
      steps: 30,
      guidance: 7.5,
      width: 768,
      height: 512,
      generation_width: 192,
      generation_height: 128,
      strength: 0.6,
      scheduler: "ddim",
      output_format: "jpeg",
      loras: [
        { path: "/loras/one.safetensors", scale: 0.8 },
        { path: "/loras/two.safetensors", scale: 1.1 },
      ],
      control_model: "controlnet-canny-sd15",
      control_scale: 0.7,
      upscale_model: "real-esrgan-x4plus:fp16",
      gif_preview: true,
      version: "0.12.0",
    };

    const next = applyMetadataToForm(current, metadata, {
      format: "png",
      models: [makeModel({ name: "sdxl:fp16", family: "sdxl" })],
    });

    expect(next.prompt).toBe("new prompt");
    expect(next.negativePrompt).toBe("blurry");
    expect(next.model).toBe("sdxl:fp16");
    expect(next.modelFamily).toBe("sdxl");
    expect(next.seedMode).toBe("static");
    expect(next.seed).toBe(42);
    expect(next.outputFormat).toBe("jpeg");
    expect(next.width).toBe(192);
    expect(next.height).toBe(128);
    expect(next.imageAttachments).toEqual([]);
    expect(next.loras).toEqual([
      { path: "/loras/one.safetensors", scale: 0.8 },
      { path: "/loras/two.safetensors", scale: 1.1 },
    ]);
  });

  it("applyMetadataToForm restores the wan recipe a print was rendered with", () => {
    const next = applyMetadataToForm(
      makeForm(),
      {
        prompt: "a courtyard at dusk",
        model: "wan22-t2v-a14b:q5",
        seed: 7,
        steps: 4,
        guidance: 1,
        width: 832,
        height: 480,
        scheduler: "euler",
        sample_shift: 12,
        distill_strength_high: 1.8,
        distill_strength_low: 0.9,
        version: "0.12.0",
      },
      { models: [makeModel({ name: "wan22-t2v-a14b:q5", family: "wan" })] },
    );

    expect(next.scheduler).toBe("euler");
    expect(next.wanRecipe).toEqual({
      sampleShift: 12,
      distillStrengthHigh: 1.8,
      distillStrengthLow: 0.9,
    });
  });

  it("applyMetadataToForm leaves the wan recipe untouched for a print without one", () => {
    const next = applyMetadataToForm(
      makeForm(),
      {
        prompt: "a courtyard at dusk",
        model: "wan22-t2v-a14b:q5",
        seed: 7,
        steps: 4,
        guidance: 1,
        width: 832,
        height: 480,
        version: "0.12.0",
      },
      { models: [makeModel({ name: "wan22-t2v-a14b:q5", family: "wan" })] },
    );

    expect(next.wanRecipe).toEqual({
      sampleShift: null,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    });
  });

  it("applyMetadataToForm preserves H3 boundary provenance without bytes", () => {
    const next = applyMetadataToForm(makeForm(), {
      prompt: "new synchronized shot",
      model: "minimax-h3-fl2va:official-bf16",
      seed: 42,
      steps: 30,
      guidance: 0,
      width: 768,
      height: 512,
      output_format: "mp4",
      source_image_name: "opening.png",
      source_image_sha256: "a".repeat(64),
      frames: 124,
      keyframes: [{ frame: 123, name: "closing.png", sha256: "b".repeat(64) }],
      version: "0.1.0",
    });
    expect(next.h3Authoring?.firstFrame).toMatchObject({
      filename: "opening.png",
      data: "",
      sha256: "a".repeat(64),
    });
    expect(next.h3Authoring?.lastFrame).toMatchObject({
      filename: "closing.png",
      data: "",
      sha256: "b".repeat(64),
    });
  });

  it("does not infer camera motion from a metadata LoRA beyond the visible cap", () => {
    const current = makeForm();
    const next = applyMetadataToForm(
      current,
      {
        prompt: "tracking shot",
        model: "ltx-2-19b-distilled:fp8",
        seed: 7,
        steps: 8,
        guidance: 3,
        width: 768,
        height: 512,
        version: "0.1.0",
        loras: [
          { path: "one", scale: 1 },
          { path: "two", scale: 1 },
          { path: "three", scale: 1 },
          { path: "four", scale: 1 },
          { path: "camera-control:dolly-in", scale: 0.45 },
        ],
      },
      {
        models: [
          makeModel({
            name: "ltx-2-19b-distilled:fp8",
            family: "ltx2",
          }),
        ],
      },
    );

    expect(next.loras.map((lora) => lora.path)).toEqual([
      "one",
      "two",
      "three",
      "four",
    ]);
    expect(next.cameraControl).toBeNull();
  });
});
