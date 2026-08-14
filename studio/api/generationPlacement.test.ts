import { describe, expect, test, vi } from "vitest";
import {
  comparePlacementPreviews,
  classifyPlacementPreview,
  previewChainPlacement,
  previewGenerationPlacement,
  previewRequestForSiblingFanout,
  redactChainForPlacement,
  redactGenerationForPlacement,
  requiresAuthoritativePlacement,
  type GenerationPlacementPreview,
} from "./generationPlacement";

const apiJsonTo = vi.hoisted(() => vi.fn().mockResolvedValue({}));
vi.mock("./client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./client")>()),
  apiJsonTo,
}));

function planned(
  completion: number,
  setup: number,
): GenerationPlacementPreview {
  return {
    version: 1,
    authoritative: true,
    state_version: 1,
    plan_version: 1,
    outcome: "planned",
    candidate: {
      device_id: "cuda:stable",
      execution_fingerprint: "exec",
      predicted_start_after_ms: 0,
      predicted_completion_after_ms: completion,
      setup_ms: setup,
      setup_kind: "cold",
      estimate_confidence: "medium",
    },
  };
}

describe("generation placement preview", () => {
  test("requires authority whenever the references field is present", () => {
    expect(requiresAuthoritativePlacement({ model: "ordinary" })).toBe(false);
    expect(
      requiresAuthoritativePlacement({ model: "h3", references: null }),
    ).toBe(true);
    expect(
      requiresAuthoritativePlacement({ model: "h3", references: [] }),
    ).toBe(true);
  });

  test("previews sibling fanout as copies of a one-image request without mutating input", () => {
    const request = { model: "preview", batch_size: 4, prompt: "reviewed" };
    expect(previewRequestForSiblingFanout(request, 4)).toEqual({
      model: "preview",
      batch_size: 1,
      prompt: "reviewed",
    });
    expect(request.batch_size).toBe(4);
    expect(previewRequestForSiblingFanout(request, 1)).toBe(request);
  });

  test("redacts prompt and media bytes while preserving structural presence", () => {
    const result = redactGenerationForPlacement({
      prompt: "secret prompt",
      negative_prompt: "secret negative",
      model: "cv:123",
      source_image_name: "/private/source.png",
      source_image: "image-bytes",
      mask_image: "mask-bytes",
      control_image: "control-bytes",
      audio_file_path: "/private/audio.wav",
      audio_file: "audio-bytes",
      source_video_path: "/private/source.mp4",
      source_video: "video-bytes",
      edit_images: ["one", "two"],
      references: [
        {
          kind: "image",
          media: { authority: "inline", data: "reference-bytes" },
          provenance: { name: "anchor.png", sha256: "a".repeat(64) },
          mime_type: "image/png",
          width: 2048,
          height: 1024,
        },
        {
          kind: "video",
          media: { authority: "upload", handle: "secret-handle" },
          provenance: { name: "motion.mp4", sha256: "b".repeat(64) },
          mime_type: "video/mp4",
          width: 1920,
          height: 1080,
          duration_ms: 4000,
          fps: 24,
        },
        {
          kind: "audio",
          media: { authority: "server_path", path: "/private/voice.wav" },
          provenance: { name: "voice.wav", sha256: "c".repeat(64) },
          mime_type: "audio/wav",
          duration_ms: 3000,
          sample_rate: 32000,
          channels: 2,
        },
      ],
      keyframes: [{ frame: 0, image: "keyframe-bytes", strength: 0.5 }],
      lora: { path: "/models/legacy.safetensors", scale: 0.7 },
      loras: [{ path: "/models/style.safetensors", scale: 1.1 }],
      width: 1024,
    });
    expect(result).toEqual({
      prompt: "",
      negative_prompt: "",
      model: "cv:123",
      source_image_name: "",
      source_image: "",
      mask_image: "",
      control_image: "",
      audio_file_path: "",
      audio_file: "",
      source_video_path: "",
      source_video: "",
      edit_images: ["", ""],
      references: [
        {
          kind: "image",
          media: { authority: "descriptor" },
          provenance: { name: "anchor.png", sha256: "a".repeat(64) },
          mime_type: "image/png",
          width: 2048,
          height: 1024,
        },
        {
          kind: "video",
          media: { authority: "descriptor" },
          provenance: { name: "motion.mp4", sha256: "b".repeat(64) },
          mime_type: "video/mp4",
          width: 1920,
          height: 1080,
          duration_ms: 4000,
          fps: 24,
        },
        {
          kind: "audio",
          media: { authority: "descriptor" },
          provenance: { name: "voice.wav", sha256: "c".repeat(64) },
          mime_type: "audio/wav",
          duration_ms: 3000,
          sample_rate: 32000,
          channels: 2,
        },
      ],
      keyframes: [{ frame: 0, image: "", strength: 0.5 }],
      lora: { path: "/models/legacy.safetensors", scale: 0.7 },
      loras: [{ path: "/models/style.safetensors", scale: 1.1 }],
      width: 1024,
    });
  });

  test("ranks normalized completion then setup then stable host id", () => {
    const rows = [
      { hostId: "z", roundTripMs: 20, preview: planned(80, 1) },
      { hostId: "b", roundTripMs: 0, preview: planned(100, 3) },
      { hostId: "a", roundTripMs: 0, preview: planned(100, 3) },
      { hostId: "c", roundTripMs: 0, preview: planned(100, 2) },
    ];
    rows.sort(comparePlacementPreviews);
    expect(rows.map((row) => row.hostId)).toEqual(["z", "c", "a", "b"]);
  });

  test("prefers a clean plan over a faster plan with multi-gigabyte pending downloads", () => {
    const rows = [
      {
        hostId: "pending",
        roundTripMs: 0,
        preview: {
          ...planned(10, 0),
          pending_downloads: [
            {
              kind: "text_encoder",
              name: "t5-v1_1-xxl-q8.gguf",
              repo: "acme/text-encoders",
              bytes: 5_100_000_000,
            },
          ],
        },
      },
      {
        hostId: "clean",
        roundTripMs: 0,
        preview: planned(10_000, 500),
      },
    ];

    rows.sort(comparePlacementPreviews);

    expect(rows.map((row) => row.hostId)).toEqual(["clean", "pending"]);
  });

  test.each([
    [{}, "invalid"],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: true,
        outcome: "unsupported",
      },
      "invalid",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: true,
        outcome: "planned",
      },
      "invalid",
    ],
    [
      {
        ...planned(100, 1),
        candidate: {
          ...planned(100, 1).candidate,
          predicted_completion_after_ms: -1,
        },
      },
      "invalid",
    ],
    [
      {
        ...planned(100, 1),
        pending_downloads: [
          {
            kind: "text_encoder",
            name: "t5.gguf",
            repo: "acme/t5",
            bytes: -1,
          },
        ],
      },
      "invalid",
    ],
    [
      {
        ...planned(100, 1),
        candidate: {
          ...planned(100, 1).candidate,
          setup_ms: Number.NaN,
        },
      },
      "invalid",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: false,
        outcome: "unsupported",
        candidate: null,
      },
      "unsupported",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: false,
        outcome: "unsupported",
        stage_candidates: [{ stage_index: 0 }],
      },
      "invalid",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: true,
        outcome: "infeasible",
        reason: "required VAE is absent",
        candidate: null,
      },
      "infeasible",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: true,
        outcome: "infeasible",
        reason: "required VAE is absent",
        candidate: null,
        missing_components: [
          {
            kind: "vae",
            name: "ae.safetensors",
            present: false,
            repair_model: "flux-dev:bf16",
          },
        ],
      },
      "infeasible",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: false,
        outcome: "temporarily_unavailable",
        reason: "scheduler state changed",
        candidate: null,
      },
      "temporarily_unavailable",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: true,
        outcome: "temporarily_unavailable",
        reason: "scheduler state changed",
        candidate: null,
      },
      "temporarily_unavailable",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: true,
        outcome: "infeasible",
        reason: "",
        candidate: null,
      },
      "invalid",
    ],
    [
      {
        version: 1,
        state_version: 1,
        plan_version: 1,
        authoritative: true,
        outcome: "infeasible",
        reason: "missing component",
        candidate: null,
        missing_components: [{ kind: "vae", name: "ae", present: "no" }],
      },
      "invalid",
    ],
    [planned(100, 1), "planned"],
  ])("strictly classifies placement authority %#", (preview, expected) => {
    expect(classifyPlacementPreview(preview)).toBe(expected);
  });

  test("redacts every sequence prompt and source while preserving topology", () => {
    expect(
      redactChainForPlacement({
        model: "cv:123",
        original_prompt: "secret original",
        stages: [
          {
            prompt: "secret one",
            frames: 97,
            negative_prompt: "secret negative",
            source_image: "secret image",
          },
          { prompt: "secret two", frames: 121, transition: "cut" },
        ],
      }),
    ).toEqual({
      model: "cv:123",
      original_prompt: "",
      stages: [
        {
          prompt: "",
          frames: 97,
          negative_prompt: "",
          source_image: "",
        },
        { prompt: "", frames: 121, transition: "cut" },
      ],
    });
  });

  test("both preview probes carry a bounded timeout so a hung server can never leave planning stuck forever", async () => {
    apiJsonTo.mockClear();
    const target = { baseUrl: "http://plato:7680", apiKey: null };
    await previewGenerationPlacement(target, { model: "m", prompt: "p" });
    await previewChainPlacement(target, { model: "m", prompt: "p" });

    expect(apiJsonTo).toHaveBeenCalledTimes(2);
    for (const call of apiJsonTo.mock.calls) {
      const init = call[2] as RequestInit;
      expect(init.signal).toBeInstanceOf(AbortSignal);
    }
  });

  test("falls back to a timer-armed controller when AbortSignal.timeout is unavailable (Safari 15)", async () => {
    apiJsonTo.mockClear();
    vi.useFakeTimers();
    const timeoutFactory = AbortSignal.timeout;
    Object.defineProperty(AbortSignal, "timeout", {
      configurable: true,
      value: undefined,
    });
    expect(typeof AbortSignal.timeout).toBe("undefined");
    try {
      await previewGenerationPlacement(
        { baseUrl: "http://plato:7680", apiKey: null },
        { model: "m", prompt: "p" },
      );
      const init = apiJsonTo.mock.calls[0]![2] as RequestInit;
      const signal = init.signal as AbortSignal;
      expect(signal).toBeInstanceOf(AbortSignal);
      expect(signal.aborted).toBe(false);
      vi.advanceTimersByTime(900_000);
      expect(signal.aborted).toBe(true);
    } finally {
      Object.defineProperty(AbortSignal, "timeout", {
        configurable: true,
        value: timeoutFactory,
      });
      vi.useRealTimers();
    }
  });
});
