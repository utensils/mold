import { MAX_LORA_STACK } from "../lib/generationCapabilities";
import { apiFetchTo, apiJsonTo } from "./client";
import { assertChainOwner, type ChainOwner } from "./chainJobs";
import { retainedSourceMediaInventory } from "./gallerySourceMedia";
import type { ChainRequestWire, ChainStageWire } from "../lib/api/chainTypes";
import type {
  GenerationReference,
  GenerationReferenceMetadata,
} from "../lib/generationReferences";
import { parseSourceFitPolicy } from "../lib/sourceFit";
import { exactSeed } from "../lib/videoSettingsReuse";
import { mirrorMetadataMatches } from "../lib/galleryMirrorIdentity";
import { blobToBase64 } from "../lib/base64";
const MAX = 32 * 1024 * 1024;
const digest = (value: unknown): value is string =>
  typeof value === "string" && /^[0-9a-f]{64}$/i.test(value);
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error(
      "This video did not save complete clip settings. Use Start again from its saved job.",
    );
  return value as Record<string, unknown>;
}
function whole(value: unknown, allowZero = false): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < (allowZero ? 0 : 1) ||
    value > 0xffffffff
  )
    throw new Error("This video has invalid saved clip lengths.");
  return value;
}
function seed(value: unknown): string {
  if (typeof value !== "string" || !exactSeed(value))
    throw new Error(
      "This video did not save exact per-clip seeds. Use Start again from its saved job.",
    );
  return value;
}
function project(
  metadata: Record<string, unknown>,
): ChainRequestWire & { stages: ChainStageWire[] } {
  const chain = record(metadata.chain);
  if (chain.continue_from)
    throw new Error(
      "This linked video needs its original incoming video. Use Start again from its saved job.",
    );
  if (
    !Array.isArray(chain.stages) ||
    !chain.stages.length ||
    chain.stages.length > 16 ||
    chain.stage_count !== chain.stages.length
  )
    throw new Error("This video did not save complete ordered clip settings.");
  const hasList = (value: unknown) => Array.isArray(value) && value.length > 0;
  const rawStages = chain.stages.map(record);
  if (
    (hasList(metadata.references) || hasList(metadata.generation_references)) &&
    !rawStages.some((stage) => hasList(stage.generation_references))
  )
    throw new Error(
      "This older video's reference conditioning lacks complete per-clip provenance. Use Start again from its saved job or reattach the original inputs.",
    );
  if (
    hasList(metadata.keyframes) &&
    !rawStages.some((stage) => hasList(stage.keyframes))
  )
    throw new Error(
      "This older video's keyframe conditioning lacks complete per-clip provenance. Use Start again from its saved job or reattach the original inputs.",
    );
  for (const key of [
    "spatial_upscale",
    "temporal_upscale",
    "ic_lora_control",
    "retake_range",
    "cfg_plus",
    "control_model",
    "reference_weight",
    "control_scale",
    "scheduler",
    "pipeline",
    "guidance_overrides",
    "sample_shift",
    "distill_strength_high",
    "distill_strength_low",
    "upscale_model",
    "true_cfg",
    "cfg_start_step",
    "duration_prediction_requested",
    "video_only",
    "pipeline_requested",
    "audio_file_path",
    "source_video_path",
    "extend_video_path",
    "extend_overlap_frames",
    "id_image_sha256",
    "mask_image_sha256",
    "control_image_sha256",
    "edit_image_sha256s",
  ])
    if (
      metadata[key] != null &&
      metadata[key] !== false &&
      metadata[key] !== "" &&
      (!Array.isArray(metadata[key]) || metadata[key].length > 0)
    )
      throw new Error(
        `These clip settings cannot preserve ${key}. Use Start again from the saved job.`,
      );
  if (metadata.source_fit && !parseSourceFitPolicy(metadata.source_fit))
    throw new Error("The saved source fitting policy cannot be restored.");
  const first = record(chain.stages[0]),
    base = seed(first.seed);
  const stages = chain.stages.map((raw, index) => {
    const s = record(raw);
    if (
      typeof s.prompt !== "string" ||
      !s.prompt.trim() ||
      new TextEncoder().encode(s.prompt).length > 77000 ||
      !["smooth", "cut", "fade"].includes(String(s.transition))
    )
      throw new Error("This video has invalid saved clip prompts or joins.");
    if (
      s.loras != null &&
      (!Array.isArray(s.loras) ||
        s.loras.some((l) => {
          const v = record(l);
          return (
            typeof v.path !== "string" ||
            typeof v.scale !== "number" ||
            !Number.isFinite(v.scale)
          );
        }))
    )
      throw new Error("This video has invalid saved clip adapters.");
    return {
      prompt: s.prompt,
      frames: whole(s.frames),
      transition: s.transition as NonNullable<ChainStageWire["transition"]>,
      seed_offset: String(BigInt(base) ^ BigInt(seed(s.seed))),
      ...(s.fade_frames != null
        ? { fade_frames: whole(s.fade_frames, true) }
        : {}),
      ...(Array.isArray(s.loras)
        ? {
            loras: structuredClone(s.loras) as NonNullable<
              ChainStageWire["loras"]
            >,
          }
        : {}),
      ...(index === 0 && typeof metadata.negative_prompt === "string"
        ? { negative_prompt: metadata.negative_prompt }
        : {}),
    };
  });
  if (
    typeof metadata.model !== "string" ||
    !metadata.model ||
    typeof metadata.guidance !== "number" ||
    !Number.isFinite(metadata.guidance)
  )
    throw new Error("This video did not save its recipe settings.");
  return {
    model: metadata.model,
    width: whole(metadata.width),
    height: whole(metadata.height),
    steps: whole(metadata.steps),
    fps: whole(metadata.fps),
    guidance: metadata.guidance,
    seed: base,
    motion_tail_frames: whole(chain.motion_tail_frames, true),
    stages,
    output_format: "mp4",
    output_mode: "sequence",
    ...(typeof metadata.strength === "number"
      ? { strength: metadata.strength }
      : {}),
    ...(typeof metadata.enable_audio === "boolean"
      ? { enable_audio: metadata.enable_audio }
      : {}),
    ...(metadata.source_fit
      ? {
          source_fit: structuredClone(metadata.source_fit) as NonNullable<
            ChainRequestWire["source_fit"]
          >,
        }
      : {}),
  };
}
function reference(raw: unknown, data: string): GenerationReference {
  const d = record(raw) as unknown as GenerationReferenceMetadata & {
    image_role?: string;
  };
  if (!digest(d.sha256) || typeof d.mime_type !== "string")
    throw new Error("The saved reference descriptor is incomplete.");
  const base = {
    media: { authority: "inline" as const, data },
    mime_type: d.mime_type,
    provenance: {
      name: d.name ?? null,
      sha256: d.sha256,
      ...(d.crop ? { crop: structuredClone(d.crop) } : {}),
    },
  };
  if (d.kind === "image")
    return {
      ...base,
      kind: "image",
      width: whole(d.width),
      height: whole(d.height),
    };
  if (
    d.kind === "named_image" &&
    ["front", "left", "back", "right"].includes(d.image_role ?? "")
  )
    return {
      ...base,
      kind: "named_image",
      role: d.image_role as "front",
      width: whole(d.width),
      height: whole(d.height),
    };
  if (d.kind === "video") {
    if (typeof d.fps !== "number" || !Number.isFinite(d.fps) || d.fps <= 0)
      throw new Error("The saved video reference FPS is missing.");
    return {
      ...base,
      kind: "video",
      width: whole(d.width),
      height: whole(d.height),
      duration_ms: whole(d.duration_ms),
      fps: d.fps,
      ...Object.fromEntries(
        [
          "frame_count",
          "has_audio",
          "audio_duration_ms",
          "audio_sample_count",
          "audio_sample_rate",
          "audio_channels",
        ]
          .filter((k) => (d as unknown as Record<string, unknown>)[k] != null)
          .map((k) => [k, (d as unknown as Record<string, unknown>)[k]]),
      ),
    };
  }
  if (d.kind === "audio")
    return {
      ...base,
      kind: "audio",
      duration_ms: whole(d.duration_ms),
      sample_rate: whole(d.sample_rate),
      channels: whole(d.channels),
      ...(d.sample_count != null
        ? { sample_count: whole(d.sample_count) }
        : {}),
    };
  throw new Error("These saved clips contain an unsupported reference kind.");
}
export interface LegacyVideoSettings {
  automaticDuration?: import("../lib/videoDurationAuthoring").VideoDurationAuthoring;
  owner: ChainOwner;
  request: ChainRequestWire & { stages: ChainStageWire[] };
  notice: string;
}
/** Restore immutable authored inputs from this exact authenticated Library copy. */
export async function inspectLegacyVideoSettings(
  owner: ChainOwner,
  filename: string,
  clickedMetadata: object,
  isCurrent: () => boolean = () => true,
): Promise<LegacyVideoSettings> {
  const clicked = JSON.parse(JSON.stringify(clickedMetadata)) as object;
  project(record(clicked));
  const captured = { ...owner, target: { ...owner.target } };
  const stamp = JSON.stringify(owner);
  const check = () => {
    if (!isCurrent() || JSON.stringify(owner) !== stamp)
      throw new Error(
        "The selected video or machine changed. Open Use settings again.",
      );
  };
  await assertChainOwner(captured);
  check();
  const inventory = await retainedSourceMediaInventory(
    filename,
    captured.target,
  );
  check();
  if (inventory.availability === "unavailable_legacy") {
    const m = record(clicked),
      chain = record(m.chain),
      stages = chain.stages as unknown[];
    if (
      m.source_image_sha256 ||
      stages.some((value) => {
        const stage = record(value);
        return (
          (Array.isArray(stage.keyframes) && stage.keyframes.length > 0) ||
          (Array.isArray(stage.generation_references) &&
            stage.generation_references.length > 0)
        );
      })
    )
      throw new Error(
        "This video's original input archive is unavailable. Reattach its original inputs or use Start again from its saved job.",
      );
    const rows = await apiJsonTo<
      Array<{ filename: string; metadata: unknown }>
    >(captured.target, `/api/gallery?filename=${encodeURIComponent(filename)}`);
    check();
    const matches = Array.isArray(rows)
      ? rows.filter((row) => row.filename === filename)
      : [];
    if (
      matches.length !== 1 ||
      !mirrorMetadataMatches(
        JSON.stringify(clicked),
        JSON.stringify(matches[0]!.metadata),
      )
    )
      throw new Error(
        "The saved video settings changed. Reopen Use settings from the original Library.",
      );
    const fresh = record(matches[0]!.metadata),
      request = project(fresh);
    await assertChainOwner(captured);
    check();
    return finishLegacySettings(fresh, request, captured);
  }
  if (inventory.availability !== "available")
    throw new Error(
      "This video’s original input archive is unavailable. Use Start again from its saved job or reattach the original inputs.",
    );
  if (
    inventory.members.length > 64 ||
    inventory.members.reduce((n, m) => n + m.size_bytes, 0) > MAX ||
    new Set(inventory.members.map((m) => m.member_id)).size !==
      inventory.members.length
  )
    throw new Error(
      "The original clip inputs exceed the supported 32 MiB / 64 input limit.",
    );
  const path = `/api/gallery/source-media/${encodeURIComponent(filename)}/transfer`;
  const offer = record(await apiJsonTo(captured.target, path));
  check();
  if (
    !digest(offer.archive_identity_sha256) ||
    !mirrorMetadataMatches(
      JSON.stringify(clicked),
      JSON.stringify(offer.metadata),
    ) ||
    !Array.isArray(offer.members) ||
    offer.members.length !== inventory.members.length
  )
    throw new Error(
      "The video input archive changed. Open Use settings again.",
    );
  const metadata = record(offer.metadata),
    request = project(metadata),
    chain = record(metadata.chain),
    rawStages = chain.stages as unknown[];
  const seen = new Set<string>(),
    refs = new Map<number, Map<number, GenerationReference>>(),
    keys = new Map<
      number,
      Map<number, NonNullable<ChainStageWire["keyframes"]>[number]>
    >();
  const rows = inventory.members.map((member) => {
    const matches = (offer.members as unknown[])
      .map(record)
      .filter((m) => m.member_id === member.member_id);
    if (matches.length !== 1)
      throw new Error("The clip archive has duplicate or missing members.");
    const m = matches[0]!;
    if (
      m.role !== member.role ||
      m.size_bytes !== member.size_bytes ||
      (member.content_sha256 != null && member.content_sha256 !== m.sha256) ||
      !digest(m.sha256) ||
      !Number.isSafeInteger(member.size_bytes) ||
      member.size_bytes <= 0 ||
      seen.has(member.role)
    )
      throw new Error("The clip archive has invalid member identities.");
    seen.add(member.role);
    const match =
      /^stage_(source|reference|keyframe):(0|[1-9]\d*)(?::(0|[1-9]\d*))?$/.exec(
        member.role,
      );
    if (
      !match ||
      Number(match[2]) >= request.stages.length ||
      (match[1] === "source") !== (match[3] === undefined)
    )
      throw new Error("The clip archive contains unrecognized stage inputs.");
    return {
      member,
      sha256: m.sha256 as string,
      kind: match[1]!,
      stage: Number(match[2]),
      index: match[3] === undefined ? undefined : Number(match[3]),
    };
  });
  if (
    (metadata.source_image_sha256 != null ||
      Boolean(metadata.source_image_name)) &&
    !rows.some((row) => row.kind === "source" && row.stage === 0)
  )
    throw new Error(
      "The original first-frame input is missing from this video archive.",
    );
  for (const row of rows) {
    check();
    const response = await apiFetchTo(
      captured.target,
      `/api/gallery/source-media/${encodeURIComponent(filename)}/${encodeURIComponent(row.member.member_id)}`,
    );
    check();
    if (!response.ok || !response.body)
      throw new Error("The original clip input could not be restored.");
    const reader = response.body.getReader(),
      chunks: Uint8Array[] = [];
    let size = 0;
    try {
      for (;;) {
        const chunk = await reader.read();
        check();
        if (chunk.done) break;
        size += chunk.value.byteLength;
        if (size > row.member.size_bytes || size > MAX)
          throw new Error(
            "The original clip input exceeded its declared size.",
          );
        chunks.push(chunk.value);
      }
    } finally {
      await reader.cancel().catch(() => undefined);
    }
    if (size !== row.member.size_bytes)
      throw new Error("The original clip input is incomplete.");
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    const actual = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    )
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    check();
    if (actual.toLowerCase() !== row.sha256.toLowerCase())
      throw new Error("The original clip input failed its integrity check.");
    const data = await blobToBase64(new Blob([bytes]));
    check();
    const stage = request.stages[row.stage]!;
    const saved = record(rawStages[row.stage]);
    if (row.kind === "source") stage.source_image = data;
    else if (row.kind === "reference") {
      const ds = saved.generation_references;
      if (!Array.isArray(ds))
        throw new Error("The saved reference order is missing.");
      const matches = ds.filter((d) => record(d).index === row.index! + 1);
      if (matches.length !== 1 || record(matches[0]).sha256 !== row.sha256)
        throw new Error("The original reference descriptor changed.");
      let group = refs.get(row.stage);
      if (!group) {
        group = new Map();
        refs.set(row.stage, group);
      }
      group.set(row.index!, reference(matches[0], data));
    } else {
      const ds = saved.keyframes;
      if (!Array.isArray(ds) || !ds[row.index!])
        throw new Error("The saved keyframe order is missing.");
      const d = record(ds[row.index!]);
      const frame = whole(d.frame, true);
      if (
        frame >= stage.frames ||
        d.sha256 !== row.sha256 ||
        (row.member.frame_index != null && row.member.frame_index !== frame)
      )
        throw new Error("The original keyframe descriptor changed.");
      let group = keys.get(row.stage);
      if (!group) {
        group = new Map();
        keys.set(row.stage, group);
      }
      group.set(row.index!, {
        frame,
        image: data,
        ...(typeof d.name === "string" ? { name: d.name } : {}),
      });
    }
  }
  for (let i = 0; i < request.stages.length; i++) {
    const saved = record(rawStages[i]);
    for (const [field, map, source] of [
      ["generation_references", refs, "generation_references"],
      ["keyframes", keys, "keyframes"],
    ] as const) {
      const expected = Array.isArray(saved[source])
        ? (saved[source] as unknown[]).length
        : 0;
      const group = map.get(i);
      if (
        (group?.size ?? 0) !== expected ||
        (group &&
          Array.from({ length: expected }, (_, n) => n).some(
            (n) => !group.has(n),
          ))
      )
        throw new Error("Some original clip inputs are missing.");
      if (group && field === "generation_references")
        request.stages[i]!.generation_references = Array.from(
          (group as Map<number, GenerationReference>).entries(),
        )
          .sort(([a], [b]) => a - b)
          .map(([, v]) => v);
      if (group && field === "keyframes")
        request.stages[i]!.keyframes = Array.from(
          (
            group as Map<
              number,
              NonNullable<ChainStageWire["keyframes"]>[number]
            >
          ).entries(),
        )
          .sort(([a], [b]) => a - b)
          .map(([, v]) => v);
    }
  }
  const after = record(await apiJsonTo(captured.target, path));
  check();
  if (after.archive_identity_sha256 !== offer.archive_identity_sha256)
    throw new Error("The video input archive changed during restoration.");
  await assertChainOwner(captured);
  check();
  return finishLegacySettings(metadata, request, captured);
}
function finishLegacySettings(
  metadata: Record<string, unknown>,
  request: ChainRequestWire & { stages: ChainStageWire[] },
  captured: ChainOwner,
): LegacyVideoSettings {
  let automaticDuration:
    import("../lib/videoDurationAuthoring").VideoDurationAuthoring | undefined;
  if (metadata.output_mode === "one-shot") {
    const first = request.stages[0]!;
    const commonReferences = JSON.stringify(first.generation_references ?? []);
    const commonAdapters = JSON.stringify(first.loras ?? []);
    const seedsUniform = request.stages.every(
      (stage) => stage.seed_offset === "0",
    );
    const seedsIndexed = request.stages.every(
      (stage, index) => stage.seed_offset === String(index),
    );
    const unsafe =
      (!seedsUniform && !seedsIndexed) ||
      Boolean(metadata.negative_prompt) ||
      (first.loras?.length ?? 0) > MAX_LORA_STACK ||
      request.stages.some(
        (stage, index) =>
          stage.prompt !== first.prompt ||
          stage.transition !== "smooth" ||
          Boolean(stage.fade_frames) ||
          JSON.stringify(stage.loras ?? []) !== commonAdapters ||
          JSON.stringify(stage.generation_references ?? []) !==
            commonReferences ||
          (index > 0 && (stage.source_image || stage.keyframes?.length)) ||
          stage.keyframes?.some((frame) => frame.frame !== 0),
      );
    const total = request.stages.reduce(
      (sum, stage, index) =>
        sum +
        BigInt(stage.frames) -
        (index ? BigInt(request.motion_tail_frames ?? 0) : 0n),
      0n,
    );
    if (unsafe || total <= 0n || BigInt(whole(metadata.frames)) !== total)
      throw new Error(
        "This older single-prompt video has clip settings that cannot be restored faithfully. Use Start again from its saved job.",
      );
    automaticDuration = {
      durationMs: String((total * 1000n) / BigInt(request.fps!)),
      scenesEnabled: false,
      scenes: [],
      reuseNotice:
        "This older video's duration comes from its saved length; current automatic joins may differ.",
    };
  }
  return {
    owner: captured,
    request,
    ...(automaticDuration ? { automaticDuration } : {}),
    notice:
      "Restored each clip’s effective seed, join, adapters and original inputs. Historical per-clip negative prompts were not saved; only the first clip’s saved negative prompt is restored. Reopen Use settings after reloading to restore these original inputs again.",
  };
}

/** Exact first-shot conditioning for a proven automatic legacy movie. */
export function legacyAutomaticRequest(settings: LegacyVideoSettings) {
  if (!settings.automaticDuration)
    throw new Error("These saved clips require the clip editor.");
  const r = settings.request,
    first = r.stages[0]!;
  return {
    model: r.model,
    prompt: first.prompt,
    width: r.width,
    height: r.height,
    steps: r.steps,
    guidance: r.guidance,
    frames: first.frames,
    fps: r.fps ?? 24,
    output_format: "mp4" as const,
    ...(r.seed != null ? { seed: r.seed } : {}),
    ...(r.strength != null ? { strength: r.strength } : {}),
    ...(r.enable_audio != null ? { enable_audio: r.enable_audio } : {}),
    ...(r.source_fit ? { source_fit: r.source_fit } : {}),
    ...(first.negative_prompt != null
      ? { negative_prompt: first.negative_prompt }
      : {}),
    ...(first.loras
      ? {
          loras: JSON.parse(JSON.stringify(first.loras)) as NonNullable<
            ChainStageWire["loras"]
          >,
        }
      : {}),
    ...(first.source_image
      ? {
          source_image: first.source_image,
          source_image_name: "Original first frame",
        }
      : {}),
    ...(first.keyframes?.length
      ? {
          keyframes: JSON.parse(JSON.stringify(first.keyframes)) as NonNullable<
            ChainStageWire["keyframes"]
          >,
        }
      : {}),
    ...(first.generation_references?.length
      ? {
          references: JSON.parse(
            JSON.stringify(first.generation_references),
          ) as NonNullable<ChainStageWire["generation_references"]>,
        }
      : {}),
  };
}
