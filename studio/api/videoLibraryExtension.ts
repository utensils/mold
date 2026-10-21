import { apiJsonTo } from "./client";
import { assertVideoJobOwner, type VideoJobOwner } from "./videoJobs";
import {
  retainedSourceMediaInventory,
  type RetainedSourceMediaInventory,
  type RetainedSourceMediaMember,
} from "./gallerySourceMedia";
import { mirrorMetadataMatches } from "../lib/galleryMirrorIdentity";
import type { VideoJobSource } from "../lib/api/videoJobTypes";

export interface VideoOriginalReference {
  descriptor: Record<string, unknown>;
  member?: RetainedSourceMediaMember;
}
export interface VideoLibraryExtension {
  owner: VideoJobOwner;
  source: VideoJobSource;
  title: string;
  /** New-footage prefill only. The original recipe remains server-owned. */
  metadata: Record<string, unknown>;
  inventory: RetainedSourceMediaInventory;
  originalReferences: VideoOriginalReference[];
}
const digest = /^[a-f0-9]{64}$/;
const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const changed = () =>
  new Error(
    "This video's saved source changed. Refresh the Library and open Extend Video again.",
  );
function prefill(metadata: Record<string, unknown>): Record<string, unknown> {
  // Never restore historical endpoints as the newly added footage's anchors.
  // A new seed is random by default; JSON Number cannot safely restore all u64s.
  const allowed = new Set([
    "model",
    "prompt",
    "negative_prompt",
    "width",
    "height",
    "fps",
    "steps",
    "guidance",
    "strength",
    "pipeline",
    "enable_audio",
    "offload",
    "loras",
    "lora",
    "lora_scale",
    "reference_weight",
    "spatial_upscale",
    "temporal_upscale",
    "guidance_overrides",
    "sample_shift",
    "distill_strength_high",
    "distill_strength_low",
    "scheduler",
    "cfg_plus",
    "true_cfg",
    "cfg_start_step",
    "control_model",
    "control_scale",
    "ic_lora_control",
    "video_only",
  ]);
  const result = Object.fromEntries(
    Object.entries(metadata).filter(([key]) => allowed.has(key)),
  );
  if (
    typeof metadata.spatial_upscale === "string" &&
    Number.isSafeInteger(metadata.generation_width) &&
    Number(metadata.generation_width) > 0
  )
    result.width = metadata.generation_width;
  if (
    typeof metadata.spatial_upscale === "string" &&
    Number.isSafeInteger(metadata.generation_height) &&
    Number(metadata.generation_height) > 0
  )
    result.height = metadata.generation_height;
  const first =
    record(metadata.chain) && Array.isArray(metadata.chain.stages)
      ? metadata.chain.stages[0]
      : undefined;
  if (record(first) && typeof first.prompt === "string")
    result.prompt = first.prompt;
  return result;
}
function references(
  metadata: Record<string, unknown>,
  inventory: RetainedSourceMediaInventory,
): VideoOriginalReference[] {
  const top = Array.isArray(metadata.references) ? metadata.references : [];
  const stages =
    record(metadata.chain) && Array.isArray(metadata.chain.stages)
      ? metadata.chain.stages
      : [];
  const index = stages.length - 1;
  const last = stages[index];
  const final =
    record(last) && Array.isArray(last.generation_references)
      ? last.generation_references
      : [];
  const selected = top.length ? top : final;
  if (
    selected.length > 64 ||
    selected.some(
      (value) =>
        !record(value) ||
        !["image", "video", "audio"].includes(String(value.kind)) ||
        !digest.test(String(value.sha256)),
    )
  )
    throw changed();
  return selected.map((value, position) => {
    const descriptor = value as Record<string, unknown>;
    return {
      descriptor,
      ...(() => {
        const member = inventory.members.find(
          (member) =>
            digest.test(String(descriptor.sha256)) &&
            member.content_sha256 === descriptor.sha256 &&
            (top.length
              ? member.role === "references" ||
                /^stage_reference:\d+:\d+$/.test(member.role)
              : member.role === `stage_reference:${index}:${position}`),
        );
        return member ? { member } : {};
      })(),
    };
  });
}
/** Explicit Library open only: metadata/inventory, never MP4/input downloads.
 * The caller continuously fences visibility, route, credentials and selection. */
export async function inspectVideoLibraryExtension(
  owner: VideoJobOwner,
  filename: string,
  clickedMetadata: Record<string, unknown>,
  isCurrent: () => boolean = () => true,
  title = filename,
  expectedArchiveIdentity?: string,
): Promise<VideoLibraryExtension> {
  const captured = { ...owner, target: { ...owner.target } };
  // Gallery rows may be reactive Vue proxies; snapshot their JSON wire value.
  const clicked = JSON.parse(JSON.stringify(clickedMetadata)) as Record<
    string,
    unknown
  >;
  const check = () => {
    if (!isCurrent() || JSON.stringify(owner) !== JSON.stringify(captured))
      throw changed();
  };
  check();
  await assertVideoJobOwner(captured, true);
  check();
  const inventory = await retainedSourceMediaInventory(
    filename,
    captured.target,
  );
  check();
  if (
    inventory.members.length > 64 ||
    new Set(inventory.members.map((m) => m.member_id)).size !==
      inventory.members.length
  )
    throw changed();
  let metadata = clicked;
  let hash: string | undefined;
  if (inventory.availability === "available") {
    const offer = await apiJsonTo<Record<string, unknown>>(
      captured.target,
      `/api/gallery/source-media/${encodeURIComponent(filename)}/transfer`,
    );
    check();
    if (
      !digest.test(String(offer.archive_identity_sha256)) ||
      !digest.test(String(offer.output_sha256)) ||
      !Number.isSafeInteger(offer.output_size_bytes) ||
      Number(offer.output_size_bytes) <= 0 ||
      !record(offer.metadata) ||
      !Array.isArray(offer.members) ||
      offer.members.length !== inventory.members.length ||
      (expectedArchiveIdentity
        ? !digest.test(expectedArchiveIdentity) ||
          offer.archive_identity_sha256 !== expectedArchiveIdentity
        : !mirrorMetadataMatches(
            JSON.stringify(clicked),
            JSON.stringify(offer.metadata),
          ))
    )
      throw changed();
    const ids = new Set<string>();
    for (const member of offer.members) {
      if (
        !record(member) ||
        typeof member.member_id !== "string" ||
        ids.has(member.member_id) ||
        !digest.test(String(member.sha256)) ||
        !Number.isSafeInteger(member.size_bytes) ||
        Number(member.size_bytes) <= 0 ||
        !inventory.members.some(
          (m) =>
            m.member_id === member.member_id &&
            m.role === member.role &&
            m.size_bytes === member.size_bytes,
        )
      )
        throw changed();
      ids.add(member.member_id);
    }
    metadata = offer.metadata;
    hash = String(offer.archive_identity_sha256);
  } else if (
    expectedArchiveIdentity ||
    inventory.availability !== "unavailable_legacy"
  ) {
    if (expectedArchiveIdentity) throw changed();
    throw new Error(
      inventory.availability === "unavailable_auth"
        ? "Reconnect with access to this video's original inputs before extending it."
        : "This video's original input archive is unavailable. Restore it before extending the video.",
    );
  }
  await assertVideoJobOwner(captured, true);
  check();
  return {
    owner: captured,
    source: {
      filename,
      ...(hash ? { expected_archive_identity_sha256: hash } : {}),
    },
    title,
    metadata: prefill(metadata),
    inventory,
    originalReferences: references(metadata, inventory),
  };
}
