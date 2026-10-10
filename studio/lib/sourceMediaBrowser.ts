import type { ApiTarget } from "../api/client";
import type { RetainedSourceMediaMember } from "../api/gallerySourceMedia";

export interface SourceMediaOwner {
  key: string;
  filename: string;
  title: string;
  hostLabel: string;
  target: ApiTarget;
  instanceId?: string;
  revision?: string | number;
}
export interface SourceMediaSelection {
  owner: SourceMediaOwner;
  member: RetainedSourceMediaMember;
}
export function sourceOwnerIdentity(owner: SourceMediaOwner): string {
  return JSON.stringify([
    owner.key,
    owner.filename,
    owner.target.baseUrl,
    owner.target.apiKey,
    owner.instanceId,
    owner.revision,
  ]);
}
export function sourceMemberKind(member: RetainedSourceMediaMember): string {
  if (member.media_kind) return member.media_kind;
  if (
    [
      "source_image",
      "identity_image",
      "identity_images",
      "edit_images",
      "mask_image",
      "control_image",
      "keyframes",
      "matting_processed_source_image",
      "matting_processed_references",
    ].includes(member.role) ||
    member.role.startsWith("stage_source:")
  )
    return "image";
  if (["audio_file", "audio_file_path"].includes(member.role)) return "audio";
  if (
    [
      "source_video",
      "source_video_path",
      "extend_video",
      "extend_video_path",
    ].includes(member.role)
  )
    return "video";
  return "unknown";
}
export function sourceMemberLabel(member: RetainedSourceMediaMember): string {
  if (member.label) return member.label;
  if (member.frame_index !== undefined)
    return `Keyframe · frame ${member.frame_index}`;
  const names: Record<string, string> = {
    source_image: "Source image",
    identity_image: "Identity photo",
    identity_images: "Identity photo",
    edit_images: "Reference image",
    references: "Reference",
    mask_image: "Mask",
    control_image: "Control image",
    keyframes: "Keyframe",
    audio_file: "Source audio",
    audio_file_path: "Source audio",
    source_video: "Source video",
    source_video_path: "Source video",
    extend_video: "Continuation video",
    extend_video_path: "Continuation video",
    matting_processed_source_image: "Processed source image",
    matting_processed_references: "Processed reference",
  };
  const label = names[member.role] ?? member.role.replaceAll("_", " ");
  return member.position === undefined
    ? label
    : `${label} ${member.position + 1}`;
}

export interface SourceMediaGroup {
  key: string;
  usages: SourceMediaSelection[];
}
export function groupSourceMedia(
  selections: SourceMediaSelection[],
): SourceMediaGroup[] {
  const groups = new Map<string, SourceMediaGroup>();
  for (const selection of selections) {
    const digest = selection.member.content_sha256;
    const key =
      digest && /^[a-f0-9]{64}$/.test(digest)
        ? `content:${digest}`
        : `usage:${sourceOwnerIdentity(selection.owner)}:${selection.member.member_id}`;
    let group = groups.get(key);
    if (!group) groups.set(key, (group = { key, usages: [] }));
    group.usages.push(selection);
  }
  return [...groups.values()];
}

export function sourceHostIdentity(owner: SourceMediaOwner): string {
  return JSON.stringify([
    owner.target.baseUrl,
    owner.target.apiKey ?? null,
    owner.instanceId ?? null,
  ]);
}
export function filterSourceMedia(
  selections: SourceMediaSelection[],
  filters: { host?: string; kind?: string; search?: string },
): SourceMediaGroup[] {
  const query = filters.search?.trim().toLocaleLowerCase() ?? "";
  return groupSourceMedia(
    selections.filter(
      ({ owner, member }) =>
        (!filters.host || sourceHostIdentity(owner) === filters.host) &&
        (!filters.kind || sourceMemberKind(member) === filters.kind) &&
        (!query ||
          `${owner.title} ${owner.hostLabel} ${sourceMemberLabel(member)} ${member.display_name}`
            .toLocaleLowerCase()
            .includes(query)),
    ),
  );
}
