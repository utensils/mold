import { apiFetchTo, type ApiTarget } from "./client";

export type RetainedSourceMediaAvailability =
  | "available"
  | "unavailable_legacy"
  | "unavailable_missing_or_corrupt"
  | "unavailable_auth";

export interface RetainedSourceMediaMember {
  member_id: string;
  role: string;
  display_name: string;
  size_bytes: number;
}

export interface RetainedSourceMediaInventory {
  availability: RetainedSourceMediaAvailability;
  members: RetainedSourceMediaMember[];
}

export interface RetainedSourceMediaReuseSession {
  instance_id: string;
  expires_at: number;
  request_sha256: string;
  session_handle: string;
}

function inventoryPath(filename: string): string {
  return `/api/gallery/source-media/${encodeURIComponent(filename)}`;
}

function memberPath(filename: string, memberId: string): string {
  return `${inventoryPath(filename)}/${encodeURIComponent(memberId)}`;
}

function reuseSessionPath(filename: string): string {
  return `${inventoryPath(filename)}/reuse-sessions`;
}

export const __testing__ = { inventoryPath, memberPath, reuseSessionPath };

export async function retainedSourceMediaInventory(
  filename: string,
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<RetainedSourceMediaInventory> {
  const response = await apiFetchTo(
    target,
    inventoryPath(filename),
    signal ? { signal } : {},
  );
  if (!response.ok) {
    throw new Error(
      `Could not inspect retained source media (HTTP ${response.status})`,
    );
  }
  return response.json();
}

export async function retainedSourceMediaBlob(
  filename: string,
  memberId: string,
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await apiFetchTo(
    target,
    memberPath(filename, memberId),
    signal ? { signal } : {},
  );
  if (!response.ok) {
    throw new Error(
      `Could not restore retained source media (HTTP ${response.status})`,
    );
  }
  return response.blob();
}

/** Issue an opaque, short-lived handle bound by the server to this exact
 * payload-free request, API-key identity, archive identity, and member set.
 * The handle is supplied only on the immediately following generation POST. */
export async function createRetainedSourceMediaReuseSession<TRequest>(
  filename: string,
  memberIds: readonly string[],
  targetRequest: TRequest,
  target: ApiTarget,
): Promise<RetainedSourceMediaReuseSession> {
  const response = await apiFetchTo(target, reuseSessionPath(filename), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      target_request: targetRequest,
      member_ids: [...memberIds],
    }),
  });
  return response.json();
}

/** Persistent wording shared by web, desktop and mobile Reuse settings. */
export function retainedSourceMediaDisclosure(
  availability: RetainedSourceMediaAvailability,
): string | null {
  switch (availability) {
    case "available":
      return null;
    case "unavailable_legacy":
      return "This older print did not retain its original source media. Reattach it before developing.";
    case "unavailable_missing_or_corrupt":
      return "This print’s retained source media is missing or damaged. Reattach it before developing.";
    case "unavailable_auth":
      return "Connect this machine with an API key to restore its private source media.";
  }
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  // Avoid `String.fromCharCode(...bytes)`: retained videos can be large enough
  // to overflow the JS argument stack.
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

function assertVacant(request: Record<string, unknown>, field: string): void {
  if (request[field] !== undefined && request[field] !== null) {
    throw new Error(`Target request already contains ${field}`);
  }
}

const REQUEST_FIELD_FOR_ROLE: Readonly<Record<string, string>> = {
  source_image: "source_image",
  identity_image: "id_image",
  identity_images: "id_images",
  edit_images: "edit_images",
  mask_image: "mask_image",
  control_image: "control_image",
  audio_file: "audio_file",
  audio_file_path: "audio_file",
  source_video: "source_video",
  source_video_path: "source_video",
  extend_video: "extend_video",
  extend_video_path: "extend_video",
  keyframes: "keyframes",
  references: "references",
};

/** Select only roles whose outgoing request still lacks byte authority. This
 * preserves a user's later reattachment while allowing descriptor-only H3
 * reference topology to be hydrated with retained bytes. */
export function retainedSourceMediaMembersForRequest(
  members: readonly RetainedSourceMediaMember[],
  targetRequest: object,
): RetainedSourceMediaMember[] {
  const request = targetRequest as Record<string, unknown>;
  return members.filter((member) => {
    const field = REQUEST_FIELD_FOR_ROLE[member.role];
    if (!field) return false;
    if (field !== "references") return request[field] == null;
    const references = request.references;
    return (
      Array.isArray(references) &&
      references.length > 0 &&
      references.every(
        (reference) =>
          typeof reference === "object" &&
          reference !== null &&
          (reference as { media?: { authority?: unknown } }).media
            ?.authority === "descriptor",
      )
    );
  });
}

/** Authenticated cross-host fallback. Bytes are downloaded from the print's
 * origin and placed directly into the destination request; paths, pin ids and
 * store identities never cross the API boundary. The returned object is a
 * fresh request so callers can keep their route-planning snapshot immutable. */
export async function relayRetainedSourceMedia<TRequest extends object>(
  filename: string,
  members: readonly RetainedSourceMediaMember[],
  targetRequest: TRequest,
  origin: ApiTarget,
  signal?: AbortSignal,
): Promise<TRequest> {
  const request = Object.assign({}, targetRequest) as Record<string, unknown>;
  const roles = new Set(members.map((member) => member.role));
  for (const [role, field] of [
    ["source_image", "source_image"],
    ["identity_image", "id_image"],
    ["identity_images", "id_images"],
    ["edit_images", "edit_images"],
    ["mask_image", "mask_image"],
    ["control_image", "control_image"],
    ["audio_file", "audio_file"],
    ["audio_file_path", "audio_file"],
    ["source_video", "source_video"],
    ["source_video_path", "source_video"],
    ["extend_video", "extend_video"],
    ["extend_video_path", "extend_video"],
    ["keyframes", "keyframes"],
  ] as const) {
    if (roles.has(role)) assertVacant(request, field);
  }
  const grouped = new Map<string, string[]>();
  for (const member of members) {
    const bytes = new Uint8Array(
      await (
        await retainedSourceMediaBlob(
          filename,
          member.member_id,
          origin,
          signal,
        )
      ).arrayBuffer(),
    );
    const encoded = bytesToBase64(bytes);
    const values = grouped.get(member.role) ?? [];
    values.push(encoded);
    grouped.set(member.role, values);
  }

  const scalar = (role: string, field: string) => {
    const values = grouped.get(role);
    if (!values?.length) return;
    if (values.length !== 1)
      throw new Error(`Retained role ${role} is ambiguous`);
    assertVacant(request, field);
    request[field] = values[0];
  };
  scalar("source_image", "source_image");
  scalar("identity_image", "id_image");
  scalar("mask_image", "mask_image");
  scalar("control_image", "control_image");
  scalar("audio_file", "audio_file");
  scalar("audio_file_path", "audio_file");
  scalar("source_video", "source_video");
  scalar("source_video_path", "source_video");
  scalar("extend_video", "extend_video");
  scalar("extend_video_path", "extend_video");

  for (const [role, field] of [
    ["identity_images", "id_images"],
    ["edit_images", "edit_images"],
  ] as const) {
    const values = grouped.get(role);
    if (!values?.length) continue;
    assertVacant(request, field);
    request[field] = values;
  }

  const keyframes = grouped.get("keyframes");
  if (keyframes?.length) {
    assertVacant(request, "keyframes");
    request.keyframes = keyframes.map((encoded) => {
      const binary = atob(encoded);
      const bytes = Uint8Array.from(binary, (character) =>
        character.charCodeAt(0),
      );
      return JSON.parse(new TextDecoder().decode(bytes));
    });
  }

  const references = grouped.get("references");
  if (references?.length) {
    const descriptors = request.references;
    if (
      !Array.isArray(descriptors) ||
      descriptors.length !== references.length
    ) {
      throw new Error(
        "Target request reference descriptors do not match retained references",
      );
    }
    request.references = descriptors.map((reference, index) => ({
      ...(reference as Record<string, unknown>),
      media: { authority: "inline", data: references[index] },
    }));
  }

  const supported = new Set(Object.keys(REQUEST_FIELD_FOR_ROLE));
  const unsupported = [...grouped.keys()].find((role) => !supported.has(role));
  if (unsupported)
    throw new Error(`Retained source-media role ${unsupported} is unsupported`);
  return request as TRequest;
}
