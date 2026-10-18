import { ApiError, apiFetchTo, type ApiTarget } from "./client";
import { retainedSourceMediaDisclosable } from "./gallerySourceMedia";
import { mirrorMetadataMatches } from "../lib/galleryMirrorIdentity";
import type { CopyMachine, CopyOffer, CopyOptions } from "./galleryCopyTypes";
export const MAX_BYTES = 64 * 1024 * 1024;
export const MAX_JSON = 1024 * 1024;
const DIGEST = /^[a-f0-9]{64}$/;

/** Stable identity deliberately excludes secrets, which must never enter persisted records. */
export function copyMachineIdentity(machine: CopyMachine): string {
  const route = new URL(machine.target.baseUrl);
  if (route.username || route.password || route.search || route.hash)
    throw new Error(
      "Library copies require a machine address without embedded credentials or tokens.",
    );
  return JSON.stringify([
    machine.id,
    machine.target.baseUrl,
    machine.target.connectionMode ?? "secure",
    machine.instanceId,
  ]);
}

/** Extract untouched JSON values: JavaScript must never round a Rust u64 seed. */
export function rawJsonProperty(text: string, name: string): string {
  let position = 0;
  const skip = () => {
    while (/\s/.test(text[position] ?? "") && position < text.length)
      position++;
  };
  function value(): string {
    skip();
    const start = position;
    let depth = 0,
      string = false,
      escape = false;
    while (position < text.length) {
      const char = text[position]!;
      if (string) {
        if (escape) escape = false;
        else if (char === "\\") escape = true;
        else if (char === '"') string = false;
      } else if (char === '"') string = true;
      else if (char === "{" || char === "[") depth++;
      else if (char === "}" || char === "]") {
        if (!depth) break;
        depth--;
      } else if (char === "," && !depth) break;
      position++;
    }
    return text.slice(start, position).trim();
  }
  skip();
  if (text[position++] !== "{") throw new Error("Invalid JSON object.");
  while (position < text.length) {
    skip();
    if (text[position] === "}") break;
    // Scan the key separately so its colon is not consumed as a value.
    const start = position;
    if (text[position++] !== '"') break;
    let escaped = false;
    while (position < text.length) {
      const c = text[position++]!;
      if (escaped) escaped = false;
      else if (c === "\\") escaped = true;
      else if (c === '"') break;
    }
    const key = JSON.parse(text.slice(start, position));
    skip();
    if (text[position++] !== ":") break;
    const found = value();
    if (key === name) return found;
    skip();
    if (text[position++] !== ",") break;
  }
  throw new Error(`Missing ${name} in gallery response.`);
}
export async function bounded(
  response: Response,
  limit: number,
  signal?: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  if (!response.ok)
    throw new Error(`Gallery transfer failed (HTTP ${response.status}).`);
  if (Number(response.headers.get("content-length")) > limit) {
    await response.body?.cancel();
    throw new Error(
      "Copy is limited to 64 MiB per print including source media.",
    );
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Gallery response is empty.");
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    for (;;) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > limit)
        throw new Error(
          "Copy is limited to 64 MiB per print including source media.",
        );
      chunks.push(value);
    }
  } catch (error) {
    await reader.cancel();
    throw error;
  } finally {
    reader.releaseLock();
  }
  const result = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}
export async function json(
  target: ApiTarget,
  path: string,
  signal?: AbortSignal,
): Promise<{ value: Record<string, any>; text: string }> {
  const text = new TextDecoder().decode(
    await bounded(
      await apiFetchTo(target, path, { signal: signal ?? null }),
      MAX_JSON,
      signal,
    ),
  );
  return { value: JSON.parse(text), text };
}
function arrayObjects(text: string): string[] {
  if (!Array.isArray(JSON.parse(text)))
    throw new Error("The machine returned an invalid Library inventory.");
  const result: string[] = [];
  let start = -1,
    depth = 0,
    quoted = false,
    escaped = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!;
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') quoted = false;
      continue;
    }
    if (char === '"') quoted = true;
    else if (char === "{" || char === "[") {
      depth++;
      if (depth === 2) start = index;
    } else if (char === "}" || char === "]") {
      if (depth === 2 && start >= 0) result.push(text.slice(start, index + 1));
      depth--;
    }
  }
  return result;
}
async function legacyCopyOffer(
  machine: CopyMachine,
  filename: string,
  originalError: ApiError,
  signal?: AbortSignal,
  limit = MAX_BYTES,
  nativeLegacyOffer?: CopyOptions["nativeLegacyOffer"],
): Promise<CopyOffer> {
  const inventory = await json(
    machine.target,
    `/api/gallery/source-media/${encodeURIComponent(filename)}`,
    signal,
  );
  if (
    inventory.value.availability !== "unavailable_legacy" ||
    (inventory.value.members?.length ?? 0) !== 0
  )
    throw originalError;
  const listing = await json(machine.target, "/api/gallery", signal);
  const entry = arrayObjects(listing.text).find(
    (row) => JSON.parse(row).filename === filename,
  );
  if (!entry) throw originalError;
  const rawMetadata = rawJsonProperty(entry, "metadata");
  const metadata = JSON.parse(rawMetadata);
  if (
    !metadata ||
    Array.isArray(metadata) ||
    typeof metadata !== "object" ||
    retainedSourceMediaDisclosable(metadata)
  )
    throw new Error(
      "Update this machine before copying source-bearing prints.",
    );
  if (nativeLegacyOffer) {
    const native = await nativeLegacyOffer(machine, filename, signal);
    if (
      native.members.length ||
      !DIGEST.test(native.output_sha256) ||
      !DIGEST.test(native.archive_identity_sha256) ||
      !Number.isSafeInteger(native.output_size_bytes) ||
      native.output_size_bytes <= 0 ||
      native.output_size_bytes > limit ||
      !mirrorMetadataMatches(rawMetadata, native.rawMetadata)
    )
      throw new Error("The older machine's output could not be verified.");
    return native;
  }
  const bytes = await bounded(
    await apiFetchTo(
      machine.target,
      `/api/gallery/image/${encodeURIComponent(filename)}`,
      { signal: signal ?? null },
    ),
    MAX_BYTES,
    signal,
  );
  const digest = await hash(bytes);
  return {
    archive_identity_sha256: await hash(
      new TextEncoder().encode(digest + rawMetadata),
    ),
    output_sha256: digest,
    output_size_bytes: bytes.length,
    metadata,
    rawMetadata,
    members: [],
  };
}
export async function offer(
  machine: CopyMachine,
  filename: string,
  signal?: AbortSignal,
  limit = MAX_BYTES,
  nativeLegacyOffer?: CopyOptions["nativeLegacyOffer"],
): Promise<CopyOffer> {
  let response: Awaited<ReturnType<typeof json>>;
  try {
    response = await json(
      machine.target,
      `/api/gallery/source-media/${encodeURIComponent(filename)}/transfer`,
      signal,
    );
  } catch (failure) {
    if (
      failure instanceof ApiError &&
      (failure.status === 404 || failure.status === 405)
    )
      return legacyCopyOffer(
        machine,
        filename,
        failure,
        signal,
        limit,
        nativeLegacyOffer,
      );
    throw failure;
  }
  const value = response.value as CopyOffer;
  if (
    !DIGEST.test(value.archive_identity_sha256) ||
    !DIGEST.test(value.output_sha256) ||
    !Number.isSafeInteger(value.output_size_bytes) ||
    value.output_size_bytes <= 0 ||
    value.output_size_bytes > limit ||
    !value.metadata ||
    typeof value.metadata !== "object" ||
    Array.isArray(value.metadata) ||
    !Array.isArray(value.members) ||
    value.members.length > 64
  )
    throw new Error(
      "The source machine cannot verify this print for copying. Update it or refresh the Library.",
    );
  let total = value.output_size_bytes;
  const slots = new Set<string>();
  for (const member of value.members) {
    const slot = JSON.stringify([member.role, member.position]);
    const indexed =
      /^(?:identity_images|edit_images|references|keyframes)$/.test(
        member.role,
      );
    const role =
      /^(?:source_image|identity_image|identity_images|edit_images|references|mask_image|control_image|audio_file|audio_file_path|source_video|source_video_path|extend_video|extend_video_path|keyframes|matting_processed_source_image|matting_processed_references|stage_source:(?:0|[1-9][0-9]*))$/.test(
        member.role,
      );
    const position = indexed
      ? /^item:(?:0|[1-9][0-9]*)$/.test(member.position)
      : member.role === "matting_processed_references"
        ? ["front", "left", "back", "right"].includes(member.position)
        : member.role.startsWith("stage_source:")
          ? member.position === "authored-source"
          : member.position === "scalar";
    if (
      !role ||
      !position ||
      !["memory", "private_staging"].includes(member.sink) ||
      !member.member_id ||
      !DIGEST.test(member.sha256) ||
      !Number.isSafeInteger(member.size_bytes) ||
      member.size_bytes <= 0 ||
      slots.has(slot)
    )
      throw new Error(
        "Retained source-media inventory is incomplete or invalid.",
      );
    slots.add(slot);
    total += member.size_bytes;
  }
  if (total > limit)
    throw new Error(
      "Copy is limited to 64 MiB per print including source media.",
    );
  return { ...value, rawMetadata: rawJsonProperty(response.text, "metadata") };
}
export const hash = async (bytes: Uint8Array<ArrayBuffer>) =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((v) => v.toString(16).padStart(2, "0"))
    .join("");
export function sameOutput(a: CopyOffer, b: CopyOffer, embedded?: string) {
  return (
    a.output_sha256 === b.output_sha256 &&
    a.output_size_bytes === b.output_size_bytes &&
    mirrorMetadataMatches(a.rawMetadata, b.rawMetadata, embedded)
  );
}
