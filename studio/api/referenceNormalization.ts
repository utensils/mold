import { protectLegacyGenerateWire } from "../lib/legacyGenerateWire";
import { sha256 } from "@noble/hashes/sha2.js";
import { apiJsonTo, apiFetchTo, type ApiTarget } from "./client";
import {
  canonicalDescriptorFromMetadata,
  canonicalVideoDescriptorFromMetadata,
  validateReferenceUploadCapabilities,
  type ReferenceUploadCapabilities,
  type ReferenceUploadLease,
  type ReferenceUploadRequest,
} from "./referenceUploads";
import type {
  FinalizeReceipt,
  NormalizationReceipt,
  ReferenceCandidate,
  VideoNormalizationCapabilities,
} from "./referenceNormalizationTypes";
import type { GenerationReference } from "../lib/generationReferences";
const DIGEST = /^[a-f0-9]{64}$/;
const SECRET = /^[A-Za-z0-9._~-]{1,1024}$/;
const MAX_FILE = 256 * 1024 * 1024;
function fail(): never {
  throw new Error(
    "The machine returned invalid reference normalization authority or metadata.",
  );
}
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail();
  return value as Record<string, unknown>;
}
function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
function copy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
function stable(value: unknown): string {
  return JSON.stringify(
    value && typeof value === "object"
      ? Array.isArray(value)
        ? value.map((v) => JSON.parse(stable(v)))
        : Object.fromEntries(
            Object.entries(value)
              .filter(([, v]) => v !== undefined)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([k, v]) => [k, JSON.parse(stable(v))]),
          )
      : value,
  );
}
/** Rust serde omits false/None fields. Compare the same typed descriptor
 * semantics while retaining exact required facts, kinds and authorities. */
function descriptorWire(value: unknown): unknown {
  const descriptor = { ...record(value) };
  for (const key of [
    "frame_count",
    "audio_duration_ms",
    "audio_sample_count",
    "audio_sample_rate",
    "audio_channels",
    "sample_count",
  ])
    if (descriptor[key] == null) delete descriptor[key];
  if (descriptor.kind === "video" && descriptor.has_audio === false)
    delete descriptor.has_audio;
  if (
    descriptor.provenance &&
    typeof descriptor.provenance === "object" &&
    !Array.isArray(descriptor.provenance)
  ) {
    const provenance = {
      ...(descriptor.provenance as Record<string, unknown>),
    };
    for (const key of ["name", "sha256", "crop"])
      if (provenance[key] == null) delete provenance[key];
    descriptor.provenance = provenance;
  }
  return descriptor;
}
function freezeWire<T>(value: T): T {
  if (value && typeof value === "object") {
    for (const nested of Object.values(value)) freezeWire(nested);
    Object.freeze(value);
  }
  return value;
}
function positive(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) > 0;
}
function safePath(value: string): boolean {
  return /^\/api\/[A-Za-z0-9/_-]+$/.test(value);
}
function validateNormalization(
  value: VideoNormalizationCapabilities | null | undefined,
): VideoNormalizationCapabilities {
  if (
    !value?.available ||
    value.protocol_version !== 1 ||
    !value.input_roles?.includes("reference_video") ||
    !safePath(value.candidate_session_path) ||
    !safePath(value.finalize_path) ||
    value.candidate_session_path === value.finalize_path ||
    value.candidate_digest_header !== "x-mold-reference-candidate-sha256" ||
    value.canonical_mime_type !== "video/mp4" ||
    !positive(value.max_original_file_bytes) ||
    !positive(value.max_canonical_file_bytes) ||
    !positive(value.max_session_bytes)
  )
    fail();
  return copy(value);
}
async function digestBlob(blob: Blob, current: () => void): Promise<string> {
  const hash = sha256.create();
  for (let offset = 0; offset < blob.size; offset += 4 * 1024 * 1024) {
    current();
    hash.update(
      new Uint8Array(
        await blob.slice(offset, offset + 4 * 1024 * 1024).arrayBuffer(),
      ),
    );
  }
  current();
  return hex(hash.digest());
}
/** Fresh typed-reference ingress only. No generation authority is returned until
 * all canonical descriptors and the exact receipt set cross the finalize barrier. */
export async function prepareNormalizedReferenceUploads<
  T extends ReferenceUploadRequest,
>(options: {
  target: ApiTarget;
  expectedInstanceId: string;
  capabilities: ReferenceUploadCapabilities;
  request: Omit<T, "references">;
  candidates: readonly ReferenceCandidate[];
  uploadBodies: ReadonlyMap<number, Blob>;
  current(): boolean;
  signal?: AbortSignal;
}): Promise<ReferenceUploadLease<T>> {
  // Capture every mutable authority and request field before the first await.
  const target = { ...options.target },
    expectedInstanceId = options.expectedInstanceId;
  const base = validateReferenceUploadCapabilities(copy(options.capabilities));
  const cap = validateNormalization(options.capabilities.video_normalization);
  const request = copy(options.request),
    candidates = copy(options.candidates),
    bodies = new Map(options.uploadBodies);
  const signal = options.signal,
    isCurrent = options.current;
  const current = () => {
    signal?.throwIfAborted();
    if (!isCurrent())
      throw new Error(
        "The reference normalization request is no longer current.",
      );
  };
  current();
  if (
    !target.apiKey?.trim() ||
    !expectedInstanceId.trim() ||
    Object.hasOwn(request, "references") ||
    !candidates.length ||
    !candidates.some((c) => c.type === "fresh_video")
  )
    fail();
  const uploaded = candidates.filter(
    (c) => c.type === "fresh_video" || c.upload,
  );
  if (uploaded.length !== bodies.size) fail();
  let total = 0;
  for (const [position, candidate] of candidates.entries()) {
    if (candidate.index !== position + 1) fail();
    if (candidate.type === "canonical" && !candidate.upload) {
      if (
        candidate.reference.media.authority !== "server_path" ||
        !DIGEST.test(candidate.reference.provenance?.sha256 ?? "")
      )
        fail();
      continue;
    }
    const body = bodies.get(candidate.index);
    const original =
      candidate.type === "fresh_video"
        ? candidate.original
        : {
            byte_length: candidate.byte_length,
            sha256: candidate.reference.provenance?.sha256,
          };
    if (
      !body ||
      !positive(original.byte_length) ||
      original.byte_length !== body.size ||
      body.size >
        Math.min(MAX_FILE, base.max_file_bytes, cap.max_original_file_bytes) ||
      !DIGEST.test(original.sha256 ?? "")
    )
      fail();
    if (candidate.type === "fresh_video") {
      if (
        candidate.kind !== "video" ||
        !cap.accepted_mime_types.includes(candidate.original.mime_type)
      )
        fail();
    } else if (candidate.reference.media.authority !== "descriptor") fail();
    total += body.size;
    if (total > Math.min(base.max_session_bytes, cap.max_session_bytes)) fail();
  }
  const frozenBody = JSON.stringify({
    protocol_version: 1,
    request: protectLegacyGenerateWire(request),
    reference_candidates: candidates,
  });
  const fingerprint = hex(
    sha256(
      new TextEncoder().encode(
        `mold-reference-normalization-candidate-v1\0${frozenBody}`,
      ),
    ),
  );
  for (const candidate of uploaded) {
    const expected =
      candidate.type === "fresh_video"
        ? candidate.original.sha256
        : candidate.reference.provenance?.sha256;
    if ((await digestBlob(bodies.get(candidate.index)!, current)) !== expected)
      fail();
  }
  current();
  const initSignal = signal ? { signal } : {};
  const sessionValue = await apiJsonTo<unknown>(
    target,
    cap.candidate_session_path,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        [cap.candidate_digest_header]: fingerprint,
      },
      body: frozenBody,
      ...initSignal,
    },
  );
  const session = record(sessionValue),
    handle = session.session_handle;
  let cancellation: Promise<void> | undefined;
  const cancel = () =>
    (cancellation ??=
      typeof handle === "string" && SECRET.test(handle)
        ? apiFetchTo(target, base.session_path, {
            method: "DELETE",
            headers: { [base.session_handle_header]: handle },
          }).then(() => undefined)
        : Promise.resolve());
  try {
    current();
    if (
      session.protocol_version !== 1 ||
      session.instance_id !== expectedInstanceId ||
      session.candidate_scope_sha256 !== fingerprint ||
      session.session_complete !== false ||
      typeof session.session_id !== "string" ||
      !SECRET.test(session.session_id) ||
      typeof handle !== "string" ||
      !SECRET.test(handle) ||
      !positive(session.expires_at_ms) ||
      session.expires_at_ms <= Date.now() ||
      !Array.isArray(session.uploads)
    )
      fail();
    const sessionId = session.session_id,
      expires = session.expires_at_ms;
    const handles = new Map<number, string>(),
      seen = new Set<string>();
    for (const item of session.uploads) {
      const slot = record(item);
      if (
        !positive(slot.reference) ||
        !uploaded.some((c) => c.index === slot.reference) ||
        handles.has(slot.reference) ||
        typeof slot.handle !== "string" ||
        !SECRET.test(slot.handle) ||
        seen.has(slot.handle)
      )
        fail();
      handles.set(slot.reference, slot.handle);
      seen.add(slot.handle);
    }
    if (handles.size !== uploaded.length) fail();
    const bound = (value: unknown): Record<string, unknown> => {
      const row = record(value);
      if (
        row.protocol_version !== 1 ||
        row.instance_id !== expectedInstanceId ||
        row.session_id !== sessionId ||
        row.candidate_scope_sha256 !== fingerprint
      )
        fail();
      return row;
    };
    const descriptors = new Map<number, GenerationReference>(),
      receipts: FinalizeReceipt[] = [];
    for (const candidate of candidates) {
      if (candidate.type === "canonical" && !candidate.upload) {
        descriptors.set(candidate.index, {
          ...candidate.reference,
          media: { authority: "descriptor" },
        });
        continue;
      }
      current();
      const original =
        candidate.type === "fresh_video"
          ? candidate.original
          : {
              sha256: candidate.reference.provenance!.sha256!,
              byte_length: candidate.byte_length!,
              mime_type: candidate.reference.mime_type,
              name: candidate.reference.provenance?.name ?? null,
            };
      const body = bodies.get(candidate.index)!;
      const row = bound(
        await apiJsonTo<unknown>(target, base.upload_path, {
          method: "PUT",
          headers: {
            "content-type": original.mime_type,
            [base.upload_handle_header]: handles.get(candidate.index)!,
          },
          body,
          ...initSignal,
        }),
      );
      current();
      if (row.reference !== candidate.index || row.session_complete !== false)
        fail();
      const receipt = record(row.receipt) as unknown as NormalizationReceipt;
      const metadata = record(row.metadata);
      const kind =
        candidate.type === "fresh_video" ? "video" : candidate.reference.kind;
      if (
        receipt.version !== 1 ||
        receipt.kind !== kind ||
        receipt.original_sha256 !== original.sha256 ||
        receipt.original_byte_length !== original.byte_length ||
        receipt.original_mime_type !== original.mime_type ||
        !DIGEST.test(receipt.canonical_sha256) ||
        !positive(receipt.canonical_byte_length) ||
        receipt.canonical_byte_length >
          Math.min(
            MAX_FILE,
            base.max_file_bytes,
            cap.max_canonical_file_bytes,
          ) ||
        typeof receipt.converted !== "boolean" ||
        metadata.index !== candidate.index ||
        metadata.kind !== kind ||
        metadata.sha256 !== receipt.canonical_sha256 ||
        metadata.mime_type !== receipt.canonical_mime_type ||
        (metadata.name ?? null) !== original.name
      )
        fail();
      if (
        (!receipt.converted || candidate.type === "canonical") &&
        (receipt.canonical_sha256 !== original.sha256 ||
          receipt.canonical_byte_length !== original.byte_length)
      )
        fail();
      if (candidate.type === "canonical" && receipt.converted) fail();
      const provenance = {
        name: original.name,
        sha256: receipt.canonical_sha256,
      };
      const descriptor =
        candidate.type === "fresh_video"
          ? canonicalVideoDescriptorFromMetadata(metadata, provenance)
          : canonicalDescriptorFromMetadata(metadata, {
              oneBased: candidate.index,
              original: candidate.reference,
            });
      if (!descriptor) fail();
      descriptors.set(candidate.index, descriptor);
      receipts.push({
        reference: candidate.index,
        kind,
        original_sha256: receipt.original_sha256,
        canonical_sha256: receipt.canonical_sha256,
      });
    }
    current();
    const finalized = bound(
      await apiJsonTo<unknown>(target, cap.finalize_path, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          [base.session_handle_header]: handle,
        },
        body: JSON.stringify({
          protocol_version: 1,
          instance_id: expectedInstanceId,
          session_id: sessionId,
          candidate_scope_sha256: fingerprint,
          receipts,
        }),
        ...initSignal,
      }),
    );
    current();
    if (
      finalized.session_complete !== true ||
      typeof finalized.request_scope_sha256 !== "string" ||
      !DIGEST.test(finalized.request_scope_sha256) ||
      stable(finalized.receipts) !== stable(receipts) ||
      !Array.isArray(finalized.references) ||
      finalized.references.length !== candidates.length ||
      expires <= Date.now()
    )
      fail();
    const references = candidates.map((candidate, index) => {
      const descriptor = descriptors.get(candidate.index)!;
      if (
        stable(descriptorWire((finalized.references as unknown[])[index])) !==
        stable(descriptorWire(descriptor))
      )
        fail();
      return {
        ...descriptor,
        media:
          candidate.type === "canonical" && !candidate.upload
            ? candidate.reference.media
            : {
                authority: "upload" as const,
                handle: handles.get(candidate.index)!,
              },
      };
    });
    const finalRequest = freezeWire({ ...request, references } as T);
    const lease = {
      expiresAtMs: expires,
      requestScopeSha256: finalized.request_scope_sha256,
      cancel,
    } as ReferenceUploadLease<T>;
    Object.defineProperty(lease, "request", {
      value: finalRequest,
      enumerable: false,
      writable: false,
      configurable: false,
    });
    return Object.freeze(lease);
  } catch (error) {
    await cancel().catch(() => undefined);
    throw error;
  }
}
