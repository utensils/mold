import type { GenerationReference } from "../lib/generationReferences";
import { minimaxH3TaskForModel } from "../lib/minimaxH3Authoring";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "./client";

const SHA256_PATTERN = /^[0-9a-f]{64}$/i;
const HTTP_TOKEN_PATTERN = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;
const SAFE_API_PATH_PATTERN = /^\/api\/[A-Za-z0-9/_-]+$/;
const MAX_SECRET_BYTES = 1_024;
const RESERVED_SECRET_HEADERS = new Set([
  "authorization",
  "connection",
  "content-length",
  "content-type",
  "cookie",
  "host",
  "origin",
  "proxy-authorization",
  "transfer-encoding",
  "x-api-key",
]);

export interface ReferenceUploadCapabilities {
  available: boolean;
  protocol_version: number;
  requires_api_key: boolean;
  session_path: string;
  upload_path: string;
  session_handle_header: string;
  upload_handle_header: string;
  max_file_bytes: number;
  max_session_bytes: number;
  session_ttl_ms: number;
}

export type ReferenceUploadRequest = Record<string, unknown> & {
  model: string;
  references?: GenerationReference[] | null;
};

/**
 * One ready, request-bound set of uploaded references. `request` is
 * intentionally non-enumerable so logging or serializing the lease itself
 * cannot accidentally disclose its one-use upload handles. The request is an
 * ephemeral generation payload and must never be persisted or retried; retry
 * creates a fresh lease from the original inline-media snapshot.
 */
export interface ReferenceUploadLease<T extends ReferenceUploadRequest> {
  readonly request: T;
  readonly expiresAtMs: number;
  readonly requestScopeSha256: string;
  cancel(): Promise<void>;
}

export class ReferenceUploadProtocolError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ReferenceUploadProtocolError";
  }
}

interface CapturedReference {
  oneBased: number;
  original: GenerationReference;
  descriptor: GenerationReference;
  byteLength: number | null;
  inlineData: string | null;
}

interface ReferenceUploadSession {
  instanceId: string;
  expiresAtMs: number;
  requestScopeSha256: string;
  sessionHandle: string;
  uploads: Map<number, string>;
}

function protocolError(code: string, message: string): never {
  throw new ReferenceUploadProtocolError(code, message);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function requirePositiveSafeInteger(value: unknown, field: string): number {
  if (!Number.isSafeInteger(value) || Number(value) <= 0) {
    protocolError(
      "REFERENCE_UPLOAD_CAPABILITY_INVALID",
      `The host advertised an invalid ${field}.`,
    );
  }
  return Number(value);
}

function validateApiPath(value: unknown, field: string): string {
  if (typeof value !== "string" || !SAFE_API_PATH_PATTERN.test(value)) {
    protocolError(
      "REFERENCE_UPLOAD_CAPABILITY_INVALID",
      `The host advertised an unsafe ${field}.`,
    );
  }
  return value;
}

function validateSecretHeader(value: unknown, field: string): string {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    value.length > 128 ||
    !HTTP_TOKEN_PATTERN.test(value) ||
    RESERVED_SECRET_HEADERS.has(value.toLowerCase())
  ) {
    protocolError(
      "REFERENCE_UPLOAD_CAPABILITY_INVALID",
      `The host advertised an unsafe ${field}.`,
    );
  }
  return value;
}

function validateCapabilities(
  value: ReferenceUploadCapabilities | null | undefined,
): ReferenceUploadCapabilities {
  if (
    !value ||
    value.available !== true ||
    value.protocol_version !== 1 ||
    value.requires_api_key !== true
  ) {
    protocolError(
      "REFERENCE_UPLOAD_UNAVAILABLE",
      "This host does not advertise the authenticated reference-upload protocol required by MiniMax H3.",
    );
  }
  const sessionPath = validateApiPath(value.session_path, "session path");
  const uploadPath = validateApiPath(value.upload_path, "upload path");
  const sessionHeader = validateSecretHeader(
    value.session_handle_header,
    "session-handle header",
  );
  const uploadHeader = validateSecretHeader(
    value.upload_handle_header,
    "upload-handle header",
  );
  if (
    sessionPath === uploadPath ||
    sessionHeader.toLowerCase() === uploadHeader.toLowerCase()
  ) {
    protocolError(
      "REFERENCE_UPLOAD_CAPABILITY_INVALID",
      "The host advertised overlapping reference-upload authorities.",
    );
  }
  const maxFileBytes = requirePositiveSafeInteger(
    value.max_file_bytes,
    "per-file byte limit",
  );
  const maxSessionBytes = requirePositiveSafeInteger(
    value.max_session_bytes,
    "session byte limit",
  );
  const sessionTtlMs = requirePositiveSafeInteger(
    value.session_ttl_ms,
    "session lifetime",
  );
  if (maxFileBytes > maxSessionBytes) {
    protocolError(
      "REFERENCE_UPLOAD_CAPABILITY_INVALID",
      "The host advertised a per-file limit larger than its session limit.",
    );
  }
  return {
    available: true,
    protocol_version: 1,
    requires_api_key: true,
    session_path: sessionPath,
    upload_path: uploadPath,
    session_handle_header: sessionHeader,
    upload_handle_header: uploadHeader,
    max_file_bytes: maxFileBytes,
    max_session_bytes: maxSessionBytes,
    session_ttl_ms: sessionTtlMs,
  };
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  throw signal.reason instanceof Error
    ? signal.reason
    : new DOMException("The operation was aborted.", "AbortError");
}

function validBase64Character(code: number): boolean {
  return (
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122) ||
    (code >= 48 && code <= 57) ||
    code === 43 ||
    code === 47
  );
}

/** Strict RFC 4648 standard-base64 length validation without allocating the
 * decoded payload. Mold's Rust wire decoder applies the same padded alphabet. */
function decodedBase64Length(value: string, reference: number): number {
  if (value.length === 0 || value.length % 4 !== 0) {
    protocolError(
      "REFERENCE_UPLOAD_MEDIA_INVALID",
      `Reference ${reference} does not contain valid base64 media.`,
    );
  }
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  for (let index = 0; index < value.length - padding; index += 1) {
    if (!validBase64Character(value.charCodeAt(index))) {
      protocolError(
        "REFERENCE_UPLOAD_MEDIA_INVALID",
        `Reference ${reference} does not contain valid base64 media.`,
      );
    }
  }
  for (let index = value.length - padding; index < value.length; index += 1) {
    if (value.charCodeAt(index) !== 61) {
      protocolError(
        "REFERENCE_UPLOAD_MEDIA_INVALID",
        `Reference ${reference} does not contain valid base64 media.`,
      );
    }
  }
  // A single remaining alphabet character cannot encode a valid final group.
  const finalAlphabetLength = value.length - padding;
  if (
    (padding === 2 && finalAlphabetLength % 4 !== 2) ||
    (padding === 1 && finalAlphabetLength % 4 !== 3)
  ) {
    protocolError(
      "REFERENCE_UPLOAD_MEDIA_INVALID",
      `Reference ${reference} does not contain valid base64 media.`,
    );
  }
  return (value.length / 4) * 3 - padding;
}

function decodeBase64(
  value: string,
  reference: number,
): Uint8Array<ArrayBuffer> {
  let binary: string;
  try {
    binary = globalThis.atob(value);
  } catch {
    protocolError(
      "REFERENCE_UPLOAD_MEDIA_INVALID",
      `Reference ${reference} does not contain valid base64 media.`,
    );
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

async function sha256(value: string, reference: number): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    protocolError(
      "REFERENCE_UPLOAD_HASH_UNAVAILABLE",
      "Secure reference hashing is unavailable on this device.",
    );
  }
  const bytes = decodeBase64(value, reference);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function cloneWireRecord(
  value: Record<string, unknown>,
): Record<string, unknown> {
  try {
    return JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
  } catch {
    protocolError(
      "REFERENCE_UPLOAD_REQUEST_INVALID",
      "The MiniMax H3 request cannot be serialized safely.",
    );
  }
}

function captureReferences(
  request: ReferenceUploadRequest,
  capabilities: ReferenceUploadCapabilities,
): CapturedReference[] {
  if (minimaxH3TaskForModel(request.model) !== "ref2va") {
    protocolError(
      "REFERENCE_UPLOAD_REQUEST_INVALID",
      "Reference uploads require an explicit MiniMax H3 Ref2VA model.",
    );
  }
  if (
    request.batch_size !== undefined &&
    request.batch_size !== null &&
    request.batch_size !== 1
  ) {
    protocolError(
      "REFERENCE_UPLOAD_REQUEST_INVALID",
      "MiniMax H3 reference uploads require a single-output request.",
    );
  }
  const references = request.references;
  if (!Array.isArray(references) || references.length === 0) {
    protocolError(
      "REFERENCE_UPLOAD_REQUEST_INVALID",
      "MiniMax H3 Ref2VA requires ordered references.",
    );
  }

  let sessionBytes = 0;
  const captured = references.map((reference, index) => {
    const oneBased = index + 1;
    if (
      !reference ||
      typeof reference !== "object" ||
      !["image", "video", "audio"].includes(reference.kind) ||
      typeof reference.mime_type !== "string" ||
      reference.mime_type.trim().length === 0
    ) {
      protocolError(
        "REFERENCE_UPLOAD_REQUEST_INVALID",
        `Reference ${oneBased} has an invalid media descriptor.`,
      );
    }
    const media = asRecord(reference.media);
    const authority = media?.authority;
    if (authority === "upload" || authority === "descriptor") {
      protocolError(
        "REFERENCE_UPLOAD_REQUEST_INVALID",
        `Reference ${oneBased} must be prepared from its original media authority.`,
      );
    }
    if (authority !== "inline" && authority !== "server_path") {
      protocolError(
        "REFERENCE_UPLOAD_REQUEST_INVALID",
        `Reference ${oneBased} has an unsupported media authority.`,
      );
    }
    const copy = {
      ...reference,
      provenance: { ...(reference.provenance ?? {}) },
      media: { ...media },
    } as GenerationReference;
    const descriptor = {
      ...copy,
      media: { authority: "descriptor" as const },
    } as GenerationReference;
    if (authority === "server_path") {
      const digest = reference.provenance?.sha256?.trim();
      if (!digest || !SHA256_PATTERN.test(digest)) {
        protocolError(
          "REFERENCE_UPLOAD_REQUEST_INVALID",
          `Reference ${oneBased} requires an exact SHA-256 digest.`,
        );
      }
      descriptor.provenance = {
        ...(descriptor.provenance ?? {}),
        sha256: digest.toLowerCase(),
      };
      copy.provenance = {
        ...(copy.provenance ?? {}),
        sha256: digest.toLowerCase(),
      };
      return {
        oneBased,
        original: copy,
        descriptor,
        byteLength: null,
        inlineData: null,
      };
    }

    const data = media?.data;
    if (typeof data !== "string") {
      protocolError(
        "REFERENCE_UPLOAD_MEDIA_INVALID",
        `Reference ${oneBased} has no inline media bytes.`,
      );
    }
    // Reject obviously over-limit values before scanning or decoding them.
    const maximumEncodedLength = Math.ceil(capabilities.max_file_bytes / 3) * 4;
    if (data.length > maximumEncodedLength) {
      protocolError(
        "REFERENCE_UPLOAD_TOO_LARGE",
        `Reference ${oneBased} exceeds this host's per-file upload limit.`,
      );
    }
    const byteLength = decodedBase64Length(data, oneBased);
    if (byteLength === 0 || byteLength > capabilities.max_file_bytes) {
      protocolError(
        "REFERENCE_UPLOAD_TOO_LARGE",
        `Reference ${oneBased} exceeds this host's per-file upload limit.`,
      );
    }
    sessionBytes += byteLength;
    if (sessionBytes > capabilities.max_session_bytes) {
      protocolError(
        "REFERENCE_UPLOAD_TOO_LARGE",
        "The ordered references exceed this host's upload-session limit.",
      );
    }
    return {
      oneBased,
      original: copy,
      descriptor,
      byteLength,
      inlineData: data,
    };
  });
  if (!captured.some((entry) => entry.inlineData !== null)) {
    protocolError(
      "REFERENCE_UPLOAD_REQUEST_INVALID",
      "The request has no inline reference media to upload.",
    );
  }
  return captured;
}

async function bindReferenceDigests(
  captured: CapturedReference[],
  signal: AbortSignal | undefined,
): Promise<void> {
  for (const entry of captured) {
    throwIfAborted(signal);
    if (entry.inlineData === null) continue;
    const digest = await sha256(entry.inlineData, entry.oneBased);
    const declared = entry.original.provenance?.sha256?.trim().toLowerCase();
    if (declared && declared !== digest) {
      protocolError(
        "REFERENCE_UPLOAD_DIGEST_MISMATCH",
        `Reference ${entry.oneBased} does not match its declared SHA-256 digest.`,
      );
    }
    const provenance = {
      ...(entry.original.provenance ?? {}),
      sha256: digest,
    };
    entry.original = { ...entry.original, provenance } as GenerationReference;
    entry.descriptor = {
      ...entry.descriptor,
      provenance,
    } as GenerationReference;
  }
}

function validSecret(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= MAX_SECRET_BYTES &&
    /^[A-Za-z0-9._~-]+$/.test(value)
  );
}

function parseSession(
  value: unknown,
  expectedInstanceId: string,
  expectedReferences: readonly number[],
  now: number,
): ReferenceUploadSession {
  const row = asRecord(value);
  if (!row || row.instance_id !== expectedInstanceId) {
    protocolError(
      "REFERENCE_UPLOAD_INSTANCE_MISMATCH",
      "The reference-upload session came from a different Mold instance.",
    );
  }
  if (
    !Number.isSafeInteger(row.expires_at_ms) ||
    Number(row.expires_at_ms) <= now
  ) {
    protocolError(
      "REFERENCE_UPLOAD_SESSION_INVALID",
      "The host returned an expired reference-upload session.",
    );
  }
  if (
    typeof row.request_scope_sha256 !== "string" ||
    !SHA256_PATTERN.test(row.request_scope_sha256)
  ) {
    protocolError(
      "REFERENCE_UPLOAD_SESSION_INVALID",
      "The host returned an invalid request-scope digest.",
    );
  }
  if (!validSecret(row.session_handle) || !Array.isArray(row.uploads)) {
    protocolError(
      "REFERENCE_UPLOAD_SESSION_INVALID",
      "The host returned an invalid reference-upload authority.",
    );
  }
  const expected = new Set(expectedReferences);
  const uploads = new Map<number, string>();
  const handles = new Set<string>();
  for (const value of row.uploads) {
    const slot = asRecord(value);
    const reference = slot?.reference;
    const handle = slot?.handle;
    if (
      !Number.isSafeInteger(reference) ||
      !expected.has(Number(reference)) ||
      !validSecret(handle) ||
      uploads.has(Number(reference)) ||
      handles.has(handle)
    ) {
      protocolError(
        "REFERENCE_UPLOAD_SESSION_INVALID",
        "The host returned invalid or duplicate reference-upload slots.",
      );
    }
    uploads.set(Number(reference), handle);
    handles.add(handle);
  }
  if (uploads.size !== expected.size) {
    protocolError(
      "REFERENCE_UPLOAD_SESSION_INVALID",
      "The host did not return every requested reference-upload slot.",
    );
  }
  return {
    instanceId: expectedInstanceId,
    expiresAtMs: Number(row.expires_at_ms),
    requestScopeSha256: row.request_scope_sha256.toLowerCase(),
    sessionHandle: row.session_handle,
    uploads,
  };
}

function normalizedOptional(value: unknown): unknown {
  return value ?? null;
}

function metadataMatchesReference(
  metadata: Record<string, unknown>,
  reference: GenerationReference,
  oneBased: number,
): boolean {
  if (
    metadata.index !== oneBased ||
    metadata.kind !== reference.kind ||
    metadata.mime_type !== reference.mime_type ||
    String(metadata.sha256 ?? "").toLowerCase() !==
      reference.provenance?.sha256?.toLowerCase()
  ) {
    return false;
  }
  if (reference.kind === "image") {
    return (
      metadata.width === reference.width && metadata.height === reference.height
    );
  }
  if (reference.kind === "audio") {
    return (
      metadata.duration_ms === reference.duration_ms &&
      metadata.sample_rate === reference.sample_rate &&
      metadata.channels === reference.channels &&
      normalizedOptional(metadata.sample_count) ===
        normalizedOptional(reference.sample_count)
    );
  }
  return (
    metadata.width === reference.width &&
    metadata.height === reference.height &&
    normalizedOptional(metadata.frame_count) ===
      normalizedOptional(reference.frame_count) &&
    metadata.duration_ms === reference.duration_ms &&
    metadata.fps === reference.fps &&
    Boolean(metadata.has_audio) === Boolean(reference.has_audio) &&
    normalizedOptional(metadata.audio_duration_ms) ===
      normalizedOptional(reference.audio_duration_ms) &&
    normalizedOptional(metadata.audio_sample_count) ===
      normalizedOptional(reference.audio_sample_count) &&
    normalizedOptional(metadata.audio_sample_rate) ===
      normalizedOptional(reference.audio_sample_rate) &&
    normalizedOptional(metadata.audio_channels) ===
      normalizedOptional(reference.audio_channels)
  );
}

function validateUploadComplete(
  value: unknown,
  expectedInstanceId: string,
  entry: CapturedReference,
): void {
  const row = asRecord(value);
  const metadata = asRecord(row?.metadata);
  if (
    !row ||
    row.instance_id !== expectedInstanceId ||
    row.reference !== entry.oneBased ||
    !metadata ||
    !metadataMatchesReference(metadata, entry.descriptor, entry.oneBased)
  ) {
    protocolError(
      "REFERENCE_UPLOAD_RESPONSE_MISMATCH",
      `The host returned mismatched metadata for reference ${entry.oneBased}.`,
    );
  }
}

/** Create, fill, and validate one request-bound reference-upload session.
 * Browser fetch supplies Content-Length for each Blob; this deliberately does
 * not attempt to set that forbidden request header in JavaScript. */
export async function prepareReferenceUploads<
  T extends ReferenceUploadRequest,
>(options: {
  target: ApiTarget;
  expectedInstanceId: string;
  capabilities: ReferenceUploadCapabilities | null | undefined;
  request: T;
  signal?: AbortSignal;
  now?: () => number;
}): Promise<ReferenceUploadLease<T>> {
  throwIfAborted(options.signal);
  const capabilities = validateCapabilities(options.capabilities);
  if (!options.target.apiKey?.trim()) {
    protocolError(
      "REFERENCE_UPLOAD_AUTH_REQUIRED",
      "Reference uploads require an API-key-authenticated Mold host.",
    );
  }
  if (!options.expectedInstanceId.trim()) {
    protocolError(
      "REFERENCE_UPLOAD_INSTANCE_REQUIRED",
      "Reference uploads require an exact Mold instance identity.",
    );
  }
  const captured = captureReferences(options.request, capabilities);
  await bindReferenceDigests(captured, options.signal);
  throwIfAborted(options.signal);

  const requestFields = { ...options.request };
  delete requestFields.references;
  const scopedRequest = cloneWireRecord({
    ...requestFields,
    references: captured.map((entry) => entry.descriptor),
  }) as T;
  const uploadReferences = captured
    .filter((entry) => entry.inlineData !== null)
    .map((entry) => entry.oneBased);
  const signal = options.signal ? { signal: options.signal } : {};
  const sessionValue = await apiJsonTo<unknown>(
    options.target,
    capabilities.session_path,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        request: scopedRequest,
        upload_references: uploadReferences,
      }),
      ...signal,
    },
  );
  let session: ReferenceUploadSession;
  try {
    session = parseSession(
      sessionValue,
      options.expectedInstanceId,
      uploadReferences,
      (options.now ?? Date.now)(),
    );
  } catch (error) {
    const candidateHandle = asRecord(sessionValue)?.session_handle;
    if (validSecret(candidateHandle)) {
      try {
        await apiFetchTo(options.target, capabilities.session_path, {
          method: "DELETE",
          headers: {
            [capabilities.session_handle_header]: candidateHandle,
          },
        });
      } catch {
        // Invalid session responses cannot be trusted further; TTL remains the
        // final bound if this best-effort cleanup is also rejected.
      }
    }
    throw error;
  }
  let cancelPromise: Promise<void> | null = null;
  const cancel = (): Promise<void> => {
    cancelPromise ??= apiFetchTo(options.target, capabilities.session_path, {
      method: "DELETE",
      headers: {
        [capabilities.session_handle_header]: session.sessionHandle,
      },
    }).then(() => undefined);
    return cancelPromise;
  };

  try {
    for (const entry of captured) {
      if (entry.inlineData === null || entry.byteLength === null) continue;
      throwIfAborted(options.signal);
      const handle = session.uploads.get(entry.oneBased);
      if (!handle) {
        protocolError(
          "REFERENCE_UPLOAD_SESSION_INVALID",
          "The host omitted a required reference-upload slot.",
        );
      }
      const bytes = decodeBase64(entry.inlineData, entry.oneBased);
      if (bytes.byteLength !== entry.byteLength) {
        protocolError(
          "REFERENCE_UPLOAD_MEDIA_INVALID",
          `Reference ${entry.oneBased} changed while preparing its upload.`,
        );
      }
      const complete = await apiJsonTo<unknown>(
        options.target,
        capabilities.upload_path,
        {
          method: "PUT",
          headers: {
            "content-type": entry.original.mime_type,
            [capabilities.upload_handle_header]: handle,
          },
          body: new Blob([bytes], { type: entry.original.mime_type }),
          ...signal,
        },
      );
      validateUploadComplete(complete, options.expectedInstanceId, entry);
    }
  } catch (error) {
    try {
      await cancel();
    } catch {
      // Preserve the first protocol/transport failure. The server TTL remains
      // a final cleanup bound if best-effort cancellation cannot reach it.
    }
    throw error;
  }

  const finalRequest = cloneWireRecord({
    ...scopedRequest,
    references: captured.map((entry) => {
      if (entry.inlineData === null) return entry.original;
      const handle = session.uploads.get(entry.oneBased);
      if (!handle) {
        protocolError(
          "REFERENCE_UPLOAD_SESSION_INVALID",
          "The host omitted a required reference-upload slot.",
        );
      }
      return {
        ...entry.descriptor,
        media: { authority: "upload", handle },
      };
    }),
  }) as T;

  const lease = {
    expiresAtMs: session.expiresAtMs,
    requestScopeSha256: session.requestScopeSha256,
    cancel,
  } as Omit<ReferenceUploadLease<T>, "request"> & { request?: T };
  Object.defineProperty(lease, "request", {
    value: finalRequest,
    enumerable: false,
    configurable: false,
    writable: false,
  });
  return Object.freeze(lease) as ReferenceUploadLease<T>;
}

export function requestNeedsReferenceUpload(
  request: ReferenceUploadRequest,
): boolean {
  return (
    minimaxH3TaskForModel(request.model) === "ref2va" &&
    Array.isArray(request.references) &&
    request.references.some(
      (reference) => reference.media.authority === "inline",
    )
  );
}
