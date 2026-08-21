import {
  deleteDraftMedia,
  getDraftMedia,
  putDurableMedia,
  type DraftMediaRecord,
} from "./draftMediaStore";
import type { SourceFitPolicy } from "./sourceFit";

const RECORD_PREFIX = "generation-source:";
const INDEX_KEY = "mold.generation-source-media.v1";
const MAX_RECORDS = 64;

export interface GenerationSourceMedia extends DraftMediaRecord {
  draftId: string;
  base64: string;
  filename: string;
  kind?: "upload" | "gallery";
  width?: number | null;
  height?: number | null;
  mime?: string | null;
  /** How the source was mapped onto the canvas. Absent for records that are
   * never fitted at all — an identity photo travels untouched (#1224). */
  sourceFit?: SourceFitPolicy;
}

export interface GenerationSourceMediaInput {
  base64: string;
  filename: string;
  kind?: "upload" | "gallery";
  width?: number | null;
  height?: number | null;
  mime?: string | null;
  sourceFit?: SourceFitPolicy;
}

export interface GenerationSourceMediaPersistence {
  put(record: GenerationSourceMedia): Promise<boolean>;
  get(id: string): Promise<GenerationSourceMedia | null>;
  delete(id: string): Promise<void>;
}

const indexedDbPersistence: GenerationSourceMediaPersistence = {
  put: putDurableMedia,
  get: (id) => getDraftMedia<GenerationSourceMedia>(id),
  delete: deleteDraftMedia,
};

function recordId(sha256: string): string {
  return `${RECORD_PREFIX}${sha256.toLowerCase()}`;
}

export async function sha256HexOfBase64(base64: string): Promise<string> {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function readIndex(): string[] {
  if (typeof localStorage === "undefined") return [];
  try {
    const parsed = JSON.parse(localStorage.getItem(INDEX_KEY) ?? "[]");
    return Array.isArray(parsed)
      ? parsed.filter((value): value is string => typeof value === "string")
      : [];
  } catch {
    return [];
  }
}

function writeIndex(ids: readonly string[]): boolean {
  if (typeof localStorage === "undefined") return false;
  try {
    localStorage.setItem(INDEX_KEY, JSON.stringify(ids));
    return true;
  } catch {
    // Source recovery is best-effort and must never block a generation.
    return false;
  }
}

async function touchAndPrune(
  id: string,
  persistence: GenerationSourceMediaPersistence,
): Promise<boolean> {
  const ids = [id, ...readIndex().filter((candidate) => candidate !== id)];
  const stale = ids.slice(MAX_RECORDS);
  if (!writeIndex(ids.slice(0, MAX_RECORDS))) {
    // Never leave an unindexed large-media record behind: without the durable
    // LRU index there is no way to prove it will be pruned later.
    await persistence.delete(id).catch(() => undefined);
    return false;
  }
  await Promise.allSettled(
    stale.map((candidate) => persistence.delete(candidate)),
  );
  return true;
}

/**
 * Save the user's editable source under the digest of the effective bytes
 * that reached the server. OutputMetadata already records that digest, so no
 * payload or private path needs to enter gallery metadata. The record retains
 * the original dimensions/type and fit policy to rebuild the source well.
 */
export async function persistGenerationSourceMedia(
  effectiveBase64: string,
  source: GenerationSourceMediaInput,
  persistence: GenerationSourceMediaPersistence = indexedDbPersistence,
): Promise<string | null> {
  try {
    const sha256 = await sha256HexOfBase64(effectiveBase64);
    const id = recordId(sha256);
    const saved = await persistence.put({ ...source, draftId: id });
    if (!saved) return null;
    return (await touchAndPrune(id, persistence)) ? sha256 : null;
  } catch {
    return null;
  }
}

export async function restoreGenerationSourceMedia(
  effectiveSha256: string | null | undefined,
  persistence: GenerationSourceMediaPersistence = indexedDbPersistence,
): Promise<GenerationSourceMedia | null> {
  if (!effectiveSha256 || !/^[a-f\d]{64}$/i.test(effectiveSha256)) return null;
  const id = recordId(effectiveSha256);
  const media = await persistence.get(id).catch(() => null);
  if (!media?.base64) return null;
  return (await touchAndPrune(id, persistence)) ? media : null;
}

export const __testing__ = { INDEX_KEY, MAX_RECORDS, recordId };
