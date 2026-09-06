/**
 * One-shot cleanup of the storage scene-by-scene authoring left behind.
 *
 * The retired sequence draft store owned four localStorage keys and wrote its
 * clip and opening-image media into the SHARED `mold-generate-drafts` IndexedDB
 * database, keyed `sequence-opening-image` and `sequence-clip-<id>`. Deleting
 * the store deleted the only code that could ever read or free any of it, so a
 * returning user carries a dead draft and potentially many megabytes of orphan
 * base64 in their quota — forever, and in the same database the one-shot
 * composer still writes to.
 *
 * Nothing reads these values any more, so this cannot fix a crash. It exists
 * because a retirement that silently keeps the user's bytes is not a
 * retirement. It is idempotent, safe to call on every boot, and must never
 * throw: private mode, a blocked database, and a corrupt draft all mean the
 * same thing here — there is nothing to reclaim.
 */
import { deleteDraftMediaByPrefix } from "./draftMediaStore";

/**
 * The draft the retired store persisted, its two legacy predecessors, the
 * mobile mode flag, and web's own sequence-rail bookkeeping.
 */
const RETIRED_LOCAL_STORAGE_KEYS = [
  "mold.sequence.draft.v1",
  "mold.chain.draft.v2",
  "mold.composer.mode",
  "mold.mobile.create-mode.v1",
  "mold.create.tracked-sequences.v1",
  "mold.create.chain-job-host",
] as const;

/**
 * Every sequence media id shares this prefix — `sequence-opening-image` and
 * `sequence-clip-<clip id>`. The one-shot composer's own draft ids do not,
 * which is what makes a prefix sweep safe in the shared database.
 */
const RETIRED_MEDIA_PREFIX = "sequence-";

export interface RetiredSequenceStorage {
  /** localStorage keys actually removed. */
  keys: string[];
  /** IndexedDB draft media ids actually removed. */
  media: string[];
}

export async function retireSequenceStorage(): Promise<RetiredSequenceStorage> {
  const removed: RetiredSequenceStorage = { keys: [], media: [] };
  for (const key of RETIRED_LOCAL_STORAGE_KEYS) {
    try {
      if (localStorage.getItem(key) === null) continue;
      localStorage.removeItem(key);
      removed.keys.push(key);
    } catch {
      // Private mode, or storage disabled — there is nothing to reclaim.
    }
  }
  try {
    removed.media = await deleteDraftMediaByPrefix(RETIRED_MEDIA_PREFIX);
  } catch {
    // A blocked or corrupt database frees nothing; the keys above still went.
  }
  return removed;
}
