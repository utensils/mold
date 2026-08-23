/**
 * "Can the machine that published this row actually run it?" — one policy for
 * web, desktop, and iPhone.
 *
 * `/api/models[].runtime_available` has always answered that, and #1276 added
 * `runtime_unavailable_reason` beside it because the answer has three
 * different causes with three different remedies: mold has no engine arm for
 * the checkpoint's weight layout, the task partition has no qualified runtime
 * on any released build, or this particular binary was compiled without the
 * engine. The server owns all three sentences (`mold_core`'s
 * `minimax_h3::RuntimeUnavailableReason`), so a surface renders what it is
 * told and never restates a family rule of its own.
 *
 * The reason a client needs this BEFORE a download: the affected checkpoints
 * are 21-42 GB, and the only honest refusal used to arrive at submit time.
 */

/** The `/api/models` fields this policy reads. */
export interface ModelRuntimeRow {
  runtime_available?: boolean | null;
  runtime_unavailable_reason?: string | null;
}

export interface ModelRuntimeNotice {
  message: string;
  /** True when the server named the obstacle; false for the fallback below. */
  fromServer: boolean;
}

/**
 * Wording for a server that reports `runtime_available: false` and predates
 * the reason field. It must stay true of every case, so it names no cause.
 */
export const RUNTIME_UNAVAILABLE_FALLBACK =
  "This machine cannot run this model. It downloads, verifies, and can be removed normally; only generation is unavailable.";

/** Short label for a badge or chip. Deliberately not the whole sentence. */
export const RUNTIME_UNAVAILABLE_BADGE = "Download only";

/**
 * `runtime_available !== false` is the compatibility contract: an older
 * server omits the field entirely, and absence has always meant runnable.
 */
export function isModelRuntimeUnavailable(
  row: ModelRuntimeRow | null | undefined,
): boolean {
  return row?.runtime_available === false;
}

/** The inline note for one row, or `null` when the row runs here. */
export function modelRuntimeNotice(
  row: ModelRuntimeRow | null | undefined,
): ModelRuntimeNotice | null {
  if (!isModelRuntimeUnavailable(row)) return null;
  const reason = row?.runtime_unavailable_reason;
  if (typeof reason === "string" && reason.trim().length > 0) {
    return { message: reason.trim(), fromServer: true };
  }
  return { message: RUNTIME_UNAVAILABLE_FALLBACK, fromServer: false };
}

/**
 * The pre-download answer for a Discover row.
 *
 * A not-yet-installed manifest still appears in `/api/models` (that is where
 * Discover's manifest entries are synthesized from), so its runtime answer is
 * already on the client — it just was not being rendered until the model was
 * installed. Matching is on the row's own `name`, the identity every request
 * addresses the model by; an id that matches nothing is simply unknown and
 * yields `null` rather than a guess.
 */
export function modelRuntimeNoticeForId(
  id: string | null | undefined,
  rows:
    readonly (ModelRuntimeRow & { name?: string | null })[] | null | undefined,
): ModelRuntimeNotice | null {
  const row = findRuntimeRow(id, rows);
  return row ? modelRuntimeNotice(row) : null;
}

function findRuntimeRow(
  id: string | null | undefined,
  rows:
    readonly (ModelRuntimeRow & { name?: string | null })[] | null | undefined,
): (ModelRuntimeRow & { name?: string | null }) | null {
  const wanted = id?.trim();
  if (!wanted || !rows) return null;
  return rows.find((candidate) => candidate.name?.trim() === wanted) ?? null;
}

/**
 * The same answer for a fleet, which is what a Discover row actually needs:
 * Pull can target any connected machine, so a row is "download only" ONLY
 * when every machine that has listed it says so. One reachable machine that
 * can run it makes the badge wrong — and a machine whose `/api/models` has
 * not been read is not evidence of anything, exactly as
 * `planModelInstall` treats an unread inventory.
 *
 * `hostRows` is one entry per machine, in whatever order the surface holds
 * them; the reported sentence is the first unavailable one, so a homogeneous
 * fleet reports its single real obstacle.
 */
export function modelRuntimeNoticeAcrossHosts(
  id: string | null | undefined,
  hostRows: readonly (
    readonly (ModelRuntimeRow & { name?: string | null })[] | null | undefined
  )[],
): ModelRuntimeNotice | null {
  let unavailable: ModelRuntimeNotice | null = null;
  let listed = false;
  for (const rows of hostRows) {
    const row = findRuntimeRow(id, rows);
    if (!row) continue;
    listed = true;
    const notice = modelRuntimeNotice(row);
    // Some machine in the fleet can run it. Nothing to warn about.
    if (!notice) return null;
    unavailable ??= notice;
  }
  return listed ? unavailable : null;
}
