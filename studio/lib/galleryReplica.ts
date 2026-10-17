/*
 * Per-host gallery replica — the TypeScript peer of MoldClient's
 * `GalleryReplica.swift`, sharing `docs/contracts/gallery-changes-v1.json`.
 *
 * A replica holds one host's whole gallery (library AND trash, split by
 * `trashed_at`) keyed by filename, plus the opaque feed cursor and the
 * server instance that issued it. `pull()` brings it current:
 *
 *   - FEED (`capabilities.gallery.changes_feed.protocol_version == 1`):
 *     `GET /api/gallery/changes?since=<cursor>` pages, applied O(k) per page,
 *     looping while `more`. No cursor is a full snapshot; `reset: true`, a
 *     changed `instance_id`, or a fresh snapshot discards every row the pages
 *     do not re-send — deferred to the end of the snapshot, so a grid never
 *     flashes empty mid-rebuild.
 *   - LEGACY (feed absent, `503 GALLERY_FEED_NOT_READY`, `404`): the existing
 *     conditional listing(s), diffed by filename and deep row equality into
 *     the SAME delta shape. A 304 hands back the identical array, which is a
 *     no-op in O(1).
 *
 * Every pull returns a `ReplicaDelta` describing exactly how the replica's
 * state moved (upserted rows, removed names), or `null` when nothing did, so
 * callers never re-merge, re-sort or re-render on an idle poll. Pulls are
 * single-flight: any number of requests during a pull coalesce into exactly
 * ONE follow-up, which is how an SSE tickle that lands mid-pull is honoured
 * without a request storm.
 *
 * Framework-free and transport-injected: web, desktop (HTTP or the local
 * IPC) and the Tauri phone (with an IndexedDB persistence adapter) each bind
 * their own transport. Rows are treated as immutable snapshots.
 */

export const GALLERY_CHANGES_PROTOCOL_VERSION = 1;
/** The server default; it accepts up to 2000. */
export const GALLERY_CHANGES_PAGE_LIMIT = 500;
/** A misbehaving server must not spin a pull forever. */
const MAX_PAGES_PER_PULL = 10_000;

/** The minimum a row must carry to live in a replica. */
export interface GalleryReplicaRow {
  filename: string;
  timestamp: number;
  trashed_at?: number | null;
}

export interface GalleryChange<Row> {
  filename: string;
  removed: boolean;
  image?: Row | null;
}

/** `GET /api/gallery/changes` page (protocol 1). */
export interface GalleryChangesPage<Row> {
  protocol_version: number;
  instance_id: string;
  cursor: string;
  more: boolean;
  reset: boolean;
  changes: GalleryChange<Row>[];
}

export type GalleryFeedUnavailableReason =
  /** `503 GALLERY_FEED_NOT_READY` — the host is still reconciling. */
  | "not_ready"
  /** `404 GALLERY_FEED_UNSUPPORTED`, an older server, or another protocol. */
  | "unsupported"
  /** No feed authority right now (e.g. desktop's engine is Off). */
  | "offline";

/** Thrown by a transport's `changes` when the feed cannot answer THIS pull;
 * the replica answers from the legacy listing instead. Any other error is a
 * real failure and rejects the pull. */
export class GalleryFeedUnavailableError extends Error {
  constructor(public readonly reason: GalleryFeedUnavailableReason) {
    super(`Gallery change feed unavailable (${reason}).`);
    this.name = "GalleryFeedUnavailableError";
  }
}

export interface GalleryReplicaTransport<Row extends GalleryReplicaRow> {
  /** Whether to try the feed at all (the host's capability). */
  feedSupported(): boolean | Promise<boolean>;
  changes(
    since: string | null,
    limit: number,
  ): Promise<GalleryChangesPage<Row>>;
  /** `GET /api/gallery` (conditional). */
  legacyLibrary(): Promise<readonly Row[]>;
  /** `GET /api/gallery?view=trash`. Absent or `null` = the trash is not
   * listed (no trash, or the caller only reads the library); trash rows the
   * replica already holds are then preserved, never reported removed. */
  legacyTrash?: () => Promise<readonly Row[] | null>;
}

export interface ReplicaDelta<Row> {
  /** Rows that are new or changed, in their current shape (live or trashed). */
  upserted: Row[];
  /** Filenames the replica no longer holds at all. */
  removed: string[];
  /** The replica was rebuilt (a snapshot, `reset`, or a new instance). The
   * delta is still an exact diff, so callers may apply it like any other. */
  reset: boolean;
  /** The replica revision after this change. */
  revision: number;
  source: "feed" | "legacy" | "hydrate";
}

export interface PersistedGalleryReplica<Row> {
  version: 1;
  instanceId: string | null;
  /** Only ever a feed cursor whose rows are all in `rows`. */
  cursor: string | null;
  rows: Row[];
}

/** Optional durable store (the phone's IndexedDB). `save` runs only after a
 * pull that changed rows or the cursor, and receives the delta so an adapter
 * can write incrementally. */
export interface GalleryReplicaPersistence<Row> {
  load(): Promise<PersistedGalleryReplica<Row> | null>;
  save(
    state: PersistedGalleryReplica<Row>,
    delta: ReplicaDelta<Row> | null,
  ): Promise<void>;
}

export interface GalleryReplicaOptions<Row> {
  persistence?: GalleryReplicaPersistence<Row>;
  pageLimit?: number;
}

export type ReplicaListener<Row> = (delta: ReplicaDelta<Row>) => void;

/** Structural equality over JSON-shaped rows. */
export function galleryRowsEqual(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (
    typeof left !== "object" ||
    typeof right !== "object" ||
    left === null ||
    right === null
  )
    return false;
  if (Array.isArray(left)) {
    if (!Array.isArray(right) || left.length !== right.length) return false;
    for (let i = 0; i < left.length; i += 1)
      if (!galleryRowsEqual(left[i], right[i])) return false;
    return true;
  }
  if (Array.isArray(right)) return false;
  const a = left as Record<string, unknown>;
  const b = right as Record<string, unknown>;
  // `undefined` and an absent key are the same JSON.
  const keys = new Set<string>();
  for (const key of Object.keys(a)) if (a[key] !== undefined) keys.add(key);
  let rightCount = 0;
  for (const key of Object.keys(b)) {
    if (b[key] === undefined) continue;
    rightCount += 1;
    if (!keys.has(key)) return false;
  }
  if (rightCount !== keys.size) return false;
  for (const key of keys) if (!galleryRowsEqual(a[key], b[key])) return false;
  return true;
}

/** Library order: newest first, filename as the tie-break. */
export function compareLibraryRows(
  left: GalleryReplicaRow,
  right: GalleryReplicaRow,
): number {
  return (
    right.timestamp - left.timestamp ||
    (left.filename < right.filename
      ? -1
      : left.filename > right.filename
        ? 1
        : 0)
  );
}

/** Trash order: newest-DELETED first (the server's `view=trash` order). */
export function compareTrashRows(
  left: GalleryReplicaRow,
  right: GalleryReplicaRow,
): number {
  return (
    (right.trashed_at ?? right.timestamp) -
      (left.trashed_at ?? left.timestamp) || compareLibraryRows(left, right)
  );
}

export function isTrashedRow(row: GalleryReplicaRow): boolean {
  return row.trashed_at !== undefined && row.trashed_at !== null;
}

class FeedStalledError extends Error {
  constructor() {
    super("Gallery change feed did not advance its cursor.");
    this.name = "FeedStalledError";
  }
}

/** Accumulates the pre-change value of every name a pull touches, so the
 * final delta is an exact O(k) diff however many pages it took. */
class TouchLedger<Row> {
  readonly prior = new Map<string, Row | undefined>();
  /** Names whose library/trash membership flipped with no content change
   * (a trash-listing row without `trashed_at`): still a change. */
  readonly moved = new Set<string>();
  note(filename: string, value: Row | undefined) {
    if (!this.prior.has(filename)) this.prior.set(filename, value);
  }
}

export class GalleryReplica<Row extends GalleryReplicaRow> {
  private readonly byName = new Map<string, Row>();
  private cursorValue: string | null = null;
  private instance: string | null = null;
  private revisionValue = 0;
  private modeValue: "feed" | "legacy" | null = null;
  private trashFreshValue = false;
  /** Names the last successful `?view=trash` listing returned. A legacy
   * trash row need not carry `trashed_at` (older servers omit it), so the
   * partition is by PROVENANCE as well as by stamp; a feed row for the name
   * clears it (feed rows always say where they live). */
  private trashNames = new Set<string>();
  private lastLibrary: readonly Row[] | null = null;
  private lastTrash: readonly Row[] | null = null;
  private liveCache: { revision: number; rows: Row[] } | null = null;
  private trashCache: { revision: number; rows: Row[] } | null = null;
  private inflight: Promise<ReplicaDelta<Row> | null> | null = null;
  private followUp: Promise<ReplicaDelta<Row> | null> | null = null;
  private hydration: Promise<ReplicaDelta<Row> | null> | null = null;
  private readonly listeners = new Set<ReplicaListener<Row>>();
  private readonly pageLimit: number;
  private disposed = false;
  private settledAtValue = 0;

  constructor(
    private readonly transport: GalleryReplicaTransport<Row>,
    private readonly options: GalleryReplicaOptions<Row> = {},
  ) {
    this.pageLimit = Math.min(
      2000,
      Math.max(1, options.pageLimit ?? GALLERY_CHANGES_PAGE_LIMIT),
    );
  }

  get revision(): number {
    return this.revisionValue;
  }
  get cursor(): string | null {
    return this.cursorValue;
  }
  get instanceId(): string | null {
    return this.instance;
  }
  /** How the last successful pull answered (`null` before the first). */
  get mode(): "feed" | "legacy" | null {
    return this.modeValue;
  }
  /** The trash rows are authoritative as of the last pull: always after a
   * feed pull, after a legacy pull only when the trash was listed. */
  get trashFresh(): boolean {
    return this.trashFreshValue;
  }
  /** `Date.now()` when the last pull succeeded (0 = never). */
  get settledAt(): number {
    return this.settledAtValue;
  }
  /** A pull (or its follow-up) is running. */
  get pulling(): boolean {
    return this.inflight !== null || this.followUp !== null;
  }
  get size(): number {
    return this.byName.size;
  }
  get(filename: string): Row | undefined {
    return this.byName.get(filename);
  }
  rows(): ReadonlyMap<string, Row> {
    return this.byName;
  }

  /** Whether this print lives in the trash (stamped, or listed there). */
  isTrashed(filename: string): boolean {
    const row = this.byName.get(filename);
    return row !== undefined && this.inTrash(row);
  }

  private inTrash(row: Row): boolean {
    return isTrashedRow(row) || this.trashNames.has(row.filename);
  }

  /** Library rows, newest first. Memoized per revision. */
  live(): readonly Row[] {
    if (this.liveCache?.revision !== this.revisionValue) {
      const rows: Row[] = [];
      for (const row of this.byName.values())
        if (!this.inTrash(row)) rows.push(row);
      rows.sort(compareLibraryRows);
      this.liveCache = { revision: this.revisionValue, rows };
    }
    return this.liveCache.rows;
  }

  /** Trashed rows, newest-deleted first. Memoized per revision. */
  trash(): readonly Row[] {
    if (this.trashCache?.revision !== this.revisionValue) {
      const rows: Row[] = [];
      for (const row of this.byName.values())
        if (this.inTrash(row)) rows.push(row);
      rows.sort(compareTrashRows);
      this.trashCache = { revision: this.revisionValue, rows };
    }
    return this.trashCache.rows;
  }

  subscribe(listener: ReplicaListener<Row>): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Load persisted state once (a no-op without an adapter). `pull` awaits
   * it, so callers that want a saved grid to paint first call it directly. */
  hydrate(): Promise<ReplicaDelta<Row> | null> {
    if (!this.hydration) this.hydration = this.loadPersisted();
    return this.hydration;
  }

  /** Bring the replica current. Single-flight: a call during a pull
   * schedules exactly one follow-up pull, shared by every such caller. */
  pull(): Promise<ReplicaDelta<Row> | null> {
    if (this.followUp) return this.followUp;
    const current = this.inflight;
    if (!current) return this.start();
    const followUp = current
      .then(
        () => undefined,
        () => undefined,
      )
      .then(() => {
        if (this.followUp === followUp) this.followUp = null;
        return this.start();
      });
    this.followUp = followUp;
    return followUp;
  }

  /** A READ that needs the replica current but carries no new evidence of
   * change: joins the pull already in flight (or the queued follow-up)
   * instead of queueing another, else starts one. Tickles use `pull`. */
  join(): Promise<ReplicaDelta<Row> | null> {
    return this.followUp ?? this.inflight ?? this.start();
  }

  /** Fire-and-forget `pull` for SSE gallery frames and resyncs. */
  tickle(): void {
    void this.pull().catch(() => undefined);
  }

  /** Forget everything (host removed or re-keyed). Listeners stay. */
  dispose(): void {
    this.disposed = true;
    this.byName.clear();
    this.trashNames.clear();
    this.cursorValue = null;
    this.instance = null;
    this.lastLibrary = null;
    this.lastTrash = null;
    this.revisionValue += 1;
  }

  private start(): Promise<ReplicaDelta<Row> | null> {
    const operation = this.run()
      .then((delta) => {
        this.settledAtValue = Date.now();
        return delta;
      })
      .finally(() => {
        if (this.inflight === operation) this.inflight = null;
      });
    this.inflight = operation;
    return operation;
  }

  private async run(): Promise<ReplicaDelta<Row> | null> {
    await this.hydrate();
    if (this.disposed) return null;
    let useFeed = false;
    try {
      useFeed = await this.transport.feedSupported();
    } catch {
      useFeed = false;
    }
    if (useFeed) {
      try {
        return await this.pullFeed();
      } catch (error) {
        if (!(error instanceof GalleryFeedUnavailableError)) throw error;
      }
    }
    return this.pullLegacy();
  }

  private async pullFeed(): Promise<ReplicaDelta<Row> | null> {
    const ledger = new TouchLedger<Row>();
    const startCursor = this.cursorValue;
    let since = this.cursorValue;
    let rebuilding = since === null;
    let stale: Set<string> | null = rebuilding
      ? new Set(this.byName.keys())
      : null;
    let reset = rebuilding;
    let applied = false;
    try {
      for (let page = 0; page < MAX_PAGES_PER_PULL; page += 1) {
        const answer = await this.transport.changes(since, this.pageLimit);
        if (this.disposed) return null;
        if (answer.protocol_version !== GALLERY_CHANGES_PROTOCOL_VERSION)
          throw new GalleryFeedUnavailableError("unsupported");
        if (
          !answer.reset &&
          since !== null &&
          this.instance !== null &&
          answer.instance_id !== this.instance
        ) {
          // A different server answered our cursor: restart from a snapshot.
          since = null;
          this.cursorValue = null;
          this.instance = answer.instance_id;
          rebuilding = true;
          reset = true;
          stale = new Set(this.byName.keys());
          continue;
        }
        if (answer.reset) {
          reset = true;
          rebuilding = true;
          stale = new Set(this.byName.keys());
        }
        this.instance = answer.instance_id;
        if (answer.more && answer.cursor === since)
          // A page that claims more without advancing would loop forever;
          // treat it as a failure (the catch drops the cursor, and the
          // stale set dies unapplied, so nothing is removed).
          throw new FeedStalledError();
        for (const change of answer.changes) {
          stale?.delete(change.filename);
          if (this.trashNames.delete(change.filename))
            ledger.moved.add(change.filename);
          const prior = this.byName.get(change.filename);
          if (change.removed || !change.image) {
            if (prior === undefined) continue;
            ledger.note(change.filename, prior);
            this.byName.delete(change.filename);
          } else {
            ledger.note(change.filename, prior);
            this.byName.set(change.filename, change.image);
          }
        }
        this.cursorValue = answer.cursor;
        applied = true;
        since = answer.cursor;
        if (!answer.more) {
          if (stale) {
            for (const filename of stale) {
              ledger.note(filename, this.byName.get(filename));
              this.byName.delete(filename);
            }
            stale = null;
          }
          rebuilding = false;
          break;
        }
      }
      if (rebuilding) this.cursorValue = null;
    } catch (error) {
      // Mid-snapshot failure: rows applied so far are fresh, but the stale
      // set dies with this pull, so the next pull must snapshot again. A
      // feed that stopped advancing is not trusted to resume either.
      if (rebuilding || error instanceof FeedStalledError)
        this.cursorValue = null;
      if (applied || ledger.prior.size > 0) {
        this.modeValue = "feed";
        await this.finish(ledger, reset, "feed", startCursor);
      }
      throw error;
    }
    this.modeValue = "feed";
    this.trashFreshValue = true;
    this.lastLibrary = null;
    this.lastTrash = null;
    return this.finish(ledger, reset, "feed", startCursor);
  }

  private async pullLegacy(): Promise<ReplicaDelta<Row> | null> {
    const listTrash = this.transport.legacyTrash;
    const [library, trash] = await Promise.all([
      this.transport.legacyLibrary(),
      // A failed trash listing degrades to "not listed" (rows preserved,
      // `trashFresh` false) rather than failing the library with it.
      listTrash ? listTrash().catch(() => null) : null,
    ]);
    if (this.disposed) return null;
    const startCursor = this.cursorValue;
    // A listing breaks feed continuity; the next feed pull snapshots.
    this.cursorValue = null;
    this.modeValue = "legacy";
    this.trashFreshValue = trash !== null;
    if (
      library === this.lastLibrary &&
      (trash === null || trash === this.lastTrash) &&
      startCursor === null
    )
      return null;
    this.lastLibrary = library;
    this.lastTrash = trash;
    const ledger = new TouchLedger<Row>();
    const next = new Map<string, Row>();
    const nextTrash = new Set<string>();
    if (trash !== null)
      for (const row of trash) {
        next.set(row.filename, row);
        nextTrash.add(row.filename);
      }
    else
      // Not listed this pull: keep what we hold in the trash, by stamp or
      // by the provenance of the last listing that did answer.
      for (const row of this.byName.values())
        if (this.inTrash(row)) {
          next.set(row.filename, row);
          if (this.trashNames.has(row.filename)) nextTrash.add(row.filename);
        }
    for (const row of library) {
      next.set(row.filename, row);
      nextTrash.delete(row.filename);
    }
    for (const name of new Set([...this.trashNames, ...nextTrash]))
      if (this.trashNames.has(name) !== nextTrash.has(name))
        ledger.moved.add(name);
    this.trashNames = nextTrash;
    for (const [filename, prior] of this.byName) {
      if (!next.has(filename)) {
        ledger.note(filename, prior);
        this.byName.delete(filename);
      }
    }
    for (const [filename, row] of next) {
      const prior = this.byName.get(filename);
      if (prior === row) continue;
      ledger.note(filename, prior);
      this.byName.set(filename, row);
    }
    return this.finish(ledger, false, "legacy", startCursor);
  }

  private async finish(
    ledger: TouchLedger<Row>,
    reset: boolean,
    source: "feed" | "legacy",
    startCursor: string | null,
  ): Promise<ReplicaDelta<Row> | null> {
    const upserted: Row[] = [];
    const removed: string[] = [];
    for (const filename of ledger.moved)
      if (!ledger.prior.has(filename))
        ledger.prior.set(filename, this.byName.get(filename));
    for (const [filename, prior] of ledger.prior) {
      const now = this.byName.get(filename);
      if (now === undefined) {
        if (prior !== undefined) removed.push(filename);
      } else if (prior === undefined || !galleryRowsEqual(prior, now)) {
        upserted.push(now);
      } else if (ledger.moved.has(filename)) {
        if (prior !== now) this.byName.set(filename, prior);
        upserted.push(prior);
      } else if (prior !== now) {
        // Equal content, new object: keep the old one so identity-keyed
        // consumers (reactive rows, thumbnail URLs) see no change at all.
        this.byName.set(filename, prior);
      }
    }
    const changed = upserted.length > 0 || removed.length > 0;
    const delta: ReplicaDelta<Row> | null = changed
      ? {
          upserted,
          removed,
          reset,
          revision: (this.revisionValue += 1),
          source,
        }
      : null;
    if (delta) for (const listener of this.listeners) listener(delta);
    if (
      this.options.persistence &&
      (delta || startCursor !== this.cursorValue)
    ) {
      try {
        await this.options.persistence.save(this.snapshot(), delta);
      } catch {
        // Persistence is a paint-ahead cache; a failed write never fails
        // the pull that already brought the replica current.
      }
    }
    return delta;
  }

  /** The persistable state. */
  snapshot(): PersistedGalleryReplica<Row> {
    return {
      version: 1,
      instanceId: this.instance,
      cursor: this.cursorValue,
      rows: [...this.byName.values()],
    };
  }

  private async loadPersisted(): Promise<ReplicaDelta<Row> | null> {
    const persistence = this.options.persistence;
    if (!persistence) return null;
    let saved: PersistedGalleryReplica<Row> | null = null;
    try {
      saved = await persistence.load();
    } catch {
      saved = null;
    }
    if (
      !saved ||
      saved.version !== 1 ||
      !Array.isArray(saved.rows) ||
      this.disposed
    )
      return null;
    // A pull that already ran would be newer than anything saved.
    if (this.byName.size > 0 || this.cursorValue !== null) return null;
    const upserted: Row[] = [];
    for (const row of saved.rows) {
      if (!row || typeof row.filename !== "string") continue;
      this.byName.set(row.filename, row);
      upserted.push(row);
    }
    this.cursorValue = typeof saved.cursor === "string" ? saved.cursor : null;
    this.instance =
      typeof saved.instanceId === "string" ? saved.instanceId : null;
    if (upserted.length === 0) return null;
    const delta: ReplicaDelta<Row> = {
      upserted,
      removed: [],
      reset: true,
      revision: (this.revisionValue += 1),
      source: "hydrate",
    };
    for (const listener of this.listeners) listener(delta);
    return delta;
  }
}
