/** Device/browser-local viewed history. A newly connected host's first successful
 * inventory is a compatibility baseline; later arrivals require individual viewing.
 * Groups contain physical copy keys supplied by the gallery's merge authority. */
export class LibraryUnreadLedger {
  known = new Map<string, Set<string>>();
  unread = new Set<string>();
  pendingViewed = new Set<string>();

  loadError: string | null = null;
  constructor(serialized?: string | null) {
    if (!serialized) return;
    try {
      const value = JSON.parse(serialized);
      const known = new Map<string, Set<string>>();
      const unread = new Set<string>();
      if (
        value.version === 2 &&
        Array.isArray(value.names) &&
        Array.isArray(value.hosts)
      ) {
        if (value.names.some((name: unknown) => typeof name !== "string"))
          throw new Error();
        for (const row of value.hosts) {
          if (!Array.isArray(row) || row.length !== 3) throw new Error();
          const [host, read, fresh] = row;
          if (
            typeof host !== "string" ||
            !Array.isArray(read) ||
            !Array.isArray(fresh)
          )
            throw new Error();
          const keys = new Set<string>();
          for (const index of [...read, ...fresh]) {
            if (
              !Number.isInteger(index) ||
              index < 0 ||
              index >= value.names.length
            )
              throw new Error();
            keys.add(`${host}|${value.names[index]}`);
          }
          known.set(host, keys);
          for (const index of fresh)
            unread.add(`${host}|${value.names[index]}`);
        }
      } else if (
        value.version === 1 &&
        Array.isArray(value.known) &&
        Array.isArray(value.unread)
      ) {
        for (const [host, keys] of value.known) {
          if (
            typeof host !== "string" ||
            !Array.isArray(keys) ||
            keys.some((key) => typeof key !== "string")
          )
            throw new Error();
          known.set(host, new Set(keys));
        }
        if (value.unread.some((key: unknown) => typeof key !== "string"))
          throw new Error();
        for (const key of value.unread) unread.add(key);
      } else {
        throw new Error();
      }
      if (
        value.pendingViewed !== undefined &&
        (!Array.isArray(value.pendingViewed) ||
          value.pendingViewed.some((key: unknown) => typeof key !== "string"))
      )
        throw new Error();
      this.pendingViewed = new Set(value.pendingViewed ?? []);
      this.known = known;
      this.unread = unread;
    } catch {
      this.loadError =
        "Saved viewing history could not be read. Its stored data has been preserved; New labels may be incomplete.";
    }
  }

  observe(
    groups: readonly (readonly string[])[],
    loadedHosts: readonly string[],
  ): void {
    const initialized = new Set(this.known.keys());
    const previous = new Set(
      [...this.known.values()].flatMap((keys) => [...keys]),
    );
    for (const copies of groups) {
      const familiar = copies.filter((key) => previous.has(key));
      if (
        copies.some((key) => this.pendingViewed.has(key)) ||
        familiar.some((key) => !this.unread.has(key))
      ) {
        this.view(copies);
      } else if (
        copies.some((key) => this.unread.has(key)) ||
        (familiar.length === 0 &&
          copies.some((key) => initialized.has(this.hostOf(key))))
      ) {
        for (const key of copies) this.unread.add(key);
      }
      for (const key of copies) {
        const host = this.hostOf(key);
        if (!this.known.has(host)) this.known.set(host, new Set());
        this.known.get(host)!.add(key);
        this.pendingViewed.delete(key);
      }
    }
    for (const host of loadedHosts)
      if (!this.known.has(host)) this.known.set(host, new Set());
  }

  isUnread(copies: readonly string[]): boolean {
    return copies.some((key) => this.unread.has(key));
  }
  count(groups: readonly (readonly string[])[]): number {
    return groups.filter((copies) => this.isUnread(copies)).length;
  }
  view(copies: readonly string[]): void {
    for (const key of copies) {
      this.unread.delete(key);
      if (!this.known.get(this.hostOf(key))?.has(key))
        this.pendingViewed.add(key);
    }
  }
  retainHosts(hosts: readonly string[]): void {
    const present = new Set(hosts);
    for (const host of this.known.keys())
      if (!present.has(host)) this.known.delete(host);
    for (const key of this.unread)
      if (!present.has(this.hostOf(key))) this.unread.delete(key);
    for (const key of this.pendingViewed)
      if (!present.has(this.hostOf(key))) this.pendingViewed.delete(key);
  }
  serialize(): string {
    // Store each filename once across hosts and encode read/unread membership as
    // integer references. Full host prefixes and mirrored names otherwise swamp
    // localStorage's quota well before the gallery's normal 20,000-print size.
    const names: string[] = [];
    const indices = new Map<string, number>();
    const hosts = [...this.known].map(([host, keys]) => {
      const read: number[] = [],
        fresh: number[] = [];
      for (const key of keys) {
        const name = key.slice(key.indexOf("|") + 1);
        let index = indices.get(name);
        if (index === undefined) {
          index = names.length;
          names.push(name);
          indices.set(name, index);
        }
        (this.unread.has(key) ? fresh : read).push(index);
      }
      return [host, read, fresh];
    });
    return JSON.stringify({
      version: 2,
      names,
      hosts,
      ...(this.pendingViewed.size
        ? { pendingViewed: [...this.pendingViewed] }
        : {}),
    });
  }
  hostOf(key: string): string {
    return key.slice(0, key.indexOf("|"));
  }
}

export function loadLibraryUnreadLedger(key: string): LibraryUnreadLedger {
  try {
    return new LibraryUnreadLedger(localStorage.getItem(key));
  } catch {
    const ledger = new LibraryUnreadLedger();
    ledger.loadError =
      "Saved viewing history is unavailable on this client. Its stored data has been preserved; New labels may be incomplete.";
    return ledger;
  }
}
export function saveLibraryUnreadLedger(
  key: string,
  ledger: LibraryUnreadLedger,
): string | null {
  if (ledger.loadError) return ledger.loadError;
  try {
    localStorage.setItem(key, ledger.serialize());
    return null;
  } catch {
    return "Viewing history could not be saved on this client. New labels may return after restart; free browser/app storage and try again.";
  }
}
