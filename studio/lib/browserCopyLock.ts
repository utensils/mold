/** IndexedDB transactions also serialize tabs on ordinary HTTP LAN origins.
 * Separate databases let a copy hold its lock while writing journal checkpoints,
 * including browsers that serialize all write transactions in one database.
 * Outstanding reads keep the transaction alive during network work; page exit
 * aborts it, without a stale lease or a timeout that could permit overlapping copies.
 */
export async function browserCopyLock<T>(
  name: string,
  operation: (lockSignal?: AbortSignal) => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  if (typeof navigator !== "undefined" && navigator.locks)
    return await navigator.locks.request(name, signal ? { signal } : {}, () =>
      operation(signal),
    );
  signal?.throwIfAborted();
  if (typeof indexedDB === "undefined")
    throw new Error(
      "This browser cannot safely save Library copy progress. Use an updated browser or the native app.",
    );
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(`${name}.lock`, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("lock");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(
        new Error(
          "This browser cannot safely save Library copy progress. Allow site storage or use the native app.",
          { cause: request.error },
        ),
      );
  });
  try {
    signal?.throwIfAborted();
    return await new Promise<T>((resolve, reject) => {
      const transaction = database.transaction("lock", "readwrite");
      let started = false;
      let settled = false;
      let ended = false;
      let value: T;
      let failure: unknown;
      let work: Promise<T> | undefined;
      const controller = new AbortController();
      const abortWaiting = () => {
        controller.abort(signal?.reason);
        if (!started) {
          failure = signal?.reason ?? new DOMException("Aborted", "AbortError");
          transaction.abort();
        }
      };
      signal?.addEventListener("abort", abortWaiting, { once: true });
      const cleanup = () => signal?.removeEventListener("abort", abortWaiting);
      transaction.oncomplete = () => {
        ended = true;
        cleanup();
        resolve(value);
      };
      transaction.onabort = () => {
        ended = true;
        cleanup();
        const error =
          failure ??
          transaction.error ??
          new Error(
            "The Library copy lock was interrupted. Copy again to repair progress.",
          );
        controller.abort(error);
        if (work) void work.catch(() => undefined).then(() => reject(error));
        else reject(error);
      };
      const store = transaction.objectStore(transaction.objectStoreNames[0]);
      const keepAlive = () => {
        const request = store.get("lock");
        request.onsuccess = () => {
          if (!started) {
            if (signal?.aborted) {
              abortWaiting();
              return;
            }
            started = true;
            work = Promise.resolve().then(() => operation(controller.signal));
            void work.then(
              (result) => {
                value = result;
                settled = true;
              },
              (error) => {
                failure = error;
                settled = true;
                if (!ended) transaction.abort();
              },
            );
          }
          if (!settled) keepAlive();
        };
      };
      keepAlive();
    });
  } finally {
    database.close();
  }
}
