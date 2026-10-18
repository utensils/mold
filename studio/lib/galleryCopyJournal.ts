import type { CopyJournal, CopyReceipt } from "../api/galleryCopy";
import { browserCopyLock as browserLock } from "./browserCopyLock";
export interface LibrarySyncDocument {
  version: 1;
  receipts: Record<string, CopyReceipt>;
  keep: Record<string, boolean>;
}
export interface SyncDocumentStorage {
  read(): Promise<LibrarySyncDocument | null>;
  write(document: LibrarySyncDocument): Promise<void>;
  exclusive?<T>(operation: () => Promise<T>): Promise<T>;
}
export const browserSyncStorage: SyncDocumentStorage = {
  exclusive: (operation) => browserLock("mold.library-sync.v1", operation),
  async read() {
    const text = localStorage.getItem("mold.library-sync.v1");
    if (!text) return null;
    const document = JSON.parse(text);
    if (document.version !== 1 || !document.receipts || !document.keep)
      throw new Error(
        "The Library copy journal is unsupported. Existing copies were preserved.",
      );
    return document;
  },
  async write(document) {
    localStorage.setItem("mold.library-sync.v1", JSON.stringify(document));
  },
};
let tail: Promise<unknown> = Promise.resolve();
export function withSyncDocument<T>(
  storage: SyncDocumentStorage,
  update: (document: LibrarySyncDocument) => Promise<T> | T,
): Promise<T> {
  const run = tail
    .catch(() => undefined)
    .then(() => {
      const mutate = async () => {
        const document = (await storage.read()) ?? {
          version: 1,
          receipts: {},
          keep: {},
        };
        const result = await update(document);
        await storage.write(document);
        return result;
      };
      return storage.exclusive ? storage.exclusive(mutate) : mutate();
    });
  tail = run;
  return run;
}
export function copyJournal(storage: SyncDocumentStorage): CopyJournal {
  return {
    ...(storage === browserSyncStorage
      ? {
          runExclusive: <T>(
            operation: (lockSignal?: AbortSignal) => Promise<T>,
            signal?: AbortSignal,
          ) => browserLock("mold.library-copy.v1", operation, signal),
        }
      : {}),
    read: async (key) => (await storage.read())?.receipts[key] ?? null,
    matching: async (destination, digest, recipe) =>
      Object.values((await storage.read())?.receipts ?? {}).find(
        (receipt) =>
          receipt.destination === destination &&
          receipt.digest === digest &&
          receipt.recipe === recipe &&
          !!receipt.destinationFilename,
      ) ?? null,
    original: async (origin) =>
      Object.values((await storage.read())?.receipts ?? {})
        .filter(
          (receipt) =>
            receipt.origin === origin &&
            JSON.stringify([receipt.source, receipt.sourceFilename]) ===
              origin &&
            receipt.stage === "complete" &&
            !!receipt.sourceOrganization,
        )
        .sort((a, b) => (b.observedAt ?? 0) - (a.observedAt ?? 0))[0] ?? null,
    lineage: async (source, filename, digest, recipe) =>
      Object.values((await storage.read())?.receipts ?? {}).find(
        (receipt) =>
          receipt.destination === source &&
          receipt.destinationFilename === filename &&
          receipt.digest === digest &&
          (receipt.destinationRecipe ?? receipt.recipe) === recipe &&
          !!receipt.origin,
      ) ?? null,
    contributors: async (destination, filename, excluding) =>
      Object.entries((await storage.read())?.receipts ?? {})
        .filter(
          ([key, receipt]) =>
            key !== excluding &&
            receipt.destination === destination &&
            receipt.destinationFilename === filename &&
            !!receipt.organization,
        )
        .map(([, receipt]) => receipt.organization!),
    write: (key, receipt) =>
      withSyncDocument(storage, (document) => {
        document.receipts[key] = receipt;
      }),
  };
}
