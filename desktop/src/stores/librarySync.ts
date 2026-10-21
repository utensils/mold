import { computed, onScopeDispose, ref, watch } from "vue";
import {
  provisionalCopyPrivacy,
  retainCopyPrivacyUntilConfirmed,
  type ProvisionalCopyPrivacy,
} from "@studio/lib/copyPrivacy";
import { defineStore } from "pinia";
import { copyGalleryPrint, copyMachineIdentity, type CopyMachine } from "@studio/api/galleryCopy";
import {
  copyJournal,
  withSyncDocument,
  type SyncDocumentStorage,
  type LibrarySyncDocument,
} from "@studio/lib/galleryCopyJournal";
import { LIBRARY_SYNC_FORGET, LIBRARY_SYNC_DOCUMENT } from "../lib/librarySyncEvents";
import { ipc, inTauri, nativeLibraryCopyHooks } from "../lib/ipc";
import { useGalleryStore, type MergedPrint } from "./gallery";
import { useHostsStore } from "./hosts";
let lastPrivacyDocument: LibrarySyncDocument | null = null;
function publishCopyPrivacy(document: LibrarySyncDocument | null) {
  const gallery = useGalleryStore();
  const confirmed = new Set<string>();
  for (const edge of gallery.provisionalCopyPrivacy ?? []) {
    let id: string;
    try {
      id = JSON.parse(edge.destination)[0];
    } catch {
      continue;
    }
    const bucket = gallery.buckets[id];
    const collections = gallery.collectionsByHost[id];
    if (!bucket?.loaded || bucket.error || !collections?.loaded || collections.error) continue;
    const row = gallery.bucketIndex
      ? gallery.bucketIndex.get(id)?.byFilename.get(edge.destinationFilename)?.item
      : bucket.items.find((item) => item.filename === edge.destinationFilename);
    if (
      row?.collections?.some((collectionId) =>
        collections.items.some((collection) => collection.id === collectionId && collection.hidden),
      )
    )
      confirmed.add(`${id}|${edge.destinationFilename}`);
  }
  gallery.provisionalCopyPrivacy = retainCopyPrivacyUntilConfirmed(
    provisionalCopyPrivacy(document?.receipts ?? {}),
    gallery.provisionalCopyPrivacy ?? [],
    document?.receipts ?? {},
    confirmed,
  );
}
const storage: SyncDocumentStorage = {
  async read() {
    const document = await ipc.librarySyncRead();
    lastPrivacyDocument = document;
    publishCopyPrivacy(document);
    return document;
  },
  async write(document) {
    // Quarantine the acknowledged filename immediately, including when a
    // later journal write fails. This is never an output-skip receipt.
    const gallery = useGalleryStore();
    const existing = gallery.provisionalCopyPrivacy ?? [];
    const pending = provisionalCopyPrivacy(document.receipts);
    const key = (edge: ProvisionalCopyPrivacy) =>
      JSON.stringify([
        edge.source,
        edge.destination,
        edge.sourceFilename,
        edge.destinationFilename,
      ]);
    gallery.provisionalCopyPrivacy = [
      ...new Map([...existing, ...pending].map((edge) => [key(edge), edge])).values(),
    ];
    await ipc.librarySyncWrite(document);
    lastPrivacyDocument = document;
    publishCopyPrivacy(document);
  },
};
interface Run {
  destination: CopyMachine;
  entries: { source: CopyMachine; filename: string; fingerprint: string }[];
  automatic: boolean;
}
export const useLibrarySyncStore = defineStore("librarySync", () => {
  const hosts = useHostsStore(),
    gallery = useGalleryStore();
  const keep = ref<Record<string, boolean>>({});
  const busy = ref(false),
    status = ref(""),
    error = ref(""),
    dismissed = ref(false);
  const machines = computed<CopyMachine[]>(() =>
    hosts.all.flatMap((host) => {
      const target = gallery.targetOf(host.id);
      return host.status === "ready" && host.instanceId && target
        ? [{ id: host.id, label: host.label, instanceId: host.instanceId, target: { ...target } }]
        : [];
    }),
  );
  const destinations = computed<CopyMachine[]>(() =>
    hosts.all.flatMap((host) =>
      host.baseUrl && host.instanceId
        ? [
            {
              id: host.id,
              label: host.label,
              instanceId: host.instanceId,
              target: {
                baseUrl: host.baseUrl,
                apiKey: host.apiKey,
                connectionMode: host.connectionMode,
              },
            },
          ]
        : [],
    ),
  );
  let active: AbortController | undefined;
  let activeMachines: CopyMachine[] = [];
  const queue: Run[] = [];
  let activeRun: Run | undefined;
  let activeEntryIndex = -1;
  let sourceForgotten = false;
  const observed = new Map<string, string>();
  const entryKey = (destination: CopyMachine, source: CopyMachine, filename: string) =>
    JSON.stringify([copyMachineIdentity(destination), copyMachineIdentity(source), filename]);
  let initialized = false;
  const currentMachine = (machine: CopyMachine) =>
    machines.value.some(
      (row) =>
        copyMachineIdentity(row) === copyMachineIdentity(machine) &&
        row.target.apiKey === machine.target.apiKey,
    );
  function stop() {
    sourceForgotten = false;
    // Suppress only this stopped snapshot. A later source edit/new print has
    // a different fingerprint; deliberate Copy/Sync remains available now.
    for (const run of [...queue, ...(activeRun ? [activeRun] : [])])
      for (const entry of run.entries)
        observed.set(entryKey(run.destination, entry.source, entry.filename), entry.fingerprint);
    queue.splice(0);
    if (timer) clearTimeout(timer);
    if (active) {
      active.abort();
      status.value = "Stopping Library transfer…";
    }
  }
  async function initialize() {
    if (initialized || !inTauri()) return;
    initialized = true;
    try {
      keep.value = (await storage.read())?.keep ?? {};
    } catch (failure) {
      error.value = String(failure);
    }
  }
  async function setKeep(machine: CopyMachine, enabled: boolean) {
    await initialize();
    const key = copyMachineIdentity(machine);
    await withSyncDocument(storage, (document) => {
      document.keep[key] = enabled;
    });
    keep.value = { ...keep.value, [key]: enabled };
    if (enabled && currentMachine(machine)) syncAll(machine, true);
    else {
      if (!enabled && activeRun?.automatic && copyMachineIdentity(activeRun.destination) === key) {
        sourceForgotten = false;
        active?.abort();
      }
      for (let index = queue.length - 1; index >= 0; index--)
        if (queue[index]!.automatic && copyMachineIdentity(queue[index]!.destination) === key)
          queue.splice(index, 1);
    }
  }
  function selectedCopies(
    entries: MergedPrint[],
    destination: CopyMachine,
    allPhysical = false,
  ): Run["entries"] {
    const seen = new Set<string>();
    return entries.flatMap((entry) => {
      const copies = entry.copies ?? [{ sourceKey: entry.sourceKey, item: entry.item }];
      const eligible = copies
        .map((copy) => ({
          copy,
          source: machines.value.find((machine) => machine.id === copy.sourceKey),
        }))
        .filter(
          (owner): owner is { copy: (typeof copies)[number]; source: CopyMachine } =>
            !!owner.source &&
            copyMachineIdentity(owner.source) !== copyMachineIdentity(destination),
        );
      // Display merging is privacy/UI evidence, never proof that two physical
      // files have equal bytes. Sync All visits every physical live owner.
      return (allPhysical ? eligible : eligible.slice(0, 1)).flatMap(({ copy, source }) => {
        const key = JSON.stringify([copyMachineIdentity(source), copy.item.filename]);
        if (seen.has(key)) return [];
        seen.add(key);
        return [
          {
            source,
            filename: copy.item.filename,
            fingerprint: JSON.stringify([copy.item, gallery.collectionsByHost[source.id]?.items]),
          },
        ];
      });
    });
  }
  function enqueue(run: Run) {
    if (!run.entries.length) return;
    if (run.automatic) {
      const prior = queue.findIndex(
        (row) =>
          row.automatic &&
          copyMachineIdentity(row.destination) === copyMachineIdentity(run.destination),
      );
      if (prior >= 0) queue.splice(prior, 1, run);
      else queue.push(run);
    } else queue.push(run);
    void drain();
  }
  function copyTo(entries: MergedPrint[], destination: CopyMachine) {
    enqueue({ entries: selectedCopies(entries, destination), destination, automatic: false });
  }
  function syncAll(destination: CopyMachine, automatic = false) {
    const entries = selectedCopies(gallery.merged, destination, true).filter(
      (entry) =>
        !automatic ||
        observed.get(entryKey(destination, entry.source, entry.filename)) !== entry.fingerprint,
    );
    enqueue({ entries, destination, automatic });
  }
  async function drain() {
    if (busy.value) return;
    busy.value = true;
    try {
      while (queue.length) {
        const run = queue.shift()!;
        activeRun = run;
        active = new AbortController();
        error.value = "";
        dismissed.value = false;
        const captured = run.destination;
        let completed = 0,
          legacyUnavailable = 0;
        for (let index = 0; index < run.entries.length; index++) {
          activeEntryIndex = index;
          const entry = run.entries[index]!;
          if (active.signal.aborted || !currentMachine(captured)) break;
          if (!currentMachine(entry.source)) continue;
          const message = `Copying ${++completed}/${run.entries.length} to ${captured.label}`;
          status.value = message;
          try {
            activeMachines = [entry.source, captured];
            const receipt = await copyGalleryPrint({
              ...nativeLibraryCopyHooks(),
              source: entry.source,
              destination: captured,
              filename: entry.filename,
              journal: copyJournal(storage),
              signal: active.signal,
              deliberate: !run.automatic,
              current: () => currentMachine(entry.source) && currentMachine(captured),
              progress: (stage) => {
                status.value = `${message} · ${stage}`;
              },
            });
            if (receipt.outcome === "legacy_inputs_unavailable") legacyUnavailable++;
            if (receipt.organization?.conflicts?.length)
              error.value = receipt.organization.conflicts.join(" · ");
            if (receipt.stage === "complete")
              observed.set(entryKey(captured, entry.source, entry.filename), entry.fingerprint);
          } catch (failure) {
            if (!active.signal.aborted) {
              error.value = `${entry.filename}: ${failure instanceof Error ? failure.message : String(failure)}`;
              dismissed.value = false;
            }
          }
          if (active.signal.aborted && sourceForgotten && currentMachine(captured)) {
            sourceForgotten = false;
            active = new AbortController();
            activeMachines = [];
          }
        }
        if (!active.signal.aborted)
          status.value = `Library transfer to ${captured.label} finished${error.value ? " with issues" : ""}.${legacyUnavailable ? ` Original inputs were not retained for ${legacyUnavailable} older print(s).` : ""}`;
        else status.value = "Library transfer stopped. Copy again to repair incomplete prints.";
        await gallery.fetchAll();
      }
    } finally {
      busy.value = false;
      active = undefined;
      activeMachines = [];
      activeRun = undefined;
      activeEntryIndex = -1;
      sourceForgotten = false;
    }
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
  watch(
    () =>
      (gallery.provisionalCopyPrivacy?.length ?? 0)
        ? JSON.stringify([
            Object.entries(gallery.buckets).map(([id, bucket]) => [id, bucket.items]),
            gallery.collectionsByHost,
          ])
        : "",
    () => publishCopyPrivacy(lastPrivacyDocument),
  );
  watch(
    () =>
      Object.values(keep.value).some(Boolean)
        ? JSON.stringify([
            Object.entries(gallery.buckets).map(([key, bucket]) => [key, bucket.items]),
            gallery.collectionsByHost,
          ])
        : "",
    () => {
      if (!initialized || !Object.values(keep.value).some(Boolean)) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        for (const machine of machines.value)
          if (keep.value[copyMachineIdentity(machine)]) syncAll(machine, true);
      }, 1000);
    },
  );
  // Host telemetry refreshes rebuild HostView objects. Only a real route,
  // credential, installation or connectivity transition starts another run.
  watch(
    () =>
      JSON.stringify(
        machines.value.map((machine) => [copyMachineIdentity(machine), machine.target.apiKey]),
      ),
    () => {
      if (active && activeMachines.some((machine) => !currentMachine(machine))) active.abort();
      if (initialized)
        for (const machine of machines.value)
          if (keep.value[copyMachineIdentity(machine)]) syncAll(machine, true);
    },
  );
  async function forget(id: string) {
    if (active && activeMachines.some((machine) => machine.id === id)) {
      sourceForgotten = activeRun?.destination.id !== id;
      active.abort();
    }
    // A forgotten source removes its own future rows, preserving unrelated
    // selections and destinations. Stop remains the explicit global action.
    if (activeRun && activeRun.destination.id !== id)
      for (let index = activeRun.entries.length - 1; index > activeEntryIndex; index--)
        if (activeRun.entries[index]!.source.id === id) activeRun.entries.splice(index, 1);
    for (let index = queue.length - 1; index >= 0; index--) {
      const run = queue[index]!;
      if (run.destination.id === id) queue.splice(index, 1);
      else {
        run.entries = run.entries.filter((entry) => entry.source.id !== id);
        if (!run.entries.length) queue.splice(index, 1);
      }
    }
    const keys = Object.keys(keep.value).filter((key) => {
      try {
        return JSON.parse(key)[0] === id;
      } catch {
        return false;
      }
    });
    if (!keys.length) return;
    await withSyncDocument(storage, (document) => {
      for (const key of keys) delete document.keep[key];
    });
    const next = { ...keep.value };
    for (const key of keys) delete next[key];
    keep.value = next;
  }
  const forgotten = (event: Event) => {
    void forget((event as CustomEvent<{ id: string }>).detail.id).catch((failure) => {
      error.value = String(failure);
    });
  };
  const journalUpdated = (event: Event) => {
    lastPrivacyDocument = (event as CustomEvent<LibrarySyncDocument>).detail;
    publishCopyPrivacy(lastPrivacyDocument);
  };
  if (typeof window !== "undefined") {
    window.addEventListener(LIBRARY_SYNC_FORGET, forgotten);
    window.addEventListener(LIBRARY_SYNC_DOCUMENT, journalUpdated);
  }
  onScopeDispose(() => {
    stop();
    if (typeof window !== "undefined") {
      window.removeEventListener(LIBRARY_SYNC_FORGET, forgotten);
      window.removeEventListener(LIBRARY_SYNC_DOCUMENT, journalUpdated);
    }
  });
  return {
    machines,
    destinations,
    keep,
    busy,
    status,
    error,
    dismissed,
    initialize,
    setKeep,
    copyTo,
    syncAll,
    stop,
    forget,
  };
});
