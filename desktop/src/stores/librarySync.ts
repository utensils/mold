import { computed, onScopeDispose, ref, watch } from "vue";
import { defineStore } from "pinia";
import { copyGalleryPrint, copyMachineIdentity, type CopyMachine } from "@studio/api/galleryCopy";
import {
  copyJournal,
  withSyncDocument,
  type SyncDocumentStorage,
} from "@studio/lib/galleryCopyJournal";
import { LIBRARY_SYNC_FORGET } from "../lib/librarySyncEvents";
import { ipc, inTauri, nativeLibraryCopyHooks } from "../lib/ipc";
import { useGalleryStore, type MergedPrint } from "./gallery";
import { useHostsStore } from "./hosts";
const storage: SyncDocumentStorage = {
  read: () => ipc.librarySyncRead(),
  write: (document) => ipc.librarySyncWrite(document),
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
    else
      for (let index = queue.length - 1; index >= 0; index--)
        if (queue[index]!.automatic && copyMachineIdentity(queue[index]!.destination) === key)
          queue.splice(index, 1);
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
        for (const entry of run.entries) {
          if (active.signal.aborted || !currentMachine(captured)) break;
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
    }
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
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
  watch(machines, () => {
    if (active && activeMachines.some((machine) => !currentMachine(machine))) active.abort();
    if (initialized)
      for (const machine of machines.value)
        if (keep.value[copyMachineIdentity(machine)]) syncAll(machine, true);
  });
  async function forget(id: string) {
    if (activeMachines.some((machine) => machine.id === id)) stop();
    for (let index = queue.length - 1; index >= 0; index--)
      if (
        queue[index]!.destination.id === id ||
        queue[index]!.entries.some((entry) => entry.source.id === id)
      )
        queue.splice(index, 1);
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
  if (typeof window !== "undefined") window.addEventListener(LIBRARY_SYNC_FORGET, forgotten);
  onScopeDispose(() => {
    stop();
    if (typeof window !== "undefined") window.removeEventListener(LIBRARY_SYNC_FORGET, forgotten);
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
