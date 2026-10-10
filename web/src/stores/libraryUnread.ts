import { reactive, ref } from "vue";
import { defineStore } from "pinia";
import {
  loadLibraryUnreadLedger,
  saveLibraryUnreadLedger,
} from "@studio/lib/libraryUnreadLedger";
import { groupLogicalGalleryPrints } from "@studio/lib/galleryPrintIdentity";
import {
  fetchMergedGallery,
  printKey,
  type HostGalleryImage,
} from "../lib/multiHostGallery";
import { listHosts } from "../lib/hostRegistry";

const KEY = "mold.web.libraryUnread.v1";
export const useLibraryUnread = defineStore("webLibraryUnread", () => {
  const ledger = reactive(loadLibraryUnreadLedger(KEY));
  const libraryReaders = ref(0);
  const persistenceError = ref<string | null>(ledger.loadError);
  return {
    ledger,
    libraryReaders,
    persistenceError,
    observe(rows: HostGalleryImage[], loadedHosts: string[]) {
      ledger.retainHosts(listHosts().map((host) => host.id));
      ledger.observe(
        groupLogicalGalleryPrints(rows).map((group) =>
          group.copies.map(printKey),
        ),
        loadedHosts,
      );
      persistenceError.value = saveLibraryUnreadLedger(KEY, ledger);
    },
    viewPhysical(hostId: string, filename: string) {
      ledger.view([`${hostId}|${filename}`]);
      persistenceError.value = saveLibraryUnreadLedger(KEY, ledger);
    },
    view(copies: HostGalleryImage[]) {
      ledger.view(copies.map(printKey));
      persistenceError.value = saveLibraryUnreadLedger(KEY, ledger);
    },
  };
});

/** Runs outside Library so a first visit does not swallow arrivals since startup. */
export function startLibraryUnreadObserver(): () => void {
  const store = useLibraryUnread();
  let stopped = false;
  let busy = false;
  const refresh = async () => {
    if (stopped || busy || document.hidden || store.libraryReaders > 0) return;
    busy = true;
    try {
      const hosts = listHosts();
      const inventory = await fetchMergedGallery(hosts);
      if (
        !stopped &&
        hosts.map((host) => `${host.id}:${host.url}`).join() ===
          listHosts()
            .map((host) => `${host.id}:${host.url}`)
            .join()
      )
        store.observe(inventory.rawEntries, inventory.reachableHostIds);
    } catch {
      /* Offline hosts retain their local viewing history. */
    } finally {
      busy = false;
    }
  };
  void refresh();
  const timer = setInterval(() => {
    void refresh();
  }, 10_000);
  document.addEventListener("visibilitychange", refresh);
  return () => {
    stopped = true;
    clearInterval(timer);
    document.removeEventListener("visibilitychange", refresh);
  };
}
