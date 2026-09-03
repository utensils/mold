import { computed, type ComputedRef } from "vue";
import { useRoute } from "vue-router";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useDownloadsStore } from "../stores/downloads";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useQueueActivity } from "./useQueueActivity";

/** "3 waiting" / "nothing waiting" — the plural-aware count phrase. */
export function countPhrase(count: number, noun: string, plural = `${noun}s`): string {
  return `${count} ${count === 1 ? noun : plural}`;
}

/**
 * The sans subtitle beside the view's mono title: one sentence of live
 * context per view (README §04 — "which machine, how full, how deep is the
 * queue" never costs a view change).
 */
export function useShellSubtitle(): ComputedRef<string> {
  const route = useRoute();
  const draft = useSequenceDraftStore();
  const downloads = useDownloadsStore();
  const gallery = useGalleryStore();
  const hosts = useHostsStore();
  const models = useModelStore();
  const activity = useQueueActivity();

  return computed(() => {
    const waiting = activity.waitingCount.value;
    const making = activity.activeCount.value;
    if (route.path.startsWith("/machines")) {
      if (route.name === "runpod") return "Rent a GPU · billed by the minute";
      return `${countPhrase(hosts.all.filter((h) => h.status === "ready").length, "machine")} connected`;
    }
    switch (route.path) {
      case "/create": {
        const output = draft.output === "sequence" ? "Short clip" : "Still picture";
        return `${output} · ${countPhrase(waiting, "waiting", "waiting")}`;
      }
      case "/queue":
        return `${countPhrase(waiting, "waiting", "waiting")} · ${making} being made`;
      case "/library":
        return `${countPhrase(gallery.basePrintCount, "picture")} · ${countPhrase(gallery.mergedCollections.length, "album")}`;
      case "/models": {
        const pulling = downloads.hostedInFlight.length;
        return `${countPhrase(models.installed.length, "style")} ready${pulling ? ` · ${pulling} downloading` : ""}`;
      }
      case "/settings":
        return hosts.primaryHost?.label ?? "";
      default:
        return "";
    }
  });
}
