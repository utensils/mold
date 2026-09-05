import { computed, type ComputedRef } from "vue";
import { useRoute } from "vue-router";
import { useDownloadsStore } from "../stores/downloads";
import { useGalleryStore } from "../stores/gallery";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { mergeInstalledAcrossFleet } from "../lib/models";
import { OUTPUT_KIND_TITLE, useCreateOutputKind } from "./useCreateOutputKind";
import { useQueueActivity } from "./useQueueActivity";

/** "3 waiting" / "nothing waiting" — the plural-aware count phrase. */
export function countPhrase(count: number, noun: string, plural = `${noun}s`): string {
  return `${count} ${count === 1 ? noun : plural}`;
}

/**
 * The view's mono title: the route's title, except on New image, where it
 * follows the output kind — "New image", "New clip", "New 3-D object" — so
 * the bar names what is being made, not the door that was clicked. The
 * sidebar's destination stays "New image"; the route is the same.
 */
export function useShellTitle(): ComputedRef<string> {
  const route = useRoute();
  const outputKind = useCreateOutputKind();
  return computed(() => {
    if (route.path === "/create") return OUTPUT_KIND_TITLE[outputKind.value];
    return (route.meta.title as string | undefined) ?? "";
  });
}

/**
 * The sans subtitle beside the view's mono title: one sentence of live
 * context per view (README §04 — "which machine, how full, how deep is the
 * queue" never costs a view change).
 */
export function useShellSubtitle(): ComputedRef<string> {
  const route = useRoute();
  const downloads = useDownloadsStore();
  const gallery = useGalleryStore();
  const hostModels = useHostModelsStore();
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
      // The output kind is the title's job now (`useShellTitle`); saying it
      // twice on one bar is noise.
      case "/create":
        return countPhrase(waiting, "waiting", "waiting");
      case "/queue":
        return `${countPhrase(waiting, "waiting", "waiting")} · ${making} being made`;
      case "/library":
        return `${countPhrase(gallery.basePrintCount, "picture")} · ${countPhrase(gallery.mergedCollections.length, "album")}`;
      case "/models": {
        const pulling = downloads.hostedInFlight.length;
        // The SAME set the Ready-to-use badge counts — the whole fleet. These
        // two read different sets and said the same word, which is how "106"
        // came to sit beside "25 styles ready".
        const ready = mergeInstalledAcrossFleet(
          models.installed,
          hostModels.unionDownloaded,
        ).length;
        const machines = hosts.all.filter((h) => h.status === "ready").length;
        const scope = machines > 1 ? " across your machines" : "";
        return `${countPhrase(ready, "style")} ready${scope}${pulling ? ` · ${pulling} downloading` : ""}`;
      }
      case "/settings":
        return hosts.primaryHost?.label ?? "";
      default:
        return "";
    }
  });
}
