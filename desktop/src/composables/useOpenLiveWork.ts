import { useRouter } from "vue-router";
import { openLiveWorkWith } from "@studio/composables/useOpenLiveWork";
import type { OutputMetadata } from "../lib/api/types";
import { useComposerStore } from "../stores/composer";
import { useHostsStore } from "../stores/hosts";
import { useToastStore } from "../stores/toasts";

/**
 * The app's half of opening server-owned work: its machines, its toasts, its
 * composer, and where a chain row and a download row go. The decision itself
 * — the chain guard, the queue lookup, the 3-D workflow route — is the shared
 * `openLiveWorkWith`.
 */
export function useOpenLiveWork() {
  const router = useRouter();
  const hosts = useHostsStore();
  const composer = useComposerStore();
  const toasts = useToastStore();

  return openLiveWorkWith<OutputMetadata>({
    targetFor: (hostId) => {
      const host = hosts.all.find((candidate) => candidate.id === hostId);
      return host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : null;
    },
    go: (to) => router.push(to).then(() => undefined),
    fail: (message) => toasts.push(message, "error"),
    restore: (selection, hostId) =>
      composer.set({
        metadata: selection.metadata,
        queueSelection: {
          hostId,
          jobId: selection.jobId,
          running: selection.running,
        },
      }),
    // A chain job — a long clip the host had to split and stitch, or one the
    // CLI authored — has no client surface to re-enter: scene-by-scene
    // authoring is retired. Its print lands in My images like any other.
    chainDestination: () => "/queue",
    openDownloads: () => "/models",
  });
}
