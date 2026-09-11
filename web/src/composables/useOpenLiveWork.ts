import { useRouter } from "vue-router";
import { openLiveWorkWith } from "@studio/composables/useOpenLiveWork";
import type { OutputMetadata } from "../types";
import type { HostRouting } from "./useHostRouting";
import { setGenerationHandoff } from "./useGenerationHandoff";
import { toast } from "../lib/toasts";

/**
 * The browser's half of opening server-owned work: which machines it can
 * reach, how it says something went wrong, how it hands settings to Create,
 * and where a chain row and a download row go. The decision itself — the
 * chain guard, the queue lookup, the 3-D workflow route — is the shared
 * `openLiveWorkWith`.
 */
export function useOpenLiveWork(routing: HostRouting) {
  const router = useRouter();

  return openLiveWorkWith<OutputMetadata>({
    targetFor: (hostId) => {
      const host = routing.hosts.value.find(
        (candidate) => candidate.id === hostId,
      );
      return host ? { baseUrl: host.url, apiKey: host.apiKey ?? null } : null;
    },
    go: (to) => router.push(to).then(() => undefined),
    fail: (message) => toast("error", message),
    restore: (selection, hostId) =>
      setGenerationHandoff({
        metadata: selection.metadata,
        seedPinned: true,
        queueSelection: {
          hostId,
          jobId: selection.jobId,
          running: selection.running,
        },
      }),
    // A chain row goes to its machine: the browser's queue page cannot
    // re-enter the work either, and the machine is where its progress is.
    chainDestination: (row) => `/machines/${row.hostId}`,
    // Downloads are a popover in the shell, not a page.
    openDownloads: () => {
      window.dispatchEvent(new CustomEvent("mold:open-downloads"));
      return null;
    },
  });
}
