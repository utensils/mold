import { useRouter } from "vue-router";
import type { FleetActiveWork } from "@studio/api/activity";
import { findQueueEntryById } from "@studio/api/queuePlan";
import { selectedQueueGeneration } from "@studio/api/generationSelection";
import type { OutputMetadata } from "../types";
import type { HostRouting } from "./useHostRouting";
import { setGenerationHandoff } from "./useGenerationHandoff";
import { toast } from "../lib/toasts";

/**
 * Opens server-owned work in the web surface that can inspect or resume it.
 *
 * A server-side chain row (a long video another client is auto-chaining, or a
 * `mold run --script` job) has no web surface any more: it falls through to
 * its machine's page rather than pretending Create can reattach to it.
 */
export function useOpenLiveWork(routing: HostRouting) {
  const router = useRouter();

  return async (row: FleetActiveWork) => {
    if (row.kind === "generation") {
      const host = routing.hosts.value.find(
        (candidate) => candidate.id === row.hostId,
      );
      if (!host) {
        toast("error", "That machine is no longer connected.");
        return;
      }
      try {
        const entry = await findQueueEntryById(
          { baseUrl: host.url, apiKey: host.apiKey ?? null },
          row.id,
        );
        const selection = selectedQueueGeneration<OutputMetadata>(
          entry ? [entry] : [],
          row.id,
        );
        if (!selection) {
          toast(
            "error",
            "This host cannot restore settings for that generation.",
          );
          return;
        }
        setGenerationHandoff({
          metadata: selection.metadata,
          seedPinned: true,
          queueSelection: {
            hostId: row.hostId,
            jobId: selection.jobId,
            running: selection.running,
          },
        });
        await router.push("/create");
      } catch (error) {
        toast("error", error instanceof Error ? error.message : String(error));
      }
      return;
    }
    if (row.kind === "download") {
      window.dispatchEvent(new CustomEvent("mold:open-downloads"));
      return;
    }
    await router.push(`/machines/${row.hostId}`);
  };
}
