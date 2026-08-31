import { useRouter } from "vue-router";
import type { FleetActiveWork } from "@studio/api/activity";
import { findQueueEntryById } from "@studio/api/queuePlan";
import { selectedQueueGeneration } from "@studio/api/generationSelection";
import type { OutputMetadata } from "../lib/api/types";
import { useComposerStore } from "../stores/composer";
import { useHostsStore } from "../stores/hosts";
import { useToastStore } from "../stores/toasts";

/** Opens server-owned work in the surface that can inspect or resume it. */
export function useOpenLiveWork() {
  const router = useRouter();
  const hosts = useHostsStore();
  const composer = useComposerStore();
  const toasts = useToastStore();

  return async (row: FleetActiveWork) => {
    if (row.kind === "sequence" || row.execution === "chain") {
      composer.setSequence({ kind: "inspect", hostId: row.hostId, jobId: row.id });
      await router.push("/create");
      return;
    }
    if (row.kind === "generation") {
      const host = hosts.all.find((candidate) => candidate.id === row.hostId);
      if (!host?.baseUrl) {
        toasts.push("That machine is no longer connected", "error");
        return;
      }
      try {
        const entry = await findQueueEntryById(
          { baseUrl: host.baseUrl, apiKey: host.apiKey },
          row.id,
        );
        const selection = selectedQueueGeneration<OutputMetadata>(entry ? [entry] : [], row.id);
        if (!selection) {
          toasts.push("This host cannot restore settings for that generation", "error");
          return;
        }
        composer.set({
          metadata: selection.metadata,
          queueSelection: {
            hostId: row.hostId,
            jobId: selection.jobId,
            running: selection.running,
          },
        });
        await router.push("/create");
      } catch (error) {
        toasts.push(error instanceof Error ? error.message : String(error), "error");
      }
      return;
    }
    await router.push(row.kind === "download" ? "/models" : `/machines/${row.hostId}`);
  };
}
