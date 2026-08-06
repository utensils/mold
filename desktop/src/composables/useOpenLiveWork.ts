import { useRouter } from "vue-router";
import type { FleetActiveWork } from "@studio/api/activity";
import { listQueue } from "@studio/api/queuePlan";
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
    if (row.kind === "sequence") {
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
        const queue = await listQueue({ baseUrl: host.baseUrl, apiKey: host.apiKey });
        const entry = queue.entries.find((candidate) => candidate.id === row.id);
        if (!entry?.metadata) {
          toasts.push("This host cannot restore settings for that generation", "error");
          return;
        }
        const metadata = entry.metadata as OutputMetadata;
        composer.set({
          metadata:
            entry.seed_pinned === false
              ? ({ ...metadata, seed: null } as unknown as OutputMetadata)
              : metadata,
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
