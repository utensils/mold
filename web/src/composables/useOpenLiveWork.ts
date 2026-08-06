import { useRouter } from "vue-router";
import type { FleetActiveWork } from "@studio/api/activity";
import { listQueue } from "@studio/api/queuePlan";
import type { OutputMetadata } from "../types";
import type { HostRouting } from "./useHostRouting";
import { setGenerationHandoff } from "./useGenerationHandoff";
import { setSequenceHandoff } from "./useSequenceHandoff";
import { toast } from "../lib/toasts";

/** Opens server-owned work in the web surface that can inspect or resume it. */
export function useOpenLiveWork(routing: HostRouting) {
  const router = useRouter();

  return async (row: FleetActiveWork) => {
    if (row.kind === "sequence") {
      setSequenceHandoff({
        kind: "inspect",
        hostId: row.hostId,
        jobId: row.id,
      });
      await router.push("/create");
      return;
    }
    if (row.kind === "generation") {
      const host = routing.hosts.value.find(
        (candidate) => candidate.id === row.hostId,
      );
      if (!host) {
        toast("error", "That machine is no longer connected.");
        return;
      }
      try {
        const queue = await listQueue({
          baseUrl: host.url,
          apiKey: host.apiKey ?? null,
        });
        const entry = queue.entries.find(
          (candidate) => candidate.id === row.id,
        );
        if (!entry?.metadata) {
          toast(
            "error",
            "This host cannot restore settings for that generation.",
          );
          return;
        }
        setGenerationHandoff({
          metadata: entry.metadata as OutputMetadata,
          seedPinned: entry.seed_pinned ?? null,
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
