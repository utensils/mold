import { computed, ref, watch, type ComputedRef } from "vue";
import {
  planningVideoRequest,
  validateVideoJob,
  type VideoJobOwner,
} from "../api/videoJobs";
import type { VideoJobPlan, VideoJobRequest } from "../lib/api/videoJobTypes";

/** Read-only scalar preview. The full prepared request gets a new validation
 * proof at Generate; this display plan can never authorize admission. */
export function createVideoDurationPlan<Generation extends object>(
  enabled: ComputedRef<boolean>,
  owner: ComputedRef<VideoJobOwner | null>,
  request: ComputedRef<VideoJobRequest<Generation> | null>,
) {
  const plan = ref<VideoJobPlan | null>(null),
    checking = ref(false),
    error = ref("");
  let revision = 0,
    disposed = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let controller: AbortController | null = null;
  // Primitive identity avoids resetting a valid plan merely because a shell
  // projects an equivalent fresh route object on a render.
  const stamp = computed(() => {
    const machine = owner.value,
      input = request.value;
    if (!enabled.value || !machine?.instanceId || !input) return null;
    return JSON.stringify([
      machine.hostId,
      machine.instanceId,
      machine.target.baseUrl,
      machine.target.apiKey,
      machine.target.connectionMode,
      planningVideoRequest(input),
    ]);
  });
  const stop = watch(
    stamp,
    (value) => {
      const attempt = ++revision;
      if (timer) clearTimeout(timer);
      controller?.abort();
      controller = null;
      plan.value = null;
      error.value = "";
      checking.value = false;
      if (value === null || disposed || !owner.value || !request.value) return;
      const captured = { ...owner.value, target: { ...owner.value.target } };
      // The projection contains only literal JSON scalars/scene strings. Clone
      // that projection to unwrap nested Vue proxies without copying media.
      const input = JSON.parse(
        JSON.stringify(planningVideoRequest(request.value)),
      ) as VideoJobRequest;
      checking.value = true;
      const isCurrent = () =>
        !disposed && attempt === revision && stamp.value === value;
      timer = setTimeout(async () => {
        const active = new AbortController();
        controller = active;
        try {
          const result = await validateVideoJob(captured, input, {
            planningOnly: true,
            isCurrent,
            signal: active.signal,
          });
          if (isCurrent()) plan.value = result;
        } catch (cause) {
          if (isCurrent())
            error.value =
              cause instanceof Error ? cause.message : String(cause);
        } finally {
          if (isCurrent()) checking.value = false;
        }
      }, 200);
    },
    { immediate: true },
  );
  function dispose() {
    disposed = true;
    revision += 1;
    stop();
    if (timer) clearTimeout(timer);
    controller?.abort();
    controller = null;
    checking.value = false;
  }
  return { plan, checking, error, dispose };
}
