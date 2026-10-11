import type { StreamTarget } from "../api";
import type { HostRoute } from "./hostRouting";
import { getHost, ORIGIN_HOST_ID } from "./hostRegistry";
import { originApiTarget } from "./originAuth";
import { standaloneStudio, studioPrimary } from "./secureStudio";

/** The hosted origin is a UI alias, never durable job ownership. */
export function freezeGenerationRoute(
  route: HostRoute | null,
): HostRoute | null {
  if (!standaloneStudio() || (route && route.hostId !== ORIGIN_HOST_ID))
    return route;
  const primary = studioPrimary();
  if (
    !primary ||
    (route?.instanceId && route.instanceId !== primary.instanceId) ||
    (route?.target.baseUrl && route.target.baseUrl !== primary.url)
  )
    throw new Error("The original machine is no longer selected.");
  return {
    ...route,
    hostId: primary.id,
    label: primary.name,
    target: { baseUrl: primary.url },
    instanceId: primary.instanceId,
  };
}

/** Undefined means unavailable, never permission to use the current origin. */
export function generationJobTarget(job: {
  hostId: string | null;
  hostInstanceId?: string;
  target: StreamTarget | null;
  durableBatch?: { expectedInstanceId: string };
}): StreamTarget | undefined {
  const hosted = standaloneStudio();
  const origin = !job.hostId || job.hostId === ORIGIN_HOST_ID;
  // Older hosted records used an alias that can now name another machine.
  // A concrete instance is the only evidence permitting its resolution.
  const expected = job.hostInstanceId ?? job.durableBatch?.expectedInstanceId;
  if (hosted && origin) {
    const primary = studioPrimary();
    if (!expected || primary?.instanceId !== expected) return undefined;
    return { baseUrl: primary.url };
  }
  const host = !origin ? getHost(job.hostId!) : null;
  if (host && expected && host.instanceId !== expected) return undefined;
  if (hosted && !host) return undefined;
  if (job.target) {
    if (
      hosted &&
      (!job.target.baseUrl || job.target.baseUrl === window.location.origin)
    )
      return { baseUrl: host!.url };
    if (
      !job.target.apiKey &&
      (!job.target.baseUrl || job.target.baseUrl === window.location.origin)
    ) {
      const key = originApiTarget().apiKey;
      return { ...job.target, ...(key ? { apiKey: key } : {}) };
    }
    return job.target;
  }
  if (origin) {
    const local = originApiTarget();
    return {
      baseUrl: local.baseUrl,
      ...(local.apiKey ? { apiKey: local.apiKey } : {}),
    };
  }
  if (!host) return undefined;
  return { baseUrl: host.url, ...(host.apiKey ? { apiKey: host.apiKey } : {}) };
}
