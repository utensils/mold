import { ApiError } from "../lib/api/client";
import { describeTransportError } from "../lib/api/errors";
import type { HostRoute } from "../stores/hosts";
import {
  classifyPlacementPreview,
  comparePlacementPreviews,
  previewChainPlacement,
  previewGenerationPlacement,
  type GenerationPlacementPreview,
} from "@studio/api/generationPlacement";
import {
  CAPABLE_TARGET_ID,
  chooseRoutedHost,
  pickAutoHost,
  pickMostCapableHost,
  type CapableHostBase,
} from "@studio/lib/hostRouting";
import type { MobileHost } from "./hosts";

/** One immutable routing candidate assembled by the mobile surface. */
export interface MobileGenerationRoutingCandidate {
  host: MobileHost;
  view: CapableHostBase;
}

/** One machine's answer to the automatic-routing fan-out. */
interface MobilePlacementProbe {
  host: MobileHost;
  /** Exact route captured immediately before this machine is probed. */
  route: HostRoute;
  view: CapableHostBase;
  roundTripMs: number;
  preview: GenerationPlacementPreview | null;
  error: unknown;
  legacyUnsupported: boolean;
}

export type MobileAutomaticRoute =
  | {
      kind: "route";
      host: MobileHost;
      route: HostRoute;
      placement: GenerationPlacementPreview | null;
      legacyUnsupported: boolean;
    }
  | { kind: "error"; message: string }
  | { kind: "abandoned" };

/** Default grace after the first machine produces a usable plan. */
export const MOBILE_PLACEMENT_SETTLE_MS = 1_500;

function sentence(text: string): string {
  return /[.!?]$/.test(text) ? text : `${text}.`;
}

export function mobilePlacementFailure(
  preview: GenerationPlacementPreview | null,
  hostLabel: string,
  subject: "print" | "sequence",
): string {
  const classification = classifyPlacementPreview(preview);
  if (classification === "infeasible" && preview) {
    const missing = (preview.missing_components ?? [])
      .filter((component) => !component.present)
      .map((component) => component.name);
    const reason =
      typeof preview.reason === "string" && preview.reason.trim()
        ? sentence(preview.reason.trim())
        : sentence(`the server reported that this ${subject} is infeasible`);
    return `${hostLabel} cannot run this ${subject}: ${reason}${missing.length ? ` Missing components: ${missing.join(", ")}.` : ""} Nothing was queued.`;
  }
  if (classification === "temporarily_unavailable") {
    const reason =
      typeof preview?.reason === "string" && preview.reason.trim()
        ? ` Reason: ${sentence(preview.reason.trim())}`
        : "";
    return `${hostLabel} could not compute a placement plan right now.${reason} Try again. Nothing was queued.`;
  }
  return `${hostLabel} returned an invalid placement response. Nothing was queued.`;
}

function mobileFleetPlacementFailure(
  probes: readonly MobilePlacementProbe[],
  subject: "print" | "sequence",
): string {
  if (probes.length === 1 && probes[0]!.preview) {
    return mobilePlacementFailure(probes[0]!.preview, probes[0]!.host.name, subject);
  }
  const detail = probes
    .map((probe) =>
      probe.preview
        ? mobilePlacementFailure(probe.preview, probe.host.name, subject).replace(
            " Nothing was queued.",
            "",
          )
        : `${probe.host.name} did not answer: ${describeTransportError(probe.error, probe.host.name)}`,
    )
    .join(" ");
  return `No connected machine could run this ${subject}. ${detail} Nothing was queued.`;
}

export interface RouteAutomaticMobileGenerationOptions {
  candidates: readonly MobileGenerationRoutingCandidate[];
  routeForHost: (host: MobileHost) => HostRoute;
  policy: string;
  request: Record<string, unknown>;
  chain: boolean;
  copies: number;
  subject: "print" | "sequence";
  requireAuthoritative: boolean;
  isCurrent?: () => boolean;
  signal?: AbortSignal;
  settleMs?: number;
}

/**
 * Ask every eligible machine for a placement plan and freeze one exact route.
 * Candidate eligibility remains a surface concern; this module owns only the
 * asynchronous fan-out, legacy policy, and deterministic winner selection.
 */
export async function routeAutomaticMobileGeneration(
  options: RouteAutomaticMobileGenerationOptions,
): Promise<MobileAutomaticRoute> {
  const isCurrent = options.isCurrent ?? (() => true);
  const carriesIdentity = Boolean(options.request.id_image);
  const probes: MobilePlacementProbe[] = [];
  const controllers = options.candidates.map(() => new AbortController());
  let pending = options.candidates.length;
  let resolveAllSettled!: () => void;
  let resolveFirstPlanned!: () => void;
  const allSettled = new Promise<void>((resolve) => (resolveAllSettled = resolve));
  const firstPlanned = new Promise<void>((resolve) => (resolveFirstPlanned = resolve));

  options.candidates.forEach((candidate, index) => {
    void (async () => {
      const controller = controllers[index]!;
      const abortFromCaller = () => controller.abort(options.signal?.reason);
      if (options.signal?.aborted) abortFromCaller();
      else options.signal?.addEventListener("abort", abortFromCaller, { once: true });
      const started = performance.now();
      const elapsed = () => Math.max(0, performance.now() - started);
      const probeOptions = { signal: controller.signal };
      const route = options.routeForHost(candidate.host);
      const probeTarget = { ...route.target };
      try {
        const preview = options.chain
          ? await previewChainPlacement(probeTarget, options.request, options.copies, probeOptions)
          : await previewGenerationPlacement(
              probeTarget,
              options.request,
              options.copies,
              probeOptions,
            );
        probes.push({
          ...candidate,
          route,
          roundTripMs: elapsed(),
          preview,
          error: null,
          legacyUnsupported: false,
        });
        if (classifyPlacementPreview(preview) === "planned") resolveFirstPlanned();
      } catch (probeError) {
        probes.push({
          ...candidate,
          route,
          roundTripMs: elapsed(),
          preview: null,
          error: probeError,
          legacyUnsupported:
            probeError instanceof ApiError &&
            (probeError.status === 404 || probeError.status === 405),
        });
      } finally {
        options.signal?.removeEventListener("abort", abortFromCaller);
        pending -= 1;
        if (pending === 0) resolveAllSettled();
      }
    })();
  });

  if (pending === 0) resolveAllSettled();
  await Promise.race([
    allSettled,
    ...(options.candidates.length > 1
      ? [
          firstPlanned.then(
            () =>
              new Promise<void>((resolve) =>
                setTimeout(resolve, options.settleMs ?? MOBILE_PLACEMENT_SETTLE_MS),
              ),
          ),
        ]
      : []),
  ]);
  if (pending > 0) for (const controller of controllers) controller.abort();
  if (!isCurrent()) return { kind: "abandoned" };

  const settledProbes = probes.slice();
  const planned = settledProbes.flatMap((probe) =>
    probe.preview && classifyPlacementPreview(probe.preview) === "planned"
      ? [{ host: probe.view, roundTripMs: probe.roundTripMs, probe }]
      : [],
  );
  const chosen = chooseRoutedHost(
    planned.map((entry) => ({
      host: entry.host,
      roundTripMs: entry.roundTripMs,
      preview: entry.probe.preview!,
    })),
    options.policy,
    comparePlacementPreviews,
    { lowestIdWins: true },
  );
  if (chosen) {
    const winner = planned.find((entry) => entry.host.id === chosen.id)!;
    return {
      kind: "route",
      host: winner.probe.host,
      route: winner.probe.route,
      placement: winner.probe.preview,
      legacyUnsupported: false,
    };
  }

  const legacy = settledProbes.filter(
    (probe) => probe.legacyUnsupported || classifyPlacementPreview(probe.preview) === "unsupported",
  );
  if (!options.requireAuthoritative && !carriesIdentity && legacy.length > 0) {
    const views = legacy.map((probe) => probe.view);
    const fallback =
      options.policy === CAPABLE_TARGET_ID
        ? pickMostCapableHost(views, null, { lowestIdWins: true })
        : pickAutoHost(views, { lowestIdWins: true });
    if (fallback) {
      const probe = legacy.find((entry) => entry.host.id === fallback.id)!;
      return {
        kind: "route",
        host: probe.host,
        route: probe.route,
        placement: null,
        legacyUnsupported: true,
      };
    }
  }
  return {
    kind: "error",
    message: mobileFleetPlacementFailure(settledProbes, options.subject),
  };
}
