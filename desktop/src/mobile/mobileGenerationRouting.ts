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
import {
  generationHostSubmissionPolicy,
  type GenerationHostSubmissionPolicy,
  type GenerationTargetPolicy,
} from "@studio/lib/generationSubmissionPolicy";
import { modelPresenceOnHost } from "@studio/lib/modelInstallTargets";
import type { MobileHost } from "./hosts";

/** One immutable routing candidate assembled by the mobile surface. */
export interface MobileGenerationRoutingCandidate {
  host: MobileHost;
  view: CapableHostBase;
}

/** One machine's answer to the automatic-routing fan-out. */
interface MobileRoutingObservation {
  host: MobileHost;
  /** Exact route captured immediately before this machine is probed. */
  route: HostRoute;
  view: CapableHostBase;
  roundTripMs: number;
  preview: GenerationPlacementPreview | null;
  error: unknown;
  telemetryOnly: boolean;
}

export type MobileAutomaticRoute =
  | {
      kind: "route";
      host: MobileHost;
      route: HostRoute;
      placement: GenerationPlacementPreview | null;
    }
  | { kind: "missing_model"; host: MobileHost; route: HostRoute; model: string }
  | { kind: "error"; message: string }
  | { kind: "abandoned" };

export type MobilePinnedPlacement =
  | {
      kind: "placement";
      placement: GenerationPlacementPreview | null;
    }
  | { kind: "missing_model"; model: string }
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
): string {
  const classification = classifyPlacementPreview(preview);
  if (classification === "infeasible" && preview) {
    const missing = (preview.missing_components ?? [])
      .filter((component) => !component.present)
      .map((component) => component.name);
    const reason =
      typeof preview.reason === "string" && preview.reason.trim()
        ? sentence(preview.reason.trim())
        : sentence("the server reported that this print is infeasible");
    return `${hostLabel} cannot run this print: ${reason}${missing.length ? ` Missing components: ${missing.join(", ")}.` : ""} Nothing was queued.`;
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

function mobileFleetPlacementFailure(probes: readonly MobileRoutingObservation[]): string {
  if (probes.length === 1 && probes[0]!.preview) {
    return mobilePlacementFailure(probes[0]!.preview, probes[0]!.host.name);
  }
  const detail = probes
    .map((probe) =>
      probe.preview
        ? mobilePlacementFailure(probe.preview, probe.host.name).replace(" Nothing was queued.", "")
        : `${probe.host.name} did not answer: ${describeTransportError(probe.error, probe.host.name)}`,
    )
    .join(" ");
  return `No connected machine could run this print. ${detail} Nothing was queued.`;
}

export interface RouteAutomaticMobileGenerationOptions {
  candidates: readonly MobileGenerationRoutingCandidate[];
  routeForHost: (host: MobileHost) => HostRoute;
  policy: string;
  request: Record<string, unknown>;
  chain: boolean;
  copies: number;
  requireAuthoritative: boolean;
  isCurrent?: () => boolean;
  signal?: AbortSignal;
  settleMs?: number;
  model?: string;
  modelOwnerIds?: readonly string[];
  inventoryKnown?: (hostId: string) => boolean;
}

export interface PreviewPinnedMobileGenerationOptions {
  route: HostRoute;
  request: Record<string, unknown>;
  chain: boolean;
  copies: number;
  requireAuthoritative: boolean;
  isCurrent?: () => boolean;
  signal?: AbortSignal;
  model?: string;
  modelOwnerIds?: readonly string[];
  inventoryKnown?: boolean;
}

function automaticTargetPolicy(policy: string): GenerationTargetPolicy {
  return policy === CAPABLE_TARGET_ID ? { kind: "capable" } : { kind: "auto" };
}

/** Translate a frozen mobile route into the shared capability contract. */
export function mobileGenerationSubmissionPolicy(options: {
  route: HostRoute;
  request: Record<string, unknown>;
  chain: boolean;
  target: GenerationTargetPolicy;
}): GenerationHostSubmissionPolicy {
  return generationHostSubmissionPolicy(
    options.target,
    {
      hostId: options.route.hostId,
      queue: {
        heterogeneous_batch_max_outputs: options.route.heterogeneousBatchMaxOutputs ?? null,
      },
      durableMedia: options.route.durableMedia ?? null,
    },
    options.chain ? "sequence" : "generation",
  );
}

/**
 * Validate one frozen machine only when the request needs an authoritative
 * capability fence. Ordinary pinned work has no routing decision to make, so
 * it goes straight to admission; the server queues it before expensive model
 * preparation and remains authoritative for every execution check.
 */
export async function previewPinnedMobileGeneration(
  options: PreviewPinnedMobileGenerationOptions,
): Promise<MobilePinnedPlacement> {
  const isCurrent = options.isCurrent ?? (() => true);
  if (!isCurrent()) return { kind: "abandoned" };
  const submission = mobileGenerationSubmissionPolicy({
    route: options.route,
    request: options.request,
    chain: options.chain,
    target: { kind: "pinned", hostId: options.route.hostId },
  });
  if (submission.routing === "none") {
    if (
      modelPresenceOnHost(
        options.route.hostId,
        options.modelOwnerIds ?? [],
        options.inventoryKnown ?? false,
      ) === "missing"
    ) {
      return {
        kind: "missing_model",
        model: options.model ?? String(options.request.model ?? ""),
      };
    }
    return { kind: "placement", placement: null };
  }
  let placement: GenerationPlacementPreview | null = null;
  try {
    placement = options.chain
      ? await previewChainPlacement(options.route.target, options.request, options.copies, {
          ...(options.signal ? { signal: options.signal } : {}),
        })
      : await previewGenerationPlacement(options.route.target, options.request, options.copies, {
          ...(options.signal ? { signal: options.signal } : {}),
        });
  } catch (error) {
    if (!isCurrent()) return { kind: "abandoned" };
    return {
      kind: "error",
      message: describeTransportError(error, options.route.label),
    };
  }
  if (!isCurrent()) return { kind: "abandoned" };
  const classification = classifyPlacementPreview(placement);
  if (options.requireAuthoritative && classification === "unsupported") {
    return {
      kind: "error",
      message: `${options.route.label} does not provide the authoritative placement preview required for reference media. Nothing was queued.`,
    };
  }
  if (classification !== "unsupported" && classification !== "planned") {
    return {
      kind: "error",
      message: mobilePlacementFailure(placement, options.route.label),
    };
  }
  return { kind: "placement", placement };
}

/**
 * Ask every eligible machine for a placement plan and freeze one exact route.
 * Candidate eligibility remains a surface concern; this module owns only the
 * asynchronous fan-out and deterministic winner selection.
 */
export async function routeAutomaticMobileGeneration(
  options: RouteAutomaticMobileGenerationOptions,
): Promise<MobileAutomaticRoute> {
  const isCurrent = options.isCurrent ?? (() => true);
  const carriesIdentity = Boolean(options.request.id_image);
  const probes: MobileRoutingObservation[] = [];
  const knownMissing: MobileRoutingObservation[] = [];
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
        const submission = mobileGenerationSubmissionPolicy({
          route,
          request: options.request,
          chain: options.chain,
          target: automaticTargetPolicy(options.policy),
        });
        if (submission.admission === "refused" && submission.routing !== "placement_preview") {
          // The machine cannot admit this print at all; keep it out of the
          // ranking rather than letting Develop throw at submit.
          probes.push({
            ...candidate,
            route,
            roundTripMs: elapsed(),
            preview: null,
            error: submission.refusal ?? "this machine cannot admit the print",
            telemetryOnly: false,
          });
          return;
        }
        if (
          submission.routing === "telemetry_only" ||
          (submission.routing === "none" && !options.chain)
        ) {
          const observation = {
            ...candidate,
            route,
            roundTripMs: elapsed(),
            preview: null,
            error: null,
            telemetryOnly: true,
          };
          if (
            modelPresenceOnHost(
              candidate.host.id,
              options.modelOwnerIds ?? [],
              options.inventoryKnown?.(candidate.host.id) ?? false,
            ) === "missing"
          ) {
            knownMissing.push(observation);
          } else {
            probes.push(observation);
            resolveFirstPlanned();
          }
          return;
        }
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
          telemetryOnly: false,
        });
        if (classifyPlacementPreview(preview) === "planned") resolveFirstPlanned();
      } catch (probeError) {
        probes.push({
          ...candidate,
          route,
          roundTripMs: elapsed(),
          preview: null,
          error: probeError,
          telemetryOnly: false,
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
  // An ordinary print is answered from the captured queue/GPU snapshot and an
  // auto-chained long video from its placement plan; one fan-out is never a
  // mix of the two.
  const telemetryOnly = settledProbes.filter((probe) => probe.telemetryOnly);
  if (telemetryOnly.length > 0) {
    const views = telemetryOnly.map((probe) => probe.view);
    const chosen =
      options.policy === CAPABLE_TARGET_ID
        ? pickMostCapableHost(views, null, { lowestIdWins: true })
        : pickAutoHost(views, { lowestIdWins: true });
    if (chosen) {
      const probe = telemetryOnly.find((entry) => entry.view.id === chosen.id)!;
      return {
        kind: "route",
        host: probe.host,
        route: probe.route,
        placement: probe.preview,
      };
    }
  }
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
    };
  }
  if (knownMissing.length > 0) {
    const views = knownMissing.map((probe) => probe.view);
    const selected =
      options.policy === CAPABLE_TARGET_ID
        ? pickMostCapableHost(views, null, { lowestIdWins: true })
        : pickAutoHost(views, { lowestIdWins: true });
    const probe = selected
      ? knownMissing.find((entry) => entry.host.id === selected.id)
      : knownMissing[0];
    if (probe) {
      return {
        kind: "missing_model",
        host: probe.host,
        route: probe.route,
        model: options.model ?? String(options.request.model ?? ""),
      };
    }
  }

  // An `unsupported` plan is a NON-AUTHORITATIVE answer, not an old machine:
  // chain and local utility plans are documented to answer it, so the machine
  // stays routable when the caller does not require authority.
  const nonAuthoritative = settledProbes.filter(
    (probe) => classifyPlacementPreview(probe.preview) === "unsupported",
  );
  if (!options.requireAuthoritative && !carriesIdentity && nonAuthoritative.length > 0) {
    const views = nonAuthoritative.map((probe) => probe.view);
    const fallback =
      options.policy === CAPABLE_TARGET_ID
        ? pickMostCapableHost(views, null, { lowestIdWins: true })
        : pickAutoHost(views, { lowestIdWins: true });
    if (fallback) {
      const probe = nonAuthoritative.find((entry) => entry.host.id === fallback.id)!;
      return {
        kind: "route",
        host: probe.host,
        route: probe.route,
        placement: null,
      };
    }
  }
  return {
    kind: "error",
    message: mobileFleetPlacementFailure(settledProbes),
  };
}
