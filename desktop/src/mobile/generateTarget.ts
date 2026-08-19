/**
 * iPhone generation-target policy.
 *
 * The phone is remote-only, so unlike desktop there is no home machine that
 * wins a dead heat and no local engine to fall back to. A target is therefore
 * one of three things: a pinned host id (today's behaviour), `auto`
 * (model-aware least busy), or `capable` (strongest GPU). The two automatic
 * policies are offered ONLY when at least two connected machines are reachable
 * — with a single machine there is nothing to choose between, and an automatic
 * label would be a lie.
 *
 * Routing policy itself lives in `@studio/lib/hostRouting`; this module owns
 * only the persisted preference, its visibility rule, and the labels. Nothing
 * here touches an API key: keys stay in the iOS Keychain and reach a request
 * exclusively through `mobileHostTarget`.
 */
import {
  AUTO_TARGET_ID,
  CAPABLE_TARGET_ID,
  isAutomaticTarget,
  normalizeTargetId,
} from "@studio/lib/hostRouting";
import type { MobileHost } from "./hosts";

export const MOBILE_GENERATE_TARGET_KEY = "mold.mobile.generate-target.v1";

/** At least this many reachable connected machines before Auto is offered. */
export const MOBILE_AUTO_ROUTING_MIN_HOSTS = 2;

type TargetStorage = Pick<Storage, "getItem" | "setItem">;

function defaultStorage(): TargetStorage | null {
  return typeof localStorage === "undefined" ? null : localStorage;
}

/** The connected machines an automatic policy may dispatch to. */
export function mobileRoutingHosts(hosts: readonly MobileHost[]): MobileHost[] {
  return hosts.filter((host) => host.connected !== false && host.online);
}

/** Auto and Most capable appear only with two or more reachable machines. */
export function mobileAutoRoutingAvailable(hosts: readonly MobileHost[]): boolean {
  return mobileRoutingHosts(hosts).length >= MOBILE_AUTO_ROUTING_MIN_HOSTS;
}

export function loadMobileGenerateTarget(storage: TargetStorage | null = defaultStorage()): string {
  const saved = storage?.getItem(MOBILE_GENERATE_TARGET_KEY)?.trim();
  return saved ? saved : AUTO_TARGET_ID;
}

export function saveMobileGenerateTarget(
  value: string,
  storage: TargetStorage | null = defaultStorage(),
): void {
  try {
    storage?.setItem(MOBILE_GENERATE_TARGET_KEY, value);
  } catch {
    // The preference is a convenience; a full storage quota must never stop a
    // print from being developed.
  }
}

/**
 * The policy actually in force.
 *
 * A saved automatic policy degrades to the currently browsed machine while
 * fewer than two machines are reachable — the saved value is kept, so plugging
 * a second machine back in restores Auto without the user re-picking it. A
 * pinned host that is no longer connected degrades to Auto when Auto is
 * available and to the browsed machine otherwise.
 */
export function resolveMobileGenerateTarget(
  saved: string | null | undefined,
  hosts: readonly MobileHost[],
  browsedHostId: string,
): string {
  const connected = hosts.filter((host) => host.connected !== false);
  const automaticAvailable = mobileAutoRoutingAvailable(hosts);
  const normalized = normalizeTargetId(saved, connected);
  if (!isAutomaticTarget(normalized)) return normalized;
  if (automaticAvailable) return normalized;
  return connected.some((host) => host.id === browsedHostId)
    ? browsedHostId
    : (connected[0]?.id ?? "");
}

/** Picker label for a policy value. Host ids resolve to the machine's name. */
export function mobileGenerateTargetLabel(value: string, hosts: readonly MobileHost[]): string {
  if (value === AUTO_TARGET_ID) return "Auto";
  if (value === CAPABLE_TARGET_ID) return "Most capable";
  return hosts.find((host) => host.id === value)?.name ?? value;
}

/** One line each, shown wherever the phone explains where work lands. */
export const MOBILE_AUTO_ROUTING_HINT =
  "Auto sends each print to the least busy machine that already has the model.";
export const MOBILE_CAPABLE_ROUTING_HINT =
  "Most capable sends it to the strongest GPU that has the model (CUDA before Metal, then VRAM).";

/**
 * Per-model availability for the union picker: which connected machines hold
 * this model. Quiet when every reachable machine has it — the tag exists to
 * say where a model is, not to repeat what is everywhere.
 */
export function mobileModelAvailabilityTag(
  hostIds: readonly string[],
  hosts: readonly MobileHost[],
): string | null {
  const reachable = mobileRoutingHosts(hosts);
  const known = reachable.filter((host) => hostIds.includes(host.id));
  if (known.length === 0 || known.length === reachable.length) return null;
  if (known.length === 1) return known[0]!.name;
  return `${known.length} machines`;
}
