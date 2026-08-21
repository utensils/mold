/**
 * iPhone "File under" policy — the phone's half of the Create-time Library
 * filing contract.
 *
 * Everything about the DRAFT (ghost tag, collection match, reducers, request
 * fields) is `@studio/lib/fileUnder` and is never restated here. What is
 * genuinely phone-shaped is the capability question, because the phone has no
 * home machine: the print lands either on one pinned remote or on whichever
 * machine an automatic policy picks, and the group may only appear when a
 * machine that can actually file is in play.
 */
import { fileUnderAvailable, type FileUnderCollectionLike } from "@studio/lib/fileUnder";
import { isAutomaticTarget } from "@studio/lib/hostRouting";
import type { MobileCollectionCard } from "./libraryOrganization";
import { mobileRoutingHosts } from "./generateTarget";
import type { MobileHost } from "./hosts";

/**
 * Whether Create may offer the File under group for the target in force.
 *
 * A PINNED machine answers for itself. Under Auto / Most capable the print
 * lands on one of several, so one filing-capable machine among them is
 * enough — but `hosts` must then be the CANDIDATE set the fan-out will
 * actually choose from (model-aware and access-filtered), not the whole
 * fleet. A peer that can file but cannot run the selected checkpoint would
 * otherwise qualify the group for a print that routes elsewhere and lands
 * unfiled, with nothing said.
 *
 * Positive knowledge only — an unread or failed capability probe is not
 * evidence, so the group hides and nothing is filed.
 *
 * `capabilities` is deliberately loose: it is the same per-host snapshot
 * record the Library organization gate reads, and `fileUnderAvailable`
 * inspects it defensively.
 */
export function mobileFileUnderAvailable(
  target: string,
  hosts: readonly MobileHost[],
  capabilities: Record<string, unknown>,
): boolean {
  if (isAutomaticTarget(target)) {
    return mobileRoutingHosts(hosts).some((host) => fileUnderAvailable(capabilities[host.id]));
  }
  return target ? fileUnderAvailable(capabilities[target]) : false;
}

/** What the collection sheet renders and `matchCollection` matches against:
 * the fleet's merged collections reduced to the slug the hosts agree on plus
 * the count. No `id` — a mobile card is a cross-host merge, and the wire
 * carries the NAME so the routed machine resolves it by slug itself. */
export function mobileFileUnderCollections(
  cards: readonly MobileCollectionCard[],
): Array<FileUnderCollectionLike & { count: number }> {
  return cards.map((card) => ({ name: card.name, slug: card.slug, count: card.count }));
}
