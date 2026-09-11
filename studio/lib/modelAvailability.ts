/**
 * The style picker's availability tag — ONE rule for desktop, web and the phone.
 *
 * The tag exists to say WHERE a style is when that is not obvious, so it is
 * quiet in the two cases that tell the reader nothing: no reachable machine has
 * it (there is nothing to point at) and every reachable machine has it
 * (pointing everywhere is pointing nowhere, and a single-machine fleet is
 * always one of these two). Otherwise it names the one machine that has it, or
 * counts them.
 *
 * There is deliberately NO home/primary concept. Desktop's retired rule was
 * keyed on the primary host, which web does not have at all (only an origin
 * id) and the phone has no analogue for, so it could not be shared; "quiet when
 * everyone has it" generalizes to all three and says the same thing on a
 * desktop whose local engine holds the style alongside every remote.
 *
 * Callers pre-narrow `hosts` to the machines they can actually reach, because
 * each surface spells reachability differently — desktop and web as a three
 * state `status`, the phone as `connected`/`online`/`instanceMismatch` — and
 * `studio/` may not import any of their host types
 * (`scripts/tests/frontend-architecture.sh` keeps studio shell-independent).
 * The structural `{ id, label }` slice is all the rule needs, the way
 * `StyleMenuModel` is all a row needs.
 *
 * Whether to ask at all stays with the caller too: every surface suppresses the
 * tag outside multi-machine routing, and each spells that differently.
 */
export function modelAvailabilityTag(
  hostIds: readonly string[],
  hosts: ReadonlyArray<{ id: string; label: string }>,
): string | null {
  const known = hosts.filter((host) => hostIds.includes(host.id));
  if (known.length === 0 || known.length === hosts.length) return null;
  if (known.length === 1) return known[0]!.label;
  return `${known.length} machines`;
}
