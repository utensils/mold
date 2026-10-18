import { normalizeTagName, tagKey } from "./libraryOrganization";
export interface MirrorOrganizationValues {
  tags: string[];
  collections: string[];
  title: string | null;
  favorite: boolean;
}
export interface MirrorOrganizationSource extends MirrorOrganizationValues {
  hidden: string[];
}
export interface MirrorOrganizationState {
  version: 1;
  origin?: string;
  conflicts?: string[];
  titleOverride?: boolean;
  favoriteOverride?: boolean;
  observedTags: string[];
  observedCollections: string[];
  ownedTags: string[];
  ownedCollections: string[];
  suppressedTags: string[];
  suppressedCollections: string[];
  title?: string | null;
  favorite?: boolean;
}
/** All ownership is earned by a confirmed write; existing destination state is never adopted. */
export function reconcileMirrorOrganization(
  source: MirrorOrganizationSource,
  destination: MirrorOrganizationValues,
  prior?: MirrorOrganizationState,
  options: {
    deliberate?: boolean;
    contributors?: MirrorOrganizationState[];
    origin?: string;
    scalarAuthority?: boolean;
    scalarFields?: string[];
    observeOnly?: boolean;
  } = {},
): { desired: MirrorOrganizationValues; state: MirrorOrganizationState } {
  const contributors = options.contributors ?? [];
  const state: MirrorOrganizationState = {
    version: 1,
    ...(options.origin ? { origin: options.origin } : {}),
    observedTags: source.tags.map(tagKey),
    observedCollections: [...source.collections],
    ownedTags: [],
    ownedCollections: [],
    suppressedTags: [],
    suppressedCollections: [],
  };
  function membership(
    kind: "Tags" | "Collections",
    incoming: string[],
    current: string[],
    hidden: string[] = [],
  ): string[] {
    const key = kind === "Tags" ? tagKey : (value: string) => value;
    const desired = new Map(
      current.map((value) => [
        key(value),
        kind === "Tags" ? normalizeTagName(value) : value,
      ]),
    );
    const wanted = new Map(
      incoming.map((value) => [
        key(value),
        kind === "Tags" ? normalizeTagName(value) : value,
      ]),
    );
    const previousOwned = new Set(prior?.[`owned${kind}`] ?? []);
    const previousObserved = new Set(prior?.[`observed${kind}`] ?? []);
    const suppressed = new Set(
      (prior?.[`suppressed${kind}`] ?? []).filter((value) => wanted.has(value)),
    );
    for (const value of previousOwned) {
      if (
        !desired.has(value) &&
        wanted.has(value) &&
        previousObserved.has(value)
      )
        suppressed.add(value);
      if (
        !wanted.has(value) &&
        !contributors.some(
          (other) =>
            (!options.origin || other.origin !== options.origin) &&
            other[`observed${kind}`].includes(value),
        )
      )
        desired.delete(value);
    }
    const owned = new Set(
      [...previousOwned].filter(
        (value) => wanted.has(value) && desired.has(value),
      ),
    );
    for (const [value, display] of wanted) {
      if (
        !previousObserved.has(value) ||
        options.deliberate ||
        hidden.includes(value)
      )
        suppressed.delete(value);
      if (
        !desired.has(value) &&
        !suppressed.has(value) &&
        (!options.observeOnly || hidden.includes(value))
      ) {
        desired.set(value, display);
        owned.add(value);
      }
    }
    state[`owned${kind}`] = [...owned];
    state[`suppressed${kind}`] = [...suppressed];
    return [...desired.values()];
  }
  const desired: MirrorOrganizationValues = {
    ...destination,
    tags: membership("Tags", source.tags, destination.tags),
    collections: membership(
      "Collections",
      source.collections,
      destination.collections,
      source.hidden,
    ),
  };
  for (const field of ["title", "favorite"] as const) {
    const othersOwn = contributors.some(
      (other) =>
        Object.hasOwn(other, field) &&
        (!options.origin || other.origin !== options.origin),
    );
    const inheritedOwner = options.origin
      ? contributors.find(
          (other) =>
            other.origin === options.origin &&
            Object.hasOwn(other, field) &&
            destination[field] === other[field],
        )
      : undefined;
    const scalarOwner =
      prior && Object.hasOwn(prior, field) ? prior : inheritedOwner;
    const owns = !!scalarOwner;
    const overrideKey =
      field === "title" ? "titleOverride" : "favoriteOverride";
    const overridden =
      prior?.[overrideKey] === true ||
      (!!scalarOwner && destination[field] !== scalarOwner[field]);
    if (overridden) state[overrideKey] = true;
    const vacant =
      field === "title" ? !destination.title : !destination.favorite;
    if (othersOwn && source[field] !== destination[field])
      (state.conflicts ??= []).push(
        `Conflicting ${field} from another original was preserved.`,
      );
    if (
      !overridden &&
      options.scalarAuthority !== false &&
      (!options.scalarFields || options.scalarFields.includes(field)) &&
      !othersOwn &&
      ((owns && destination[field] === scalarOwner[field]) ||
        (!prior && vacant))
    ) {
      if (field === "title") {
        desired.title = source.title;
        state.title = source.title;
      } else {
        desired.favorite = source.favorite;
        state.favorite = source.favorite;
      }
    } else if (owns) {
      // Preserve the last applied value so an unchanged local override never
      // becomes a new source authority on later traversals.
      if (field === "title") state.title = scalarOwner.title ?? null;
      else state.favorite = scalarOwner.favorite ?? false;
    }
  }
  return { desired, state };
}

/** Durable intent records the pre-write baseline as well as the planned result. */
export interface MirrorOrganizationIntent {
  state: MirrorOrganizationState;
  baseline: MirrorOrganizationValues;
  desired: MirrorOrganizationValues;
}
export function recoverMirrorOrganization(
  prior: MirrorOrganizationState | undefined,
  pending: MirrorOrganizationIntent,
  current: MirrorOrganizationValues,
): MirrorOrganizationState {
  const state: MirrorOrganizationState = { ...pending.state };
  for (const kind of ["Tags", "Collections"] as const) {
    const key = kind === "Tags" ? tagKey : (value: string) => value;
    const field = kind === "Tags" ? "tags" : "collections";
    const actual = new Set(current[field].map(key)),
      wanted = new Set(pending.desired[field].map(key));
    const old = new Set(prior?.[`owned${kind}`] ?? []);
    // An unconfirmed removal retains ownership so it can be repaired. New
    // ownership is earned only when its journaled addition is now visible.
    const owned = new Set(
      [...old].filter((value) => actual.has(value) || wanted.has(value)),
    );
    for (const value of pending.state[`owned${kind}`])
      if (actual.has(value)) owned.add(value);
    state[`owned${kind}`] = [...owned];
  }
  for (const field of ["title", "favorite"] as const) {
    delete state[field];
    if (
      Object.hasOwn(pending.state, field) &&
      current[field] === pending.desired[field]
    ) {
      if (field === "title") state.title = pending.state.title ?? null;
      else state.favorite = pending.state.favorite ?? false;
    } else if (prior && Object.hasOwn(prior, field)) {
      if (field === "title") state.title = prior.title ?? null;
      else state.favorite = prior.favorite ?? false;
    }
  }
  return state;
}
