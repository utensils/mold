import { expect, it } from "vitest";
import {
  reconcileMirrorOrganization,
  recoverMirrorOrganization,
  type MirrorOrganizationState,
} from "./mirrorOrganizationPolicy";
const source = {
  tags: ["Shared"],
  collections: ["album"],
  hidden: [],
  title: "Source",
  favorite: true,
};
const destination = {
  tags: ["Local"],
  collections: [],
  title: null,
  favorite: false,
};
it("removes only sync contributions while preserving destination-only organization", () => {
  const first = reconcileMirrorOrganization(source, destination);
  expect(first.desired.tags).toEqual(["Local", "Shared"]);
  const next = reconcileMirrorOrganization(
    { ...source, tags: [], collections: [], title: null, favorite: false },
    first.desired,
    first.state,
  );
  expect(next.desired.tags).toEqual(["Local"]);
  expect(next.desired.collections).toEqual([]);
  expect(next.desired.title).toBeNull();
  expect(next.desired.favorite).toBe(false);
});
it("preserves local removals and scalar overrides but hidden membership always reasserts", () => {
  const first = reconcileMirrorOrganization(source, destination);
  const locallyEdited = {
    ...first.desired,
    title: "Mine",
    favorite: false,
    tags: ["Local"],
    collections: [],
  };
  const second = reconcileMirrorOrganization(
    source,
    locallyEdited,
    first.state,
  );
  expect(second.desired).toEqual(locallyEdited);
  const hidden = reconcileMirrorOrganization(
    { ...source, hidden: ["album"] },
    locallyEdited,
    second.state,
  );
  expect(hidden.desired.collections).toEqual(["album"]);
  const explicit = reconcileMirrorOrganization(
    source,
    locallyEdited,
    second.state,
    { deliberate: true },
  );
  expect(explicit.desired.tags).toContain("Shared");
});
it("clears removal suppression only after a genuine source remove and re-add", () => {
  const first = reconcileMirrorOrganization(source, destination);
  const removed = { ...first.desired, tags: ["Local"] };
  const suppressed = reconcileMirrorOrganization(source, removed, first.state);
  const sourceRemoved = reconcileMirrorOrganization(
    { ...source, tags: [] },
    suppressed.desired,
    suppressed.state,
  );
  expect(
    reconcileMirrorOrganization(
      source,
      sourceRemoved.desired,
      sourceRemoved.state,
    ).desired.tags,
  ).toContain("Shared");
});
it("never retroactively owns destination baseline or removes another verified source contribution", () => {
  const priorExisting = {
    ...destination,
    tags: ["Shared"],
    collections: ["album"],
  };
  const first = reconcileMirrorOrganization(source, priorExisting);
  expect(first.state.ownedTags).toEqual([]);
  const removed = reconcileMirrorOrganization(
    { ...source, tags: [], collections: [] },
    first.desired,
    first.state,
  );
  expect(removed.desired.tags).toEqual(["Shared"]);
  const own = reconcileMirrorOrganization(source, destination);
  const other = {
    ...own.state,
    ownedTags: ["shared"],
    observedTags: ["shared"],
  } as MirrorOrganizationState;
  expect(
    reconcileMirrorOrganization(
      { ...source, tags: [] },
      own.desired,
      own.state,
      { contributors: [other] },
    ).desired.tags,
  ).toContain("Shared");
});
it("recovers only confirmed pending writes before the source changes again", () => {
  const first = reconcileMirrorOrganization(source, destination);
  const intent = {
    state: first.state,
    desired: first.desired,
    baseline: destination,
  };
  const recovered = recoverMirrorOrganization(undefined, intent, first.desired);
  const next = reconcileMirrorOrganization(
    { ...source, tags: [], collections: [] },
    first.desired,
    recovered,
  );
  expect(next.desired.tags).toEqual(["Local"]);
  expect(next.desired.collections).toEqual([]);
  const partial = recoverMirrorOrganization(undefined, intent, destination);
  expect(partial.ownedTags).toEqual([]);
  expect(Object.hasOwn(partial, "title")).toBe(false);
});
it("treats a local re-add after observed removal as destination-owned", () => {
  const first = reconcileMirrorOrganization(source, destination);
  const removed = reconcileMirrorOrganization(
    source,
    { ...first.desired, tags: ["Local"] },
    first.state,
  );
  const readded = reconcileMirrorOrganization(
    source,
    { ...removed.desired, tags: ["Local", "Shared"] },
    removed.state,
  );
  expect(readded.state.ownedTags).toEqual([]);
  expect(
    reconcileMirrorOrganization(
      { ...source, tags: [] },
      readded.desired,
      readded.state,
    ).desired.tags,
  ).toContain("Shared");
});
it("preserves and reports scalar conflicts between independent originals", () => {
  const first = reconcileMirrorOrganization(source, destination, undefined, {
    origin: "original-a",
  });
  const conflict = reconcileMirrorOrganization(
    { ...source, title: "Independent" },
    first.desired,
    undefined,
    { origin: "original-b", contributors: [first.state] },
  );
  expect(conflict.desired.title).toBe("Source");
  expect(conflict.state.conflicts).toContain(
    "Conflicting title from another original was preserved.",
  );
});

it("keeps observed scalar overrides through source clear and re-add cycles", () => {
  const first = reconcileMirrorOrganization(source, destination);
  const local = { ...first.desired, title: null, favorite: false };
  const override = reconcileMirrorOrganization(source, local, first.state);
  expect(override.state.titleOverride).toBe(true);
  expect(override.state.favoriteOverride).toBe(true);
  const cleared = reconcileMirrorOrganization(
    { ...source, title: null, favorite: false },
    local,
    override.state,
  );
  const restored = reconcileMirrorOrganization(source, local, cleared.state);
  expect(restored.desired.title).toBeNull();
  expect(restored.desired.favorite).toBe(false);
  // Matching an older applied value later is still an authored override.
  const matched = reconcileMirrorOrganization(
    { ...source, title: "Changed" },
    { ...local, title: "Source" },
    restored.state,
  );
  expect(matched.desired.title).toBe("Source");
});
