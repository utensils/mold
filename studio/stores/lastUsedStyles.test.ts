import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import {
  LAST_USED_STYLES_KEY,
  setLastUsedStylesStorage,
  useLastUsedStylesStore,
  type LastUsedStylesStorage,
} from "./lastUsedStyles";

function memoryStorage(): LastUsedStylesStorage & {
  data: Map<string, string>;
} {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => void data.set(key, value),
    removeItem: (key) => void data.delete(key),
  };
}

const flux = { name: "flux-dev:q8", family: "flux" };
const flux2 = { name: "flux2-klein-9b:q8", family: "flux2" };
const ltx = { name: "ltx-video", family: "ltx-video" };
const wan = { name: "wan22-ti2v-5b:dmd", family: "wan" };

let storage: ReturnType<typeof memoryStorage>;

beforeEach(() => {
  storage = memoryStorage();
  setLastUsedStylesStorage(storage);
  setActivePinia(createPinia());
});

/*
 * Each section — Still picture, Short clip, 3-D object — keeps the style it
 * was last used with, across section switches AND across a restart. A person
 * who spent the afternoon on FLUX.2 for pictures and Wan for clips must find
 * both waiting the next morning: the section switch used to park one style
 * for exactly one round trip, and a fresh launch always opened on FLUX or the
 * first style installed.
 */
describe("last-used styles", () => {
  it("starts empty and picks the first candidate", () => {
    const store = useLastUsedStylesStore();
    expect(store.bySection).toEqual({ still: null, clip: null, mesh: null });
    expect(store.lastSection).toBeNull();
    expect(store.pick("still", [flux, flux2])).toBe(flux);
    expect(store.pick("still", [])).toBeNull();
  });

  it("remembers one style per section and which section was used last", () => {
    const store = useLastUsedStylesStore();
    store.remember("still", flux2.name);
    store.remember("clip", wan.name);
    store.remember("still", flux2.name);
    expect(store.bySection).toEqual({
      still: flux2.name,
      clip: wan.name,
      mesh: null,
    });
    expect(store.lastSection).toBe("still");
  });

  it("picks the remembered style when it is on offer, else the first", () => {
    const store = useLastUsedStylesStore();
    store.remember("clip", wan.name);
    expect(store.pick("clip", [ltx, wan])).toBe(wan);
    // Another machine, without Wan: the section's first style, never nothing.
    expect(store.pick("clip", [ltx])).toBe(ltx);
    // The remembered name survives that machine — it is not forgotten by an
    // inventory that lacks it.
    expect(store.bySection.clip).toBe(wan.name);
  });

  it("survives a restart", () => {
    useLastUsedStylesStore().remember("clip", wan.name);
    useLastUsedStylesStore().remember("mesh", "hunyuan3d-mini-turbo:fp16");

    setActivePinia(createPinia());
    const fresh = useLastUsedStylesStore();
    expect(fresh.bySection).toEqual({
      still: null,
      clip: wan.name,
      mesh: "hunyuan3d-mini-turbo:fp16",
    });
    expect(fresh.lastSection).toBe("mesh");
    expect(fresh.pick("clip", [ltx, wan])).toBe(wan);
  });

  it("ignores an empty name and treats a corrupt record as empty", () => {
    const store = useLastUsedStylesStore();
    store.remember("still", "");
    expect(store.bySection.still).toBeNull();

    storage.data.set(LAST_USED_STYLES_KEY, "{not json");
    setActivePinia(createPinia());
    expect(useLastUsedStylesStore().bySection).toEqual({
      still: null,
      clip: null,
      mesh: null,
    });

    storage.data.set(
      LAST_USED_STYLES_KEY,
      JSON.stringify({ version: 1, bySection: { still: 7 } }),
    );
    setActivePinia(createPinia());
    expect(useLastUsedStylesStore().bySection.still).toBeNull();
  });

  it("forgets a section on request", () => {
    const store = useLastUsedStylesStore();
    store.remember("still", flux.name);
    store.forget("still");
    expect(store.bySection.still).toBeNull();
    expect(
      JSON.parse(storage.data.get(LAST_USED_STYLES_KEY) ?? "{}").bySection
        .still,
    ).toBeNull();
  });

  it("keeps working with no storage at all", () => {
    setLastUsedStylesStorage({
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
      removeItem: () => {},
    });
    setActivePinia(createPinia());
    const store = useLastUsedStylesStore();
    store.remember("still", flux.name);
    expect(store.pick("still", [flux2, flux])).toBe(flux);
  });
});
