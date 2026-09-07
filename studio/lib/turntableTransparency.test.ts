import { describe, expect, it } from "vitest";
import {
  TURNTABLE_TRANSPARENCY_DEFAULT,
  TURNTABLE_TRANSPARENCY_STORAGE_KEY,
  loadTurntableTransparency,
  saveTurntableTransparency,
  type TurntableTransparencyStorage,
} from "./turntableTransparency";

function memory(
  seed: Record<string, string> = {},
): TurntableTransparencyStorage & {
  values: Record<string, string>;
} {
  const values = { ...seed };
  return {
    values,
    getItem: (key) => values[key] ?? null,
    setItem: (key, value) => {
      values[key] = value;
    },
  };
}

describe("turntable transparency preference", () => {
  it("defaults to an opaque backdrop", () => {
    expect(TURNTABLE_TRANSPARENCY_DEFAULT).toBe(false);
    expect(loadTurntableTransparency(memory())).toBe(false);
  });

  it("round-trips a choice", () => {
    const storage = memory();
    saveTurntableTransparency(true, storage);
    expect(storage.values[TURNTABLE_TRANSPARENCY_STORAGE_KEY]).toBe("true");
    expect(loadTurntableTransparency(storage)).toBe(true);
    saveTurntableTransparency(false, storage);
    expect(loadTurntableTransparency(storage)).toBe(false);
  });

  it("falls back to the default rather than throwing", () => {
    expect(
      loadTurntableTransparency(
        memory({ [TURNTABLE_TRANSPARENCY_STORAGE_KEY]: "yes" }),
      ),
    ).toBe(false);
    expect(loadTurntableTransparency(null)).toBe(false);
    const hostile: TurntableTransparencyStorage = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    };
    expect(loadTurntableTransparency(hostile)).toBe(false);
    expect(() => saveTurntableTransparency(true, hostile)).not.toThrow();
  });
});
