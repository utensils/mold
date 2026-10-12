import { describe, expect, it } from "vitest";
import { fuzzyModelScore, rankModels } from "./fuzzyModelSearch";

const fields = ["FLUX.1 Dev Q4", "flux-dev:q4", "black-forest-labs/FLUX.1-dev"];
describe("fuzzy model search", () => {
  it.each([
    "FLUX",
    "dev flux",
    "flux_dev:q4",
    "fluxdevq4",
    "flx dev",
    "flxu dev",
    "fluz dev",
    "fluxx dev",
    "flx q4",
    "forest q4",
  ])("matches %s", (q) => {
    expect(fuzzyModelScore(q, fields)).toBeGreaterThan(0);
  });
  it("folds accents and keeps tokens across separate fields", () => {
    expect(fuzzyModelScore("cafe video", ["Café", "Video"])).toBeGreaterThan(0);
  });
  it.each(["q8", "banana", "flux banana", "dx", "4q"])(
    "refuses unrelated or wrong variant %s",
    (q) => {
      expect(fuzzyModelScore(q, fields)).toBe(0);
    },
  );
  it("ranks exact names first and keeps ties and empty lists stable", () => {
    const rows = [{ name: "fluxx" }, { name: "flux" }, { name: "flux" }];
    expect(rankModels(rows, "flux")).toEqual([rows[1], rows[2], rows[0]]);
    expect(rankModels(rows, "  ")).toEqual(rows);
  });
  it("uses presentation and metadata without replacing wire identities", () => {
    const rows = [
      {
        name: "cv:42",
        display_name: "Café Dream",
        family: "flux",
        description: "painterly",
      },
    ];
    expect(rankModels(rows, "cafe flux")).toEqual(rows);
    expect(rankModels(rows, "paint")).toEqual(rows);
  });
  it("bounds long queries and accepts non-Latin direct matches", () => {
    expect(fuzzyModelScore("猫", ["猫のモデル"])).toBeGreaterThan(0);
    expect(fuzzyModelScore("x".repeat(10000), fields)).toBe(0);
  });
});
