import { describe, expect, it } from "vitest";
import { meshDetailLadder } from "./meshDetailLadder";

describe("meshDetailLadder", () => {
  it("shows the floor, the advertised default and the ceiling of a long ladder", () => {
    expect(meshDetailLadder([128, 192, 256, 320, 384], 256)).toEqual([
      { value: 128, label: "Rough" },
      { value: 256, label: "Normal" },
      { value: 384, label: "Fine" },
    ]);
  });

  it("maps a three-rung ladder one-to-one, default or not", () => {
    expect(meshDetailLadder([192, 256, 384], 192)).toEqual([
      { value: 192, label: "Rough" },
      { value: 256, label: "Normal" },
      { value: 384, label: "Fine" },
    ]);
  });

  it("collapses to two words when the default is an end of the ladder", () => {
    expect(meshDetailLadder([128, 192, 256, 384], 128)).toEqual([
      { value: 128, label: "Rough" },
      { value: 384, label: "Fine" },
    ]);
  });

  it("ignores a default the recipe does not actually offer", () => {
    expect(meshDetailLadder([128, 256, 320, 384], 512)).toEqual([
      { value: 128, label: "Rough" },
      { value: 384, label: "Fine" },
    ]);
  });

  it("renders nothing for a recipe with no ladder", () => {
    expect(meshDetailLadder([], 256)).toEqual([]);
    expect(meshDetailLadder(null, null)).toEqual([]);
  });
});
