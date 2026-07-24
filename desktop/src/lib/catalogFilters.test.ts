import { describe, expect, it } from "vitest";
import { CATALOG_KIND_OPTIONS, CATALOG_SORT_OPTIONS } from "./catalogFilters";

// The server 422s on unknown values — every option must stay inside the wire
// vocabulary of GET /api/catalog/search (crates/mold-catalog).
const SERVER_KINDS = [
  "checkpoint",
  "lora",
  "vae",
  "text-encoder",
  "tokenizer",
  "clip",
  "control-net",
];
const SERVER_SORTS = ["downloads", "recent", "rating"];

describe("catalog filter options", () => {
  it("keeps every kind option inside the server wire vocabulary", () => {
    for (const option of CATALOG_KIND_OPTIONS) {
      expect(SERVER_KINDS).toContain(option.value);
    }
  });

  it("keeps every sort option inside the server wire vocabulary", () => {
    for (const option of CATALOG_SORT_OPTIONS) {
      expect(SERVER_SORTS).toContain(option.value);
    }
  });

  it("pins the web CatalogTopbar labels and order verbatim", () => {
    expect(CATALOG_KIND_OPTIONS).toEqual([
      { value: "checkpoint", label: "Models" },
      { value: "lora", label: "LoRAs" },
      { value: "clip", label: "CLIP" },
      { value: "text-encoder", label: "Text encoders" },
      { value: "vae", label: "VAEs" },
      { value: "tokenizer", label: "Tokenizers" },
      { value: "control-net", label: "ControlNet" },
    ]);
    expect(CATALOG_SORT_OPTIONS).toEqual([
      { value: "downloads", label: "Downloads" },
      { value: "rating", label: "Rating" },
      { value: "recent", label: "Recent" },
    ]);
  });
});
