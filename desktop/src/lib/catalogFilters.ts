/**
 * Discover filter vocabularies shared by the desktop and iPhone catalog views.
 * Values are the exact `GET /api/catalog/search` wire words (unknown values
 * 422); labels and order mirror the web `CatalogTopbar.vue` verbatim.
 */
export type CatalogKindFilter =
  "checkpoint" | "lora" | "clip" | "text-encoder" | "vae" | "tokenizer" | "control-net";

export type CatalogSortOption = "downloads" | "rating" | "recent";

export const CATALOG_KIND_OPTIONS: { value: CatalogKindFilter; label: string }[] = [
  { value: "checkpoint", label: "Models" },
  { value: "lora", label: "LoRAs" },
  { value: "clip", label: "CLIP" },
  { value: "text-encoder", label: "Text encoders" },
  { value: "vae", label: "VAEs" },
  { value: "tokenizer", label: "Tokenizers" },
  { value: "control-net", label: "ControlNet" },
];

export const CATALOG_SORT_OPTIONS: { value: CatalogSortOption; label: string }[] = [
  { value: "downloads", label: "Downloads" },
  { value: "rating", label: "Rating" },
  { value: "recent", label: "Recent" },
];
