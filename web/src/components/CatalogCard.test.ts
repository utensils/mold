import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import type { CatalogEntryWire } from "../types";
import CatalogCard from "./CatalogCard.vue";

const baseEntry: CatalogEntryWire = {
  id: "hf:a",
  name: "Alpha",
  family: "flux",
  engine_phase: 1,
  source: "hf",
  source_id: "a",
  author: "alice",
  family_role: "finetune",
  sub_family: null,
  modality: "image",
  kind: "checkpoint",
  file_format: "safetensors",
  bundling: "separated",
  size_bytes: 6_000_000_000,
  download_count: 1234,
  rating: 4.7,
  likes: 0,
  nsfw: false,
  thumbnail_url: null,
  description: null,
  license: null,
  license_flags: null,
  tags: [],
  companions: [],
  download_recipe: { files: [], needs_token: null },
  created_at: null,
  updated_at: null,
  added_at: 0,
};

describe("CatalogCard", () => {
  it("renders name + size + author + downloads", () => {
    const w = mount(CatalogCard, { props: { entry: baseEntry } });
    expect(w.text()).toContain("Alpha");
    expect(w.text()).toContain("alice");
    expect(w.text()).toContain("6.0 GB");
    expect(w.text()).toContain("1,234");
  });

  it("shows phase badge for engine_phase >= 2", () => {
    const entry: CatalogEntryWire = { ...baseEntry, engine_phase: 3 };
    const w = mount(CatalogCard, { props: { entry } });
    expect(w.text()).toMatch(/phase 3|coming/i);
  });
});
