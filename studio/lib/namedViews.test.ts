import { describe, expect, it } from "vitest";
import {
  emptyNamedViews,
  namedViewValidationError,
  serializeNamedViews,
  setNamedView,
} from "./namedViews";

const profile = {
  mode: "adjustable" as const,
  roles: ["front", "left", "back", "right"] as Array<
    "front" | "left" | "back" | "right"
  >,
  min_count: 1,
  max_count: 4,
};

const image = (name: string, data = name) => ({
  base64: data,
  filename: `${name}.png`,
  mimeType: "image/png",
  width: 32,
  height: 24,
});

describe("named Hunyuan3D views", () => {
  it("serializes present slots in canonical semantic order", () => {
    let state = emptyNamedViews();
    state = setNamedView(state, "right", image("right"));
    state = setNamedView(state, "front", image("front"));
    state = setNamedView(state, "back", image("back"));

    expect(
      serializeNamedViews(state, profile).map((item) =>
        item.kind === "named_image" ? item.role : null,
      ),
    ).toEqual(["front", "back", "right"]);
    expect(serializeNamedViews(state, profile)[0]).toEqual({
      kind: "named_image",
      role: "front",
      media: { authority: "inline", data: "front" },
      provenance: { name: "front.png" },
      mime_type: "image/png",
      width: 32,
      height: 24,
    });
  });

  it("replaces one slot without renumbering any other view", () => {
    const state = setNamedView(
      setNamedView(emptyNamedViews(), "left", image("old")),
      "left",
      image("new"),
    );
    expect(serializeNamedViews(state, profile)).toHaveLength(1);
    expect(serializeNamedViews(state, profile)[0]?.media).toEqual({
      authority: "inline",
      data: "new",
    });
  });

  it("requires the advertised minimum and complete image facts", () => {
    expect(namedViewValidationError(emptyNamedViews(), profile)).toBe(
      "Add at least one named view.",
    );
    const incomplete = setNamedView(emptyNamedViews(), "front", {
      ...image("front"),
      width: 0,
    });
    expect(namedViewValidationError(incomplete, profile)).toBe(
      "Front view could not be decoded as a PNG or JPEG image.",
    );
  });
});
