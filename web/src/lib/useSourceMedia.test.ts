import { describe, it, expect, vi } from "vitest";
import * as pickedMedia from "@studio/lib/minimaxH3PickedMedia";
import { ref } from "vue";
import { attachSourceMedia } from "./useSourceMedia";
import { __testing__ } from "../composables/useGenerateForm";
import type { SourceMediaSelection } from "@studio/lib/sourceMediaBrowser";
const selection: SourceMediaSelection = {
  owner: {
    key: "one",
    filename: "print.png",
    title: "Print",
    hostLabel: "Machine",
    target: { baseUrl: "http://machine", apiKey: null },
  },
  member: {
    member_id: "id",
    role: "audio_file",
    media_kind: "audio",
    display_name: "input.wav",
    size_bytes: 4,
  },
};
describe("independent source attachment", () => {
  it("preserves draft edits made while original bytes load", async () => {
    const state = ref({
      ...__testing__.defaultForm(),
      model: "ltx-2",
      modelFamily: "ltx-2",
    });
    let finish!: (b: Blob) => void;
    const pending = attachSourceMedia(
      selection,
      state,
      undefined,
      () => true,
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    state.value.prompt = "newer prompt";
    finish(new Blob(["test"]));
    await expect(pending).rejects.toThrow("changed");
    expect(state.value.audioFile).toBeNull();
    expect(state.value.prompt).toBe("newer prompt");
  });
  it("routes retained audio through the mixed-media reference importer", async () => {
    const state = ref({
      ...__testing__.defaultForm(),
      model: "minimax-h3-ref2va:official-bf16",
      modelFamily: "minimax-h3",
    });
    const draft = { firstFrame: null, lastFrame: null, references: [] };
    const append = vi
      .spyOn(pickedMedia, "appendMinimaxH3PickedMediaReference")
      .mockResolvedValue(draft);
    try {
      await attachSourceMedia(
        selection,
        state,
        undefined,
        () => true,
        async () => new Blob(["audio"], { type: "audio/wav" }),
      );
      expect(append).toHaveBeenCalledOnce();
      expect(state.value.h3Authoring).toEqual(draft);
      expect(state.value.audioFile).toBeNull();
    } finally {
      append.mockRestore();
    }
  });
  it("refuses completion if the source or destination route changes", async () => {
    const state = ref({
      ...__testing__.defaultForm(),
      model: "ltx-2",
      modelFamily: "ltx-2",
    });
    await expect(
      attachSourceMedia(
        selection,
        state,
        undefined,
        () => false,
        async () => new Blob(["audio"]),
      ),
    ).rejects.toThrow("changed");
    expect(state.value.audioFile).toBeNull();
  });
  it("refuses an unsupported type without reading bytes", async () => {
    const state = ref(__testing__.defaultForm());
    await expect(
      attachSourceMedia(
        { ...selection, member: { ...selection.member, media_kind: "mesh" } },
        state,
        undefined,
        () => true,
      ),
    ).rejects.toThrow("cannot");
  });
});

it("uses retained image bytes as the LTX last keyframe while preserving intermediate frames", async () => {
  const state = ref({
    ...__testing__.defaultForm(),
    model: "ltx-2",
    modelFamily: "ltx-2",
    frames: 25,
    keyframes: [
      {
        frame: 12,
        image: {
          kind: "upload" as const,
          filename: "middle.png",
          base64: "middle",
        },
      },
    ],
  });
  const size = [state.value.width, state.value.height];
  await attachSourceMedia(
    {
      ...selection,
      attachment: "last",
      member: {
        ...selection.member,
        media_kind: "image",
        role: "source_image",
        display_name: "last.png",
      },
    },
    state,
    undefined,
    () => true,
    async () => new Blob(["abc"]),
  );
  expect(state.value.keyframes.map((frame) => frame.frame)).toEqual([12, 24]);
  expect(state.value.keyframes[0]?.image.filename).toBe("middle.png");
  expect(state.value.keyframes[1]?.image.base64).toBe("YWJj");
  expect([state.value.width, state.value.height]).toEqual(size);
});
