import { describe, it, expect, vi } from "vitest";
import { newGenerateForm } from "./generateForm";
import { attachRetainedSource } from "./retainedSourceAttachment";

describe("retained source attachment", () => {
  it("preserves edits made while original media downloads", async () => {
    const form = newGenerateForm();
    form.family = "flux";
    let finish!: (value: Blob) => void;
    const result = attachRetainedSource(
      form,
      "source.png",
      "image",
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
      () => true,
    );
    form.prompt = "newer draft";
    finish(new Blob(["abc"], { type: "image/png" }));
    expect((await result).ok).toBe(false);
    expect(form.sourceImage).toBeNull();
    expect(form.prompt).toBe("newer draft");
  });
  it("rejects an origin change without touching the draft", async () => {
    const form = newGenerateForm();
    form.family = "flux";
    const read = vi.fn(async () => new Blob(["abc"]));
    const result = await attachRetainedSource(form, "source.png", "image", read, () => false);
    expect(result.ok).toBe(false);
    expect(read).not.toHaveBeenCalled();
    expect(form.sourceImage).toBeNull();
  });
});

it("rejects edit then undo during a pending read", async () => {
  const { reactive } = await import("vue");
  const form = reactive(newGenerateForm());
  form.family = "flux";
  let finish!: (value: Blob) => void;
  const result = attachRetainedSource(
    form,
    "source.png",
    "image",
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
    () => true,
  );
  form.prompt = "temporary";
  form.prompt = "";
  finish(new Blob(["abc"]));
  expect((await result).ok).toBe(false);
  expect(form.sourceImage).toBeNull();
});
it("normalizes a source and clears its obsolete mask", async () => {
  const form = newGenerateForm();
  form.family = "flux";
  form.maskImage = "old-mask";
  const result = await attachRetainedSource(
    form,
    "source.png",
    "image",
    async () => new Blob(["abc"], { type: "image/png" }),
    () => true,
  );
  expect(result.ok).toBe(true);
  expect(form.sourceImage).toBe("YWJj");
  expect(form.maskImage).toBeNull();
  expect(form.exclusiveWell).toBe("source");
});
it("appends references while retaining the target", async () => {
  const form = newGenerateForm();
  form.family = "qwen-image-edit";
  form.model = "qwen-image-edit";
  form.imageAttachments = ["existing-target"];
  const result = await attachRetainedSource(
    form,
    "reference.png",
    "image",
    async () => new Blob(["abc"], { type: "image/png" }),
    () => true,
  );
  expect(result.ok).toBe(true);
  expect(form.imageAttachments).toEqual(["existing-target", "YWJj"]);
  expect(form.exclusiveWell).toBe("references");
});

it("routes H3 audio through typed mixed references instead of legacy audio", async () => {
  const helper = await import("@studio/lib/minimaxH3PickedMedia");
  const { emptyMinimaxH3AuthoringState } = await import("@studio/lib/minimaxH3Authoring");
  const state = emptyMinimaxH3AuthoringState();
  const append = vi.spyOn(helper, "appendMinimaxH3PickedMediaReference").mockResolvedValue(state);
  const form = newGenerateForm();
  form.family = "minimax-h3-ref2va";
  form.model = "minimax-h3-ref2va:comfy-pruned-int8";
  const outcome = await attachRetainedSource(
    form,
    "audio_file-4",
    "audio",
    async () => new Blob(["fixture"], { type: "application/octet-stream" }),
    () => true,
  );
  expect(outcome.ok).toBe(true);
  expect(append).toHaveBeenCalledOnce();
  expect(append.mock.calls[0]![1].type).toBe("audio/wav");
  expect(form.audioFile).toBeNull();
  append.mockRestore();
});

it("routes extensionless octet-stream H3 images through the normal image path", async () => {
  const form = newGenerateForm();
  form.family = "minimax-h3-ref2va";
  form.model = "minimax-h3-ref2va:comfy-pruned-int8";
  const outcome = await attachRetainedSource(
    form,
    "source_image-1",
    "image",
    async () => new Blob(["abc"], { type: "application/octet-stream" }),
    () => true,
  );
  expect(outcome.ok).toBe(true);
  expect(form.h3Authoring?.references[0]?.reference.media).toEqual({
    authority: "inline",
    data: "YWJj",
  });
  expect(form.h3Authoring?.references[0]?.reference.kind).toBe("image");
});

it("inserts LTX endpoints without replacing middle keyframes or resizing the canvas", async () => {
  const form = newGenerateForm();
  form.family = "ltx-2";
  form.model = "ltx-2";
  form.frames = 25;
  form.keyframes = [{ frame: 12, image: { filename: "middle.png", base64: "bWlk" } }];
  form.sourceImage = "old";
  form.sourceImageName = "existing-first.png";
  form.maskImage = "mask";
  const originalSize = [form.width, form.height];
  const outcome = await attachRetainedSource(
    form,
    "last.png",
    "image",
    async () => new Blob(["abc"]),
    () => true,
    "last",
  );
  expect(outcome.ok).toBe(true);
  expect(form.keyframes.map((frame) => frame.frame)).toEqual([0, 12, 24]);
  expect(form.keyframes[0]?.image).toEqual({ filename: "existing-first.png", base64: "old" });
  expect(form.keyframes[1]?.image.filename).toBe("middle.png");
  expect(form.keyframes[2]?.image.base64).toBe("YWJj");
  expect(form.sourceImage).toBeNull();
  expect(form.maskImage).toBeNull();
  expect([form.width, form.height]).toEqual(originalSize);
});
it("replaces an existing LTX endpoint in place and refuses an unsupported endpoint before reading", async () => {
  const form = newGenerateForm();
  form.family = "ltx-2";
  form.model = "ltx-2";
  form.frames = 25;
  form.keyframes = [
    { frame: 0, image: { filename: "old.png", base64: "old" } },
    { frame: 12, image: { filename: "middle.png", base64: "middle" } },
  ];
  expect(
    (
      await attachRetainedSource(
        form,
        "first.png",
        "image",
        async () => new Blob(["abc"]),
        () => true,
        "first",
      )
    ).ok,
  ).toBe(true);
  expect(form.keyframes).toHaveLength(2);
  expect(form.keyframes[0]?.image.filename).toBe("first.png");
  form.family = "flux";
  const read = vi.fn(async () => new Blob(["abc"]));
  expect((await attachRetainedSource(form, "no.png", "image", read, () => true, "last")).ok).toBe(
    false,
  );
  expect(read).not.toHaveBeenCalled();
});
