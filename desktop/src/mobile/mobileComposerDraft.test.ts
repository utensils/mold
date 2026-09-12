import { describe, expect, it, vi } from "vitest";
import type { DraftMediaRecord } from "@studio/lib/draftMediaStore";
import { newGenerateForm } from "../lib/generateForm";
import { createMobileComposerDraft, MOBILE_COMPOSER_DRAFT_KEY } from "./mobileComposerDraft";
import { MOBILE_DRAFT_MEDIA_PREFIX } from "./mobileDraftMedia";

function harness() {
  const metadata = new Map<string, string>();
  const media = new Map<string, DraftMediaRecord>();
  let revision = 0;
  const storage = {
    getItem: (key: string) => metadata.get(key) ?? null,
    setItem: vi.fn((key: string, value: string) => {
      metadata.set(key, value);
    }),
    removeItem: (key: string) => {
      metadata.delete(key);
    },
  };
  const write = vi.fn(async (records: DraftMediaRecord[]) => {
    for (const record of records) media.set(record.draftId!, record);
    return true;
  });
  const remove = vi.fn(async (prefix: string) => {
    for (const key of media.keys()) if (key.startsWith(prefix)) media.delete(key);
  });
  const dependencies = {
    storage,
    write,
    remove,
    read: async (key: string) => media.get(key) ?? null,
    revision: () => `test-${++revision}`,
  };
  return {
    metadata,
    media,
    storage,
    write,
    remove,
    dependencies,
    draft: createMobileComposerDraft(dependencies),
  };
}

describe("mobile composer persistence", () => {
  it("restores authored settings and media across a new controller, excluding resolved/private state", async () => {
    const h = harness();
    const form = newGenerateForm();
    form.prompt = "unfinished prompt";
    form.title = "Unfinished title";
    form.model = "currently-unavailable";
    form.width = 768;
    form.identityImage = { filename: "face.png", base64: "PRIVATE_FACE" };
    form.sourceImage = "PRIVATE_SOURCE";
    form.fileUnderAutoTag = true;
    Object.assign(form, { apiKey: "PRIVATE_KEY", liveQueues: [{ prompt: "REMOTE_PROMPT" }] });
    expect(await h.draft.save(form, "manual")).toBe(true);
    const raw = h.metadata.get(MOBILE_COMPOSER_DRAFT_KEY)!;
    expect(raw).not.toMatch(
      /PRIVATE_FACE|PRIVATE_SOURCE|PRIVATE_KEY|REMOTE_PROMPT|fileUnderAutoTag|recipeCapabilities/,
    );
    const restored = await createMobileComposerDraft(h.dependencies).restore();
    expect(restored.error).toBe("");
    expect(restored.missing).toEqual([]);
    expect(restored.form).toMatchObject({
      prompt: form.prompt,
      title: form.title,
      model: form.model,
      width: 768,
      sourceImage: "PRIVATE_SOURCE",
      identityImage: form.identityImage,
      fileUnderAutoTag: false,
    });
    expect(restored.canvasIntent).toBe("manual");
  });

  it.each([
    "{",
    '{"version":999}',
    JSON.stringify({
      version: 1,
      revision: "safe",
      canvasIntent: "manual",
      form: { imageAttachments: 7 },
    }),
    JSON.stringify({
      version: 1,
      revision: "safe",
      canvasIntent: "manual",
      form: { h3Authoring: { references: null } },
    }),
  ])("reports unreadable drafts without overwriting them: %s", async (raw) => {
    const h = harness();
    h.metadata.set(MOBILE_COMPOSER_DRAFT_KEY, raw);
    const restored = await h.draft.restore();
    expect(restored.error).toContain("unreadable");
    expect(restored.form).toBeUndefined();
    expect(h.metadata.get(MOBILE_COMPOSER_DRAFT_KEY)).toBe(raw);
  });

  it("never restores a saved prompt preset — the chips that showed it are gone", async () => {
    // A draft written before the preset retired still carries the key. It
    // would restyle the prompt at submit with no control left to show or
    // clear it, so restore tolerates the extra key and drops it.
    const h = harness();
    h.metadata.set(
      MOBILE_COMPOSER_DRAFT_KEY,
      JSON.stringify({
        version: 1,
        revision: "legacy-1",
        canvasIntent: "manual",
        form: { prompt: "a cat", stylePreset: "cinematic" },
      }),
    );
    const restored = await createMobileComposerDraft(h.dependencies).restore();
    expect(restored.form?.prompt).toBe("a cat");
    expect("stylePreset" in restored.form!).toBe(false);
  });

  it("commits media before metadata, and preserves the last good draft if media fails", async () => {
    const h = harness();
    const form = newGenerateForm();
    form.sourceImage = "OLD";
    await h.draft.save(form, "source");
    const previous = h.metadata.get(MOBILE_COMPOSER_DRAFT_KEY);
    h.write.mockImplementationOnce(async () => {
      expect(h.metadata.get(MOBILE_COMPOSER_DRAFT_KEY)).toBe(previous);
      return false;
    });
    form.sourceImage = "NEW";
    expect(await h.draft.save(form, "manual")).toBe(false);
    expect(h.metadata.get(MOBILE_COMPOSER_DRAFT_KEY)).toBe(previous);
    expect((await h.draft.restore()).form?.sourceImage).toBe("OLD");
  });

  it("prevents an older in-flight write from publishing after a newer edit", async () => {
    const h = harness();
    let release!: (value: boolean) => void;
    h.write.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    const form = newGenerateForm();
    form.prompt = "old";
    const old = h.draft.save(form, "manual");
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    form.prompt = "new";
    const latest = h.draft.save(form, "manual");
    release(true);
    expect(await old).toBe(false);
    expect(await latest).toBe(true);
    expect((await h.draft.restore()).form?.prompt).toBe("new");
    expect(h.storage.setItem).toHaveBeenCalledTimes(1);
  });

  it("clears only mobile composer media and fences an active save", async () => {
    const h = harness();
    h.media.set("template/keep", { base64: "KEEP" });
    const form = newGenerateForm();
    form.sourceImage = "BYTES";
    await h.draft.save(form, "manual");
    let release!: (value: boolean) => void;
    h.write.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    form.sourceImage = "CHANGED_BYTES";
    const pending = h.draft.save(form, "manual");
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    const clear = h.draft.clear();
    release(true);
    await pending;
    await clear;
    expect(h.metadata.has(MOBILE_COMPOSER_DRAFT_KEY)).toBe(false);
    expect([...h.media.keys()]).toEqual(["template/keep"]);
    expect(h.remove).toHaveBeenCalledWith(MOBILE_DRAFT_MEDIA_PREFIX);
  });

  it("reuses immutable media when only authored text changes", async () => {
    const h = harness();
    const form = newGenerateForm();
    form.sourceVideo = { filename: "large.mp4", base64: "LARGE_VIDEO" };
    await h.draft.save(form, "manual");
    const mediaKeys = [...h.media.keys()];
    form.prompt = "edited";
    await h.draft.save(form, "manual");
    expect(h.write).toHaveBeenCalledTimes(1);
    expect([...h.media.keys()]).toEqual(mediaKeys);
    expect((await h.draft.restore()).form?.prompt).toBe("edited");
  });

  it("does not rewrite media for the first edit after relaunch", async () => {
    const h = harness();
    const form = newGenerateForm();
    form.sourceVideo = { filename: "large.mp4", base64: "LARGE_VIDEO" };
    await h.draft.save(form, "manual");
    const relaunched = createMobileComposerDraft(h.dependencies);
    const restored = await relaunched.restore();
    restored.form!.prompt = "After relaunch";
    expect(await relaunched.save(restored.form!, "manual")).toBe(true);
    expect(h.write).toHaveBeenCalledTimes(1);
  });

  it.each(["clear", "save"] as const)("fences delayed restoration after %s", async (action) => {
    const h = harness();
    const form = newGenerateForm();
    form.sourceImage = "OLD_SOURCE";
    await h.draft.save(form, "manual");
    let release!: (value: DraftMediaRecord | null) => void;
    const relaunched = createMobileComposerDraft({
      ...h.dependencies,
      read: () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    });
    const restore = relaunched.restore();
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    if (action === "clear") await relaunched.clear();
    else await relaunched.save(newGenerateForm(), "manual");
    release({ base64: "OLD_SOURCE" });
    expect((await restore).form).toBeUndefined();
  });

  it.each([
    { retakeRange: "bad" },
    { spatialUpscale: {} },
    { temporalUpscale: 7 },
    ...[{ picked: {} }, { picked: { name: "Collection", id: 7 } }, { clearedMatchSlug: {} }].map(
      (invalid) => ({ fileUnder: { ...newGenerateForm().fileUnder, ...invalid } }),
    ),
    { sourceFit: { mode: "unknown" } },
    { sourceFit: { mode: "crop-fill", alignX: "bad" } },
    { sourceFit: { mode: "upscale-then-fit", fit: {} } },
    { sourceFit: { mode: "upscale-then-fit", upscalerModel: "x", fit: { mode: "unknown" } } },
  ])("refuses malformed nullable controls: %j", async (invalid) => {
    const h = harness();
    await h.draft.save(newGenerateForm(), "manual");
    const envelope = JSON.parse(h.metadata.get(MOBILE_COMPOSER_DRAFT_KEY)!);
    Object.assign(envelope.form, invalid);
    h.metadata.set(MOBILE_COMPOSER_DRAFT_KEY, JSON.stringify(envelope));
    expect((await h.draft.restore()).error).toContain("unreadable");
  });

  it("reports evicted source media rather than restoring a text-only draft", async () => {
    const h = harness();
    const form = newGenerateForm();
    form.sourceImage = "SOURCE";
    await h.draft.save(form, "source");
    h.media.clear();
    const restored = await h.draft.restore();
    expect(restored.form?.sourceImage).toBe("");
    expect(restored.missing).toEqual(["sourceImage"]);
  });
});
