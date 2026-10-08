import { reactive } from "vue";
import { beforeEach, expect, it, vi } from "vitest";
import { restoreRetainedDraftMedia } from "./retainedDraftMedia";
const mocks = vi.hoisted(() => ({ relay: vi.fn(), status: vi.fn() }));
vi.mock("../api/client", () => ({ apiFetchTo: mocks.status }));
vi.mock("../api/gallerySourceMedia", () => ({
  relayRetainedSourceMedia: mocks.relay,
}));
const inventory = {
  availability: "available" as const,
  members: [
    {
      member_id: "frames",
      role: "keyframes",
      display_name: "last.png",
      size_bytes: 4,
    },
  ],
};
const layout = { web: false, boundary: false, sourceMode: "single", h3: true };
beforeEach(() => {
  mocks.status
    .mockReset()
    .mockResolvedValue({ json: async () => ({ instance_id: "same" }) });
  mocks.relay.mockReset();
});
it("never resurrects a slot attached and removed while reading", async () => {
  let finish!: (value: object) => void;
  mocks.relay.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const form = reactive({ model: "h3", keyframes: [] as object[] });
  const result = restoreRetainedDraftMedia({
    filename: "clip",
    origin: { baseUrl: "https://origin", apiKey: null },
    inventory,
    read: () => form,
    layout,
    isCurrent: () => true,
  });
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  form.keyframes = [{ frame: 1 }];
  form.keyframes = [];
  finish({ keyframes: [{ frame: 140, image: "data", name: "last" }] });
  expect(await result).toEqual({
    patch: {},
    inventory: { availability: "available", members: [] },
  });
});
it("rejects a restarted origin and leaves all wells unchanged", async () => {
  mocks.status
    .mockResolvedValueOnce({ json: async () => ({ instance_id: "old" }) })
    .mockResolvedValueOnce({ json: async () => ({ instance_id: "new" }) });
  mocks.relay.mockResolvedValue({ keyframes: [{ frame: 140, image: "data" }] });
  const form = reactive({ model: "h3", keyframes: [] });
  await expect(
    restoreRetainedDraftMedia({
      filename: "clip",
      origin: { baseUrl: "https://origin", apiKey: null },
      inventory,
      read: () => form,
      layout,
      isCurrent: () => true,
    }),
  ).rejects.toThrow("restarted");
  expect(form.keyframes).toEqual([]);
});
it("checks the budget before any reads and fences a newer reuse", async () => {
  const form = { model: "h3" };
  const options = {
    filename: "clip",
    origin: { baseUrl: "https://origin", apiKey: null },
    inventory,
    read: () => form,
    layout,
    isCurrent: () => false,
  };
  await expect(
    restoreRetainedDraftMedia({ ...options, maxBytes: 1 }),
  ).rejects.toThrow("limit");
  expect(mocks.relay).not.toHaveBeenCalled();
  mocks.relay.mockResolvedValue({ keyframes: [{ frame: 140, image: "data" }] });
  expect(await restoreRetainedDraftMedia(options)).toBeNull();
});

it("keeps unrelated retained inputs when one role is edited", async () => {
  let finish!: (value: object) => void;
  mocks.relay.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const form = reactive({
    model: "ltx",
    sourceImage: null as string | null,
    audioFile: null,
  });
  const media = {
    availability: "available" as const,
    members: [
      {
        member_id: "source",
        role: "source_image",
        display_name: "original.png",
        size_bytes: 1,
      },
      {
        member_id: "mask",
        role: "mask_image",
        display_name: "mask.png",
        size_bytes: 1,
      },
      {
        member_id: "audio",
        role: "audio_file",
        display_name: "sound.wav",
        size_bytes: 1,
      },
    ],
  };
  const result = restoreRetainedDraftMedia({
    filename: "clip",
    origin: { baseUrl: "https://origin", apiKey: null },
    inventory: media,
    read: () => form,
    layout: { ...layout, h3: false },
    isCurrent: () => true,
  });
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  form.sourceImage = "replacement";
  finish({
    source_image: "original",
    mask_image: "original-mask",
    audio_file: "sound",
  });
  const restored = await result;
  expect(restored?.patch.audioFile).toMatchObject({
    base64: "sound",
    filename: "sound.wav",
  });
  expect(restored?.patch).not.toHaveProperty("sourceImage");
  expect(restored?.patch).not.toHaveProperty("maskImage");
  expect(restored?.inventory.members).toEqual([]);
});

it.each([false, true])(
  "does not overwrite a Wan opening frame edited in flight on web=%s",
  async (web) => {
    let finish!: (value: object) => void;
    mocks.relay.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const form = reactive({
      model: "wan",
      sourceImage: null as string | null,
      imageAttachments: [] as object[],
      keyframes: [],
      endFrame: null,
    });
    const result = restoreRetainedDraftMedia({
      filename: "clip",
      origin: { baseUrl: "https://origin", apiKey: null },
      inventory,
      read: () => form,
      layout: { web, h3: false, boundary: true, sourceMode: "single" },
      isCurrent: () => true,
    });
    await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
    if (web) {
      form.imageAttachments = [{ base64: "new" }];
      form.imageAttachments = [];
    } else {
      form.sourceImage = "new";
      form.sourceImage = null;
    }
    finish({
      keyframes: [
        { frame: 0, image: "old-first" },
        { frame: 96, image: "old-last" },
      ],
    });
    expect((await result)?.patch).toEqual({});
  },
);
