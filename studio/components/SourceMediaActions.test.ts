import { mount, flushPromises } from "@vue/test-utils";
import { computed } from "vue";
import { beforeEach, expect, it, vi } from "vitest";
import SourceMediaActions from "./SourceMediaActions.vue";
import { NATIVE_SOURCE_FILE_ACTIONS } from "../lib/sourceMediaActions";
import { SOURCE_ATTACHMENT_CHOICES } from "../lib/sourceAttachmentChoices";
import {
  invalidateSourceOwner,
  clearSourceResources,
} from "../lib/sourceMediaResources";
import { sourceOwnerIdentity } from "../lib/sourceMediaBrowser";
const read = vi.hoisted(() => vi.fn());
vi.mock("../api/sourceMediaBrowser", () => ({
  retainedSourceMediaContent: read,
}));
const selection = {
  owner: {
    key: "a",
    filename: "print.png",
    title: "A",
    hostLabel: "Workstation",
    target: { baseUrl: "https://host", apiKey: null },
  },
  member: {
    member_id: "abcdef0123456789",
    role: "source_image",
    display_name: "source.png",
    size_bytes: 3,
    media_kind: "image",
  },
};
beforeEach(() => {
  clearSourceResources();
  read.mockReset();
  URL.createObjectURL = vi.fn(() => "blob:saved");
  URL.revokeObjectURL = vi.fn();
});
it("offers exact source IDs, endpoint actions and the same actions on right-click", async () => {
  const view = mount(SourceMediaActions, {
    props: { selection },
    global: {
      provide: {
        [SOURCE_ATTACHMENT_CHOICES as symbol]: computed(() => [
          "first",
          "last",
        ]),
      },
    },
  });
  expect(view.text()).toContain("source:print.png#abcdef0123456789");
  expect(view.text()).toContain("Copy CLI Command");
  await view
    .findAll("button")
    .find((button) => button.text() === "Use as last frame")!
    .trigger("click");
  expect(view.emitted("use")?.[0]).toEqual([
    { ...selection, attachment: "last" },
  ]);
  view.vm.open(new MouseEvent("contextmenu", { clientX: 200, clientY: 200 }));
  await flushPromises();
  expect(view.get('[role="menu"]').text()).toContain("Save a Copy");
  view.unmount();
});
it("never publishes a downloaded original after owner revocation", async () => {
  let finish!: (blob: Blob) => void;
  read.mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  const view = mount(SourceMediaActions, { props: { selection } });
  await view
    .findAll("button")
    .find((button) => button.text() === "Save a Copy")!
    .trigger("click");
  invalidateSourceOwner(sourceOwnerIdentity(selection.owner));
  finish(new Blob(["abc"]));
  await flushPromises();
  expect(URL.createObjectURL).not.toHaveBeenCalled();
  view.unmount();
});

it("uses the desktop original file adapters for image clipboard and Save a Copy", async () => {
  const blob = new Blob(["png-original"], { type: "image/png" });
  read.mockResolvedValue(blob);
  const copyImage = vi.fn(async (_blob, current) => {
    expect(current()).toBe(true);
  });
  const saveFile = vi.fn(async (_blob, _filename, current) => {
    expect(current()).toBe(true);
  });
  const view = mount(SourceMediaActions, {
    props: { selection },
    global: {
      provide: {
        [NATIVE_SOURCE_FILE_ACTIONS as symbol]: { copyImage, saveFile },
      },
    },
  });
  await view
    .findAll("button")
    .find((button) => button.text() === "Copy File")!
    .trigger("click");
  await flushPromises();
  expect(copyImage).toHaveBeenCalledWith(blob, expect.any(Function));
  await view
    .findAll("button")
    .find((button) => button.text() === "Save a Copy")!
    .trigger("click");
  await flushPromises();
  expect(saveFile).toHaveBeenCalledWith(
    blob,
    "source.png",
    expect.any(Function),
  );
  expect(URL.createObjectURL).not.toHaveBeenCalled();
  view.unmount();
});
