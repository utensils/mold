import { mount, flushPromises } from "@vue/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import SourceMediaCard from "./SourceMediaCard.vue";
import {
  clearSourceResources,
  invalidateSourceOwner,
} from "../lib/sourceMediaResources";
import {
  sourceOwnerIdentity,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
const thumbnail = vi.hoisted(() => vi.fn());
vi.mock("../api/sourceMediaBrowser", () => ({
  retainedSourceMediaThumbnail: thumbnail,
}));
const selection: SourceMediaSelection = {
  owner: {
    key: "a",
    filename: "a.png",
    title: "Print A",
    hostLabel: "Machine",
    target: { baseUrl: "http://a", apiKey: "secret" },
  },
  member: {
    member_id: "one",
    role: "source_image",
    display_name: "image",
    size_bytes: 1,
  },
};
beforeEach(() => {
  clearSourceResources();
  thumbnail.mockReset().mockResolvedValue(new Blob(["preview"]));
  vi.stubGlobal("IntersectionObserver", undefined);
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:preview");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
it("reuses warm previews on remount and immediately revokes hidden owner URLs", async () => {
  let view = mount(SourceMediaCard, { props: { selection } });
  await flushPromises();
  expect(view.find("img").exists()).toBe(true);
  view.unmount();
  view = mount(SourceMediaCard, { props: { selection } });
  await flushPromises();
  expect(thumbnail).toHaveBeenCalledTimes(1);
  invalidateSourceOwner(sourceOwnerIdentity(selection.owner));
  await flushPromises();
  expect(view.find("img").exists()).toBe(false);
  expect(URL.revokeObjectURL).toHaveBeenCalled();
  view.unmount();
});
it("keeps usage choices bounded and emits the exact selected route", async () => {
  const usages = Array.from({ length: 100 }, (_, n) => ({
    ...selection,
    owner: { ...selection.owner, key: String(n), title: `Print ${n}` },
    member: {
      ...selection.member,
      member_id: String(n),
      role: n === 1 ? "keyframes" : "source_image",
    },
  }));
  const view = mount(SourceMediaCard, {
    props: { selection: usages[0]!, usages },
  });
  await flushPromises();
  expect(view.findAll("li")).toHaveLength(0);
  (view.get("details").element as HTMLDetailsElement).open = true;
  await view.get("details").trigger("toggle");
  expect(view.findAll("li")).toHaveLength(20);
  await view.findAll("li button")[1]!.trigger("click");
  await view
    .findAll("button")
    .find((b) => b.text() === "Use as source")!
    .trigger("click");
  expect(view.emitted("use")?.[0]?.[0]).toEqual(usages[1]);
  await view.setProps({ usages: [usages[2]!], selection: usages[2]! });
  await flushPromises();
  expect(view.text()).not.toContain("Print 1");
  view.unmount();
});
it("refreshes the same member ID when verified bytes change", async () => {
  const first = {
    ...selection,
    member: { ...selection.member, content_sha256: "a".repeat(64) },
  };
  const view = mount(SourceMediaCard, { props: { selection: first } });
  await flushPromises();
  await view.setProps({
    selection: {
      ...first,
      member: { ...first.member, content_sha256: "b".repeat(64) },
    },
  });
  await flushPromises();
  expect(thumbnail).toHaveBeenCalledTimes(2);
  view.unmount();
});
