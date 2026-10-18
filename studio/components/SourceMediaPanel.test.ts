import { mount, flushPromises } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import SourceMediaPanel from "./SourceMediaPanel.vue";
import { clearSourceResources } from "../lib/sourceMediaResources";
const inventory = vi.hoisted(() => vi.fn());
vi.mock("../api/gallerySourceMedia", () => ({
  retainedSourceMediaInventory: inventory,
  retainedSourceMediaDisclosure: () => "unavailable",
  retainedSourceMediaDisclosable: (metadata: {
    source_image_sha256?: string;
  }) => !!metadata.source_image_sha256,
}));
beforeEach(() => clearSourceResources());
it("deduplicates identical bytes inside details without losing role choices", async () => {
  inventory.mockResolvedValue({
    availability: "available",
    members: ["source_image", "keyframes"].map((role, n) => ({
      member_id: String(n),
      role,
      display_name: role,
      size_bytes: 1,
      content_sha256: "a".repeat(64),
    })),
  });
  const view = mount(SourceMediaPanel, {
    props: {
      owner: {
        key: "a",
        filename: "a.png",
        title: "A",
        hostLabel: "Host",
        target: { baseUrl: "http://a", apiKey: null },
      },
    },
    global: {
      stubs: {
        SourceMediaCard: {
          props: ["usages"],
          template: '<div data-test="group">{{usages.length}}</div>',
        },
      },
    },
  });
  await flushPromises();
  expect(view.findAll('[data-test="group"]')).toHaveLength(1);
  expect(view.get('[data-test="group"]').text()).toBe("2");
  view.unmount();
});

it("shows established absence without retry, preserving read failures as errors", async () => {
  const props = {
    owner: {
      key: "empty",
      filename: "empty.png",
      title: "Empty",
      hostLabel: "Host",
      target: { baseUrl: "http://empty", apiKey: null },
    },
  };
  inventory.mockResolvedValue({ availability: "available", members: [] });
  const view = mount(SourceMediaPanel, { props });
  await flushPromises();
  expect(view.text()).toContain("No source media.");
  expect(view.findAll("button").some((b) => b.text() === "Retry")).toBe(false);
  view.unmount();
  clearSourceResources();
  inventory.mockRejectedValue(new Error("Unauthorized"));
  const failed = mount(SourceMediaPanel, { props });
  await flushPromises();
  expect(failed.text()).toContain("Unauthorized");
  expect(failed.text()).toContain("Retry");
  expect(failed.text()).not.toContain("No source media.");
  failed.unmount();
});

it("shows no-source text for an inventoried legacy text-only print", async () => {
  inventory.mockResolvedValue({
    availability: "unavailable_legacy",
    members: [],
  });
  const view = mount(SourceMediaPanel, {
    props: {
      owner: {
        key: "legacy",
        filename: "text.png",
        title: "Text",
        hostLabel: "Host",
        target: { baseUrl: "http://legacy", apiKey: null },
        metadata: {},
      },
    },
  });
  await flushPromises();
  expect(inventory).toHaveBeenCalled();
  expect(view.text()).toContain("No source media.");
  expect(view.text()).not.toContain("Retry");
  view.unmount();
});
