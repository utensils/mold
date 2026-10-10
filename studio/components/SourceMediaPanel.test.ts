import { mount, flushPromises } from "@vue/test-utils";
import { beforeEach, expect, it, vi } from "vitest";
import SourceMediaPanel from "./SourceMediaPanel.vue";
import { clearSourceResources } from "../lib/sourceMediaResources";
const inventory = vi.hoisted(() => vi.fn());
vi.mock("../api/gallerySourceMedia", () => ({
  retainedSourceMediaInventory: inventory,
  retainedSourceMediaDisclosure: () => "unavailable",
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
