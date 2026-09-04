/**
 * The Recent tab: the pictures already made, newest first. The door that
 * opens it says "Use these settings again", so a row hands the whole print
 * back to the view rather than a prompt string.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import RecentPrints from "./RecentPrints.vue";
import type { MergedPrint } from "../../stores/gallery";

vi.mock("../gallery/AuthedMedia.vue", () => ({ default: { template: "<img />" } }));

function print(filename: string, title: string, model: string): MergedPrint {
  return {
    sourceKey: "local",
    hostLabel: "This device",
    item: {
      filename,
      timestamp: 1,
      size_bytes: 10,
      title,
      metadata: { model },
    },
  } as unknown as MergedPrint;
}

beforeEach(() => setActivePinia(createPinia()));

describe("RecentPrints", () => {
  it("lists the prints it is given, newest first as handed over", () => {
    const wrapper = mount(RecentPrints, {
      props: {
        prints: [
          print("teapot.png", "Brass teapot", "sdxl-base:fp16"),
          print("harbour.png", "Harbour", "flux-dev:q4"),
        ],
      },
    });
    const rows = wrapper.get("[data-test='recent-prints']").findAll("[data-test='recent-print']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("Brass teapot");
    expect(rows[1]!.text()).toContain("Harbour");
  });

  it("hands the whole print back on click", async () => {
    const entry = print("teapot.png", "Brass teapot", "sdxl-base:fp16");
    const wrapper = mount(RecentPrints, { props: { prints: [entry] } });
    await wrapper.get("[data-test='recent-print']").trigger("click");
    expect(wrapper.emitted("reuse")?.[0]).toEqual([entry]);
  });

  it("renders an empty list without a row", () => {
    const wrapper = mount(RecentPrints, { props: { prints: [] } });
    expect(
      wrapper.get("[data-test='recent-prints']").findAll("[data-test='recent-print']"),
    ).toHaveLength(0);
  });
});
