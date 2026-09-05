/**
 * The Recent tab: the pictures already made, newest first. The door that
 * opens it says "Use these settings again", so a row hands the whole print
 * back to the view rather than a prompt string.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import RecentPrints from "./RecentPrints.vue";
import { useGalleryStore, type MergedPrint } from "../../stores/gallery";

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

  it("says how long a print took beside its style, and nothing when it does not know", () => {
    // `OutputMetadata.generation_time_ms` is additive: an older host or a
    // synthesized row carries none, and the mono line then says only the
    // style — never "0.0s".
    const timed = print("teapot.png", "Brass teapot", "sdxl-base:fp16");
    (timed.item.metadata as { generation_time_ms?: number }).generation_time_ms = 4_000;
    const wrapper = mount(RecentPrints, {
      props: { prints: [timed, print("harbour.png", "Harbour", "flux-dev:q4")] },
    });
    const rows = wrapper.findAll("[data-test='recent-print']");
    expect(rows[0]!.get(".ms-recent__meta").text()).toBe("sdxl-base:fp16 · 4.0s");
    expect(rows[1]!.get(".ms-recent__meta").text()).toBe("flux-dev:q4");
  });

  it("hands the whole print back on click", async () => {
    const entry = print("teapot.png", "Brass teapot", "sdxl-base:fp16");
    const wrapper = mount(RecentPrints, { props: { prints: [entry] } });
    await wrapper.get("[data-test='recent-print']").trigger("click");
    expect(wrapper.emitted("reuse")?.[0]).toEqual([entry]);
  });

  /**
   * A resolved local bucket that carries no authority makes `targetOf` throw,
   * and a throw inside a render kills the whole inspector tab. The row is
   * worth showing without its picture: the recipe behind it still restores.
   */
  it("still lists a print whose bucket has no authority to hand it", () => {
    const gallery = useGalleryStore();
    // Resolved, but with no authority recorded — the shape `targetOf` refuses
    // to answer for, which the type deliberately cannot express.
    gallery.buckets.local = {
      ...gallery.ensureBucket("local"),
      authorityResolved: true,
    } as (typeof gallery.buckets)["local"];

    const wrapper = mount(RecentPrints, {
      props: { prints: [print("teapot.png", "Brass teapot", "sdxl-base:fp16")] },
    });

    expect(wrapper.get("[data-test='recent-print']").text()).toContain("Brass teapot");
    // ...and shows no picture rather than reading the filesystem directly
    // while the local server may well be running.
    expect(wrapper.findComponent({ name: "AuthedMedia" }).exists()).toBe(false);
  });

  it("renders an empty list without a row", () => {
    const wrapper = mount(RecentPrints, { props: { prints: [] } });
    expect(
      wrapper.get("[data-test='recent-prints']").findAll("[data-test='recent-print']"),
    ).toHaveLength(0);
  });
});
