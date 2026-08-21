import { beforeEach, describe, expect, it } from "vitest";
import { nextTick } from "vue";
import { AUTO_TAG_SETTING_WEB } from "@studio/lib/fileUnder";
import {
  autoTagTitle,
  fileUnderPreviewName,
  loadAutoTagTitle,
  reloadAutoTagTitle,
  saveAutoTagTitle,
  titleTagWasApplied,
} from "./fileUnder";

describe("auto-tag-from-title preference", () => {
  beforeEach(() => {
    localStorage.clear();
    reloadAutoTagTitle();
  });

  it("defaults to on for a fresh browser", () => {
    expect(loadAutoTagTitle()).toBe(true);
    expect(autoTagTitle.value).toBe(true);
  });

  it("persists an explicit off under the shared studio key", () => {
    saveAutoTagTitle(false);
    expect(localStorage.getItem(AUTO_TAG_SETTING_WEB)).toBe("false");
    expect(loadAutoTagTitle()).toBe(false);
  });

  it("keeps a saved off across a reload", () => {
    localStorage.setItem(AUTO_TAG_SETTING_WEB, "false");
    reloadAutoTagTitle();
    expect(autoTagTitle.value).toBe(false);
  });

  it("falls back to on when the stored value is junk", () => {
    localStorage.setItem(AUTO_TAG_SETTING_WEB, "perhaps");
    reloadAutoTagTitle();
    expect(autoTagTitle.value).toBe(true);
  });

  it("writes through when the ref is set", async () => {
    autoTagTitle.value = false;
    await nextTick();
    expect(localStorage.getItem(AUTO_TAG_SETTING_WEB)).toBe("false");
  });
});

describe("fileUnderPreviewName", () => {
  it("mirrors the creation-time gallery grammar with the title slug", () => {
    expect(
      fileUnderPreviewName({
        model: "z-image-turbo:bf16",
        title: "Smurfs",
        ext: "png",
        timestamp: 1787320481000,
      }),
    ).toBe("mold-z-image-turbo-bf16-1787320481000~smurfs.png");
  });

  it("omits the ~slug segment for an untitled print", () => {
    expect(
      fileUnderPreviewName({
        model: "z-image-turbo:bf16",
        title: "",
        ext: "png",
        timestamp: 1787320481000,
      }),
    ).toBe("mold-z-image-turbo-bf16-1787320481000.png");
  });

  it("omits the ~slug segment when the title has nothing sluggable", () => {
    expect(
      fileUnderPreviewName({
        model: "flux-dev",
        title: "!!!",
        ext: "png",
        timestamp: 7,
      }),
    ).toBe("mold-flux-dev-7.png");
  });

  it("uses the request's output format as the extension", () => {
    expect(
      fileUnderPreviewName({
        model: "ltx2",
        title: "Riverbank at dusk",
        ext: "mp4",
        timestamp: 9,
      }),
    ).toBe("mold-ltx2-9~riverbank-at-dusk.mp4");
  });
});

describe("titleTagWasApplied", () => {
  it("recognises the title's own tag case-insensitively", () => {
    expect(titleTagWasApplied("Smurf 04!", ["blue", "Smurf-04"])).toBe(true);
  });

  it("is false when the print was filed without its title tag", () => {
    expect(titleTagWasApplied("Smurfs", ["blue"])).toBe(false);
  });

  it("is false for an untitled or unsluggable print", () => {
    expect(titleTagWasApplied(null, ["blue"])).toBe(false);
    expect(titleTagWasApplied("!!!", ["blue"])).toBe(false);
  });
});
