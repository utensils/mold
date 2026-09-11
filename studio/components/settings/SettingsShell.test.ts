import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { sectionsForSurface, type SectionId } from "../../lib/settingsSchema";
import SettingsShell from "./SettingsShell.vue";

/*
 * The shell is the Settings page's frame: a search field, a jump nav, and one
 * scroll of always-open sections whose bodies arrive lazily.
 *
 * Lazily matters. A body is a live component — Advanced alone opens three HTTP
 * calls and a device subscription on mount — and most visits never scroll to
 * it. But a body that arrived must never leave, or a half-typed edit vanishes
 * when the page scrolls past.
 */

type ObserverCallback = (entries: IntersectionObserverEntry[]) => void;

const observers: {
  callback: ObserverCallback;
  options: IntersectionObserverInit;
}[] = [];

class FakeIntersectionObserver {
  constructor(
    public callback: ObserverCallback,
    public options: IntersectionObserverInit = {},
  ) {
    observers.push({ callback, options });
  }
  observe() {}
  unobserve() {}
  disconnect() {}
}

/** Report a section as the top one on the page, as the scroll-spy would. */
function reportVisible(id: SectionId, index = 0) {
  const target = document.querySelector(`[data-section='${id}']`);
  observers[index]?.callback([
    {
      isIntersecting: true,
      target: target ?? document.createElement("div"),
      boundingClientRect: { top: 0 } as DOMRect,
    } as unknown as IntersectionObserverEntry,
  ]);
}

const sections = sectionsForSurface("web");

function mountShell() {
  return mount(SettingsShell, {
    props: { sections },
    slots: {
      section: `<template #section="{ section, mounted }">
        <div v-if="mounted" :data-test="'body-' + section.id">{{ section.label }} body</div>
      </template>`,
    },
    attachTo: document.body,
  });
}

beforeEach(() => {
  observers.length = 0;
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.body.innerHTML = "";
});

describe("SettingsShell without an IntersectionObserver", () => {
  it("mounts every body eagerly — no scroll signal, so an eager page beats an empty one", async () => {
    vi.stubGlobal("IntersectionObserver", undefined);
    const wrapper = mountShell();
    await nextTick();
    for (const section of sections) {
      expect(
        wrapper.find(`[data-test='body-${section.id}']`).exists(),
        section.id,
      ).toBe(true);
    }
  });
});

describe("SettingsShell navigation", () => {
  beforeEach(() =>
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver),
  );

  it("lists every section it was given, and starts on the first", () => {
    const wrapper = mountShell();
    for (const section of sections) {
      expect(
        wrapper.find(`[data-test='settings-nav-${section.id}']`).exists(),
      ).toBe(true);
    }
    expect(
      wrapper
        .get(`[data-test='settings-nav-${sections[0]!.id}']`)
        .attributes("aria-current"),
    ).toBe("true");
  });

  /* The frame is the shell's, not each caller's: a body that had to draw its
   * own label and panel is how web and desktop came to look different in the
   * first place. */
  it("frames each section with its own label and summary", () => {
    const wrapper = mountShell();
    const cloud = wrapper.get("[data-test='section-cloud']");
    expect(cloud.text()).toContain("Cloud GPUs");
    expect(cloud.text()).toContain("Rented machines, their keys");
  });

  it("jumping mounts that body and moves the highlight", async () => {
    const wrapper = mountShell();
    expect(wrapper.find("[data-test='body-advanced']").exists()).toBe(false);
    await wrapper.get("[data-test='settings-nav-advanced']").trigger("click");
    expect(wrapper.find("[data-test='body-advanced']").exists()).toBe(true);
    expect(
      wrapper
        .get("[data-test='settings-nav-advanced']")
        .attributes("aria-current"),
    ).toBe("true");
    expect(wrapper.emitted("update:active")?.at(-1)).toEqual(["advanced"]);
  });

  /* A smooth scroll passes every section on its way. Without the hold, the
   * highlight raced down the nav and landed wherever the scroll finished
   * reporting. */
  it("holds the jumped-to section through the settling window", async () => {
    vi.useFakeTimers();
    const wrapper = mountShell();
    await wrapper.get("[data-test='settings-nav-advanced']").trigger("click");

    reportVisible("generation", 1);
    await nextTick();
    expect(
      wrapper
        .get("[data-test='settings-nav-advanced']")
        .attributes("aria-current"),
    ).toBe("true");

    vi.advanceTimersByTime(800);
    reportVisible("generation", 1);
    await nextTick();
    expect(
      wrapper
        .get("[data-test='settings-nav-generation']")
        .attributes("aria-current"),
    ).toBe("true");
  });

  it("can be jumped from outside — the caller owns `?section=`", async () => {
    const wrapper = mountShell();
    (wrapper.vm as unknown as { jump: (id: SectionId) => void }).jump(
      "updates",
    );
    await nextTick();
    expect(
      wrapper
        .get("[data-test='settings-nav-updates']")
        .attributes("aria-current"),
    ).toBe("true");
    expect(wrapper.find("[data-test='body-updates']").exists()).toBe(true);
  });

  it("keeps a body once it has arrived", async () => {
    const wrapper = mountShell();
    await wrapper.get("[data-test='settings-nav-advanced']").trigger("click");
    await wrapper.get("[data-test='settings-nav-generation']").trigger("click");
    expect(wrapper.find("[data-test='body-advanced']").exists()).toBe(true);
  });
});

describe("SettingsShell search", () => {
  beforeEach(() =>
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver),
  );

  it("narrows the nav and the page together", async () => {
    const wrapper = mountShell();
    await wrapper.get("[data-test='settings-search']").setValue("runpod");
    expect(wrapper.find("[data-test='settings-nav-cloud']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='settings-nav-generation']").exists()).toBe(
      false,
    );
    expect(wrapper.find("[data-test='section-generation']").exists()).toBe(
      false,
    );
  });

  /* A match IS reached: the person asked for it by name and there is nothing
   * to scroll past. It also survives the search being cleared, so a body
   * holding a half-typed edit does not unmount. */
  it("mounts a searched section, and keeps it after the search is cleared", async () => {
    const wrapper = mountShell();
    await wrapper.get("[data-test='settings-search']").setValue("runpod");
    expect(wrapper.find("[data-test='body-cloud']").exists()).toBe(true);
    await wrapper.get("[data-test='settings-search']").setValue("");
    expect(wrapper.find("[data-test='body-cloud']").exists()).toBe(true);
  });

  it("matches a section on the raw keys the caller says it renders", async () => {
    const wrapper = mount(SettingsShell, {
      props: {
        sections,
        rawKeysBySection: { styleDefaults: ["models.z-image.lora"] },
      },
      slots: {
        section: `<template #section="{ section, mounted }">
          <div v-if="mounted" :data-test="'body-' + section.id">{{ section.label }}</div>
        </template>`,
      },
      attachTo: document.body,
    });
    await wrapper.get("[data-test='settings-search']").setValue("z-image");
    expect(
      wrapper.find("[data-test='settings-nav-styleDefaults']").exists(),
    ).toBe(true);
    expect(wrapper.find("[data-test='settings-nav-advanced']").exists()).toBe(
      false,
    );
  });

  it("says so when nothing matches", async () => {
    const wrapper = mountShell();
    await wrapper.get("[data-test='settings-search']").setValue("zzzznothing");
    expect(wrapper.get("[data-test='no-search-results']").text()).toContain(
      "zzzznothing",
    );
  });

  it("takes the placeholder the caller gives it", () => {
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver);
    const wrapper = mount(SettingsShell, {
      props: { sections, searchPlaceholder: "Search settings…" },
      attachTo: document.body,
    });
    expect(
      wrapper.get("[data-test='settings-search']").attributes("placeholder"),
    ).toBe("Search settings…");
  });
});

describe("SettingsShell layout", () => {
  it("is a sticky column at 900px and a scrolling chip strip below it", async () => {
    // A browser page has no fixed second pane, so the nav folds rather than
    // being dropped: below 900px it is a horizontally scrolling strip.
    const source = (await import("./SettingsShell.vue?raw")).default;
    expect(source).toMatch(/position:\s*sticky/);
    expect(source).toMatch(/var\(--mold-shell-settingsnav-w,\s*200px\)/);
    expect(source).toMatch(/@media\s*\(max-width:\s*899px\)/);
    expect(source).toMatch(/overflow-x:\s*auto/);
  });
});

describe("SettingsShell pane layout", () => {
  /*
   * A browser page scrolls the window, so fifteen always-open sections make a
   * seven-screen page whatever the nav does — which is the complaint that
   * started this. In `layout="pane"` the nav is a real navigation: one section
   * is on screen at a time, a nav click swaps it, and search shows every match
   * stacked so a word still finds its row wherever it lives.
   */
  beforeEach(() =>
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver),
  );
  afterEach(() => vi.unstubAllGlobals());

  function mountPane() {
    return mount(SettingsShell, {
      props: { sections, layout: "pane" },
      slots: {
        section: `<template #section="{ section, mounted }">
          <div v-if="mounted" :data-test="'body-' + section.id">{{ section.label }} body</div>
        </template>`,
      },
      attachTo: document.body,
    });
  }

  it("shows exactly one section, the active one, with its body mounted", () => {
    const wrapper = mountPane();
    expect(wrapper.findAll("[data-test^='section-']")).toHaveLength(1);
    expect(wrapper.find("[data-test='section-app']").exists()).toBe(true);
    expect(wrapper.find("[data-test='body-app']").exists()).toBe(true);
    expect(wrapper.findAll("[data-test^='settings-nav-']")).toHaveLength(
      sections.length,
    );
    wrapper.unmount();
  });

  it("a nav click swaps the pane and reports the active section", async () => {
    const wrapper = mountPane();
    await wrapper.get("[data-test='settings-nav-cloud']").trigger("click");
    expect(wrapper.findAll("[data-test^='section-']")).toHaveLength(1);
    expect(wrapper.find("[data-test='section-cloud']").exists()).toBe(true);
    expect(wrapper.find("[data-test='section-app']").exists()).toBe(false);
    expect(wrapper.emitted("update:active")?.at(-1)).toEqual(["cloud"]);
    expect(
      wrapper
        .get("[data-test='settings-nav-cloud']")
        .attributes("aria-current"),
    ).toBe("true");
    wrapper.unmount();
  });

  it("jump from outside swaps the pane without waiting on any scroll", async () => {
    const wrapper = mountPane();
    (wrapper.vm as unknown as { jump: (id: SectionId) => void }).jump(
      "library",
    );
    await nextTick();
    expect(wrapper.find("[data-test='section-library']").exists()).toBe(true);
    expect(wrapper.findAll("[data-test^='section-']")).toHaveLength(1);
    wrapper.unmount();
  });

  it("search stacks every match, then returns to the one pane when cleared", async () => {
    const wrapper = mountPane();
    await wrapper.get("[data-test='settings-search']").setValue("trash");
    const shown = wrapper
      .findAll("[data-test^='section-']")
      .map((el) => el.attributes("data-test"));
    expect(shown).toContain("section-library");
    expect(shown.length).toBeGreaterThanOrEqual(1);
    await wrapper.get("[data-test='settings-search']").setValue("");
    expect(wrapper.findAll("[data-test^='section-']")).toHaveLength(1);
    wrapper.unmount();
  });

  it("creates no scroll-spy — there is nothing to spy on", () => {
    observers.length = 0;
    const wrapper = mountPane();
    expect(observers).toHaveLength(0);
    wrapper.unmount();
  });
});

describe("SettingsShell scroll ownership", () => {
  beforeEach(() =>
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver),
  );
  afterEach(() => vi.unstubAllGlobals());

  it("observes against the viewport when the page scrolls, and against its own column when it owns the scroll", () => {
    observers.length = 0;
    const page = mount(SettingsShell, {
      props: { sections },
      attachTo: document.body,
    });
    expect(observers.map((o) => o.options.root ?? null)).toEqual([null, null]);
    page.unmount();

    observers.length = 0;
    const own = mount(SettingsShell, {
      props: { sections, scroll: "content" },
      attachTo: document.body,
    });
    const content = own.get(".ms-settings-content").element;
    expect(observers.map((o) => o.options.root)).toEqual([content, content]);
    expect(own.get(".ms-settings-shell").classes()).toContain(
      "ms-settings-shell--own-scroll",
    );
    own.unmount();
  });

  it("stretches the folded layout so the chip strip, not the page, scrolls sideways", async () => {
    // In a column flex layout `align-items: start` is the HORIZONTAL axis, so
    // the nav shrink-wrapped to 1,423px inside an 880px viewport.
    const source = (await import("./SettingsShell.vue?raw")).default;
    const folded = source.slice(source.indexOf("@media (max-width: 899px)"));
    expect(folded).toMatch(/align-items:\s*stretch/);
    expect(folded).toMatch(/align-self:\s*stretch/);
  });
});
