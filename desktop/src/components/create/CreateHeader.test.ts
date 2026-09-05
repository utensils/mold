import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import CreateHeader from "./CreateHeader.vue";
import createHeaderSource from "./CreateHeader.vue?raw";
import segmentedControlSource from "@ui/components/SegmentedControl.vue?raw";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useHostModelsStore } from "../../stores/hostModels";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ModelEntry } from "../../lib/api/types";

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    appSettingsGet: vi.fn().mockResolvedValue({}),
  },
}));

const { routerPush } = vi.hoisted(() => ({ routerPush: vi.fn() }));
vi.mock("vue-router", () => ({ useRouter: () => ({ push: routerPush }) }));

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

function form(): GenerateForm {
  return reactive({ ...newGenerateForm(), family: "flux" });
}

function readyLocal() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
}

/** Install models on the local host so the header's mesh door can appear. */
function installLocal(entries: ModelEntry[]) {
  useHostModelsStore().byHost.local = { entries, fetchedAt: Date.now(), error: null };
}

const meshModel = {
  name: "hunyuan3d-mini-turbo:fp16",
  family: "hunyuan3d",
  downloaded: true,
  default_width: 0,
  default_height: 0,
  default_steps: 5,
  default_guidance: 5,
} as ModelEntry;

const stillModel = {
  name: "flux-dev:q8",
  family: "flux",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 20,
  default_guidance: 4.5,
} as ModelEntry;

function outputSegments(wrapper: ReturnType<typeof mount>) {
  return wrapper.get("[data-test='output-kind']").findAll("button");
}

/*
 * The toolbar must hold one row at every width the window can reach
 * (`minWidth: 1080` in src-tauri/tauri.conf.json, minus the sidebar and a
 * dragged-wide inspector). "Still picture" wrapped onto two lines and doubled
 * the toolbar's height: `.ms-seg__btn` is `flex: 1` with wrappable text, so
 * its min-content width is only its longest WORD and the flex line let it
 * shrink there. The order of yielding is title first, then the doors' labels;
 * the segments never wrap.
 */
describe("CreateHeader — the toolbar holds one row", () => {
  it("keeps every segment on one line", () => {
    expect(segmentedControlSource).toMatch(/\.ms-seg__btn\s*\{[^}]*white-space:\s*nowrap/s);
  });

  /*
   * The output kind picks a SETTING, so it takes the neutral treatment. In
   * the accent it was louder than the mock and, sitting on the same view as
   * the accent-tinted Quality rows and mesh ladder, it stopped saying which
   * of the two kinds of choice it was.
   */
  it("paints the output-kind control neutral, not accent-tinted", () => {
    expect(createHeaderSource).toMatch(/<SegmentedControl[^>]*variant="neutral"/s);
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    expect(wrapper.get("[data-test='output-kind']").classes()).toContain("ms-seg--neutral");
  });

  it("never lets the segmented control shrink below its own segments", () => {
    expect(createHeaderSource).toMatch(/\.ms-header__seg\s*\{[^}]*flex:\s*0\s+0\s+auto/s);
    expect(createHeaderSource).toMatch(/<SegmentedControl[^>]*class="ms-header__seg"/s);
  });

  it("truncates the title first", () => {
    expect(createHeaderSource).toMatch(/\.ms-header__title\s*\{[^}]*min-width:\s*0/s);
    expect(createHeaderSource).toMatch(
      /\.ms-header__title-text\s*\{[^}]*text-overflow:\s*ellipsis/s,
    );
    expect(createHeaderSource).toMatch(/\.ms-header__title-text\s*\{[^}]*white-space:\s*nowrap/s);
  });

  it("drops the doors' labels to icons before anything wraps", () => {
    // A container query, not a viewport one: the toolbar's width is the
    // window minus the sidebar and a user-dragged inspector.
    expect(createHeaderSource).toMatch(/\.ms-header\s*\{[^}]*container-type:\s*inline-size/s);
    expect(createHeaderSource).toMatch(/container-name:\s*create-header/s);
    expect(createHeaderSource).toMatch(
      /@container create-header \(max-width:[^)]*\)\s*\{[\s\S]*?\.ms-header__door-label\s*\{[^}]*display:\s*none/s,
    );
  });

  it("names an icon-only door for the reader who cannot see the icon", () => {
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    for (const id of ["open-starters", "open-recent"]) {
      const door = wrapper.get(`[data-test='${id}']`);
      expect(door.attributes("aria-label"), id).toBeTruthy();
      expect(door.attributes("title"), id).toBe(door.attributes("aria-label"));
      expect(door.find(".ms-header__door-label").exists(), id).toBe(true);
    }
  });
});

describe("CreateHeader", () => {
  it("no longer renders the retired Single | Sequence switch", () => {
    // Output is a setting, not a place — the header must not push a route to
    // change modes.
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    expect(wrapper.find("[data-test='composer-mode']").exists()).toBe(false);
    expect(routerPush).not.toHaveBeenCalled();
  });

  describe("what to make", () => {
    it("offers Still picture and Short clip, with the 3-D door only where a 3-D style is installed", () => {
      readyLocal();
      installLocal([stillModel]);
      const wrapper = mount(CreateHeader, { props: { form: form() } });
      expect(outputSegments(wrapper).map((b) => b.text())).toEqual(["Still picture", "Short clip"]);

      installLocal([stillModel, meshModel]);
      const withMesh = mount(CreateHeader, { props: { form: form() } });
      expect(outputSegments(withMesh).map((b) => b.text())).toEqual([
        "Still picture",
        "Short clip",
        "3-D object",
      ]);
    });

    it("hands Short clip to the inspector, which owns the model swap", async () => {
      readyLocal();
      installLocal([stillModel]);
      const wrapper = mount(CreateHeader, { props: { form: form() } });
      await outputSegments(wrapper)[1]!.trigger("click");
      expect(wrapper.emitted("set-output")).toEqual([["sequence"]]);
    });

    it("returns a clip draft to one shot", async () => {
      readyLocal();
      installLocal([stillModel]);
      useSequenceDraftStore().output = "sequence";
      const wrapper = mount(CreateHeader, { props: { form: form() } });
      await outputSegments(wrapper)[0]!.trigger("click");
      expect(wrapper.emitted("set-output")).toEqual([["single"]]);
    });

    it("applies the first installed 3-D style, and restores the parked still style", async () => {
      readyLocal();
      installLocal([stillModel, meshModel]);
      const store = useGenerateFormStore();
      store.form.model = stillModel.name;
      store.form.family = stillModel.family;
      const wrapper = mount(CreateHeader, { props: { form: store.form } });

      await outputSegments(wrapper)[2]!.trigger("click");
      expect(store.form.model).toBe(meshModel.name);
      expect(store.form.family).toBe("hunyuan3d");

      await outputSegments(wrapper)[0]!.trigger("click");
      expect(store.form.model).toBe(stillModel.name);
    });
  });

  it("opens the inspector's Starters and Recent tabs instead of floating a popover", async () => {
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    await wrapper.get("[data-test='open-starters']").trigger("click");
    await wrapper.get("[data-test='open-recent']").trigger("click");
    expect(wrapper.emitted("open-tab")).toEqual([["starters"], ["recent"]]);
  });

  describe("editable print title", () => {
    it("shows the form title in place of the placeholder", () => {
      readyLocal();
      const titled = form();
      titled.title = "Smurf village";
      const wrapper = mount(CreateHeader, { props: { form: titled } });
      expect(wrapper.get("[data-test='print-title']").text()).toBe("Smurf village");
      expect(wrapper.get("[data-test='print-title']").attributes("aria-label")).toBe("Print title");
    });

    it("opens an input on click, commits on Enter, and writes the trimmed title", async () => {
      readyLocal();
      const f = form();
      const wrapper = mount(CreateHeader, { props: { form: f }, attachTo: document.body });
      await wrapper.get("[data-test='print-title']").trigger("click");
      const input = wrapper.get<HTMLInputElement>("[data-test='print-title-input']");
      expect(input.attributes("aria-label")).toBe("Print title");
      expect(input.attributes("placeholder")).toBe("Untitled picture");
      // No raw maxlength: it counts UTF-16 code units, truncating emoji
      // titles early. The scalar-aware validator enforces the 120 limit.
      expect(input.attributes("maxlength")).toBeUndefined();
      await input.setValue("  Smurf village  ");
      await input.trigger("keydown", { key: "Enter" });
      expect(f.title).toBe("Smurf village");
      expect(wrapper.find("[data-test='print-title-input']").exists()).toBe(false);
      expect(wrapper.get("[data-test='print-title']").text()).toBe("Smurf village");
    });

    it("commits on blur and reverts on Escape", async () => {
      readyLocal();
      const f = form();
      f.title = "Before";
      const wrapper = mount(CreateHeader, { props: { form: f }, attachTo: document.body });
      await wrapper.get("[data-test='print-title']").trigger("click");
      let input = wrapper.get<HTMLInputElement>("[data-test='print-title-input']");
      await input.setValue("Typed then escaped");
      await input.trigger("keydown", { key: "Escape" });
      expect(f.title).toBe("Before");
      expect(wrapper.find("[data-test='print-title-input']").exists()).toBe(false);

      await wrapper.get("[data-test='print-title']").trigger("click");
      input = wrapper.get<HTMLInputElement>("[data-test='print-title-input']");
      await input.setValue("After");
      await input.trigger("blur");
      expect(f.title).toBe("After");
    });

    it("refuses an invalid title, keeping the editor open with the reason", async () => {
      readyLocal();
      const f = form();
      const wrapper = mount(CreateHeader, { props: { form: f }, attachTo: document.body });
      await wrapper.get("[data-test='print-title']").trigger("click");
      const input = wrapper.get<HTMLInputElement>("[data-test='print-title-input']");
      await input.setValue("badtitle");
      await input.trigger("keydown", { key: "Enter" });
      expect(f.title).toBe("");
      expect(wrapper.find("[data-test='print-title-input']").exists()).toBe(true);
      expect(input.attributes("aria-invalid")).toBe("true");
      expect(wrapper.get("[data-test='print-title-error']").text()).toContain("control");
    });

    it("accepts a 120-emoji title and refuses 121 via the scalar-aware validator", async () => {
      // Each 🦋 is 2 UTF-16 code units but ONE Unicode scalar — a raw
      // maxlength="120" would have truncated this valid title at 60 emoji.
      readyLocal();
      const f = form();
      const wrapper = mount(CreateHeader, { props: { form: f }, attachTo: document.body });
      await wrapper.get("[data-test='print-title']").trigger("click");
      const input = wrapper.get<HTMLInputElement>("[data-test='print-title-input']");
      const emoji120 = "🦋".repeat(120);
      await input.setValue(emoji120);
      await input.trigger("keydown", { key: "Enter" });
      expect(f.title).toBe(emoji120);
      expect(wrapper.find("[data-test='print-title-input']").exists()).toBe(false);

      await wrapper.get("[data-test='print-title']").trigger("click");
      const reopened = wrapper.get<HTMLInputElement>("[data-test='print-title-input']");
      await reopened.setValue("🦋".repeat(121));
      await reopened.trigger("keydown", { key: "Enter" });
      // The commit is blocked: the editor stays open with the inline reason
      // and the form keeps the previous title.
      expect(f.title).toBe(emoji120);
      expect(wrapper.find("[data-test='print-title-input']").exists()).toBe(true);
      expect(reopened.attributes("aria-invalid")).toBe("true");
      expect(wrapper.get("[data-test='print-title-error']").text()).toContain("120");
    });

    it("uses the clip placeholder for a sequence draft", async () => {
      readyLocal();
      useSequenceDraftStore().output = "sequence";
      const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
      await wrapper.get("[data-test='print-title']").trigger("click");
      expect(wrapper.get("[data-test='print-title-input']").attributes("placeholder")).toBe(
        "Untitled clip",
      );
    });
  });
});
