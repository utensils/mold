import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import CreateHeader from "./CreateHeader.vue";
import createHeaderSource from "./CreateHeader.vue?raw";
import segmentedControlSource from "@ui/components/SegmentedControl.vue?raw";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useHostModelsStore } from "../../stores/hostModels";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useLastUsedStylesStore } from "@studio/stores/lastUsedStyles";
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

/** A second machine, pinned as the one Create sends work to. */
function pinRemote(entries: ModelEntry[]) {
  useHostsStore().extras.push({
    id: "plato-7680",
    label: "plato",
    url: "http://plato:7680",
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: null,
  });
  useHostModelsStore().byHost["plato-7680"] = { entries, fetchedAt: Date.now(), error: null };
  useAppPrefsStore().settings = { generateTargetHost: "plato-7680" } as never;
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

/** Listed FIRST wherever it appears, so a restore that ignores the section
 *  rule picks it up instead of the picture style. */
const clipModel = {
  ...stillModel,
  name: "ltx-video",
  family: "ltx-video",
  supports_sequence: true,
} as ModelEntry;

function outputSegments(wrapper: ReturnType<typeof mount>) {
  return wrapper.get("[data-test='output-kind']").findAll("button");
}

/** A form already holding a clip style — the Simple sub-mode on screen. */
function clipForm(): GenerateForm {
  const f = form();
  f.model = clipModel.name;
  f.family = clipModel.family;
  return f;
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

    /*
     * Short clip opens onto the REMEMBERED sub-mode. With no clip style on the
     * machine there is nothing Simple could select, so the door still opens
     * onto Scenes, which owns that empty state and says where to get one.
     */
    it("hands Short clip to the inspector when the machine has no clip style", async () => {
      readyLocal();
      installLocal([stillModel]);
      const wrapper = mount(CreateHeader, { props: { form: form() } });
      await outputSegments(wrapper)[1]!.trigger("click");
      expect(wrapper.emitted("set-output")).toEqual([["sequence"]]);
    });

    it("hands Short clip to the inspector when Scenes is the remembered way", async () => {
      readyLocal();
      installLocal([stillModel, clipModel]);
      useSequenceDraftStore().clipMode = "scenes";
      const wrapper = mount(CreateHeader, { props: { form: form() } });
      await outputSegments(wrapper)[1]!.trigger("click");
      expect(wrapper.emitted("set-output")).toEqual([["sequence"]]);
    });

    /*
     * Simple is the plain render: the output stays one shot and the STYLE is
     * what makes it a clip, so the header adopts one the way the 3-D door
     * adopts a 3-D style. Handing the inspector `single` would have restored a
     * PICTURE style, which is the opposite of what the door was asked for.
     */
    it("adopts a clip style in place for Simple, without an output switch", async () => {
      readyLocal();
      installLocal([stillModel, clipModel]);
      const store = useGenerateFormStore();
      store.form.model = stillModel.name;
      store.form.family = stillModel.family;
      const wrapper = mount(CreateHeader, { props: { form: store.form } });

      await outputSegments(wrapper)[1]!.trigger("click");

      expect(wrapper.emitted("set-output")).toBeUndefined();
      expect(store.form.model).toBe(clipModel.name);
      expect(useSequenceDraftStore().output).toBe("single");
    });

    it("restores the parked picture style on the way back out of a simple clip", async () => {
      readyLocal();
      const otherStill = { ...stillModel, name: "sdxl-base:fp16", family: "sdxl" } as ModelEntry;
      installLocal([otherStill, clipModel, stillModel]);
      const store = useGenerateFormStore();
      store.form.model = stillModel.name;
      store.form.family = stillModel.family;
      const wrapper = mount(CreateHeader, { props: { form: store.form } });

      await outputSegments(wrapper)[1]!.trigger("click");
      expect(store.form.model).toBe(clipModel.name);
      await outputSegments(wrapper)[0]!.trigger("click");
      expect(store.form.model).toBe(stillModel.name);
    });

    it("returns a clip draft to one shot", async () => {
      readyLocal();
      installLocal([stillModel]);
      useSequenceDraftStore().output = "sequence";
      const wrapper = mount(CreateHeader, { props: { form: form() } });
      await outputSegments(wrapper)[0]!.trigger("click");
      expect(wrapper.emitted("set-output")).toEqual([["single"]]);
    });

    /*
     * Each door opens onto the style its section was last used with, from
     * `lastUsedStyles`, and only falls to the first installed one when that
     * style is not on this machine. Parking still covers the immediate round
     * trip; the memory covers every later visit and the next launch.
     */
    it("opens Short clip on the clip style last used there", async () => {
      readyLocal();
      const wan = { ...clipModel, name: "wan22-ti2v-5b:dmd", family: "wan" } as ModelEntry;
      installLocal([stillModel, clipModel, wan]);
      useLastUsedStylesStore().remember("clip", wan.name);
      const store = useGenerateFormStore();
      const wrapper = mount(CreateHeader, { props: { form: store.form } });
      await outputSegments(wrapper)[1]!.trigger("click");
      expect(store.form.model).toBe(wan.name);
    });

    it("never opens a door onto a style no machine can run, remembered or first", async () => {
      // `targetModels` keeps downloaded-but-unrunnable rows so the picker can
      // disclose them, disabled; a door must skip them the way the picker
      // refuses them.
      readyLocal();
      const dead = {
        ...clipModel,
        name: "minimax-h3:exotic",
        family: "minimax-h3",
        runtime_available: false,
        runtime_unavailable_reason: "No loader for this layout.",
      } as ModelEntry;
      installLocal([stillModel, dead, clipModel]);
      useLastUsedStylesStore().remember("clip", dead.name);
      const store = useGenerateFormStore();
      const wrapper = mount(CreateHeader, { props: { form: store.form } });
      await outputSegments(wrapper)[1]!.trigger("click");
      expect(store.form.model).toBe(clipModel.name);
    });

    it("opens 3-D on the 3-D style last used there", async () => {
      readyLocal();
      const other = { ...meshModel, name: "hunyuan3d-full:fp16" } as ModelEntry;
      installLocal([stillModel, meshModel, other]);
      useLastUsedStylesStore().remember("mesh", other.name);
      const store = useGenerateFormStore();
      const wrapper = mount(CreateHeader, { props: { form: store.form } });
      await outputSegments(wrapper)[2]!.trigger("click");
      expect(store.form.model).toBe(other.name);
    });

    it("returns to Still picture on the picture style last used there when nothing is parked", async () => {
      readyLocal();
      const other = { ...stillModel, name: "flux2-klein-9b:q8", family: "flux2" } as ModelEntry;
      installLocal([stillModel, other, clipModel]);
      useLastUsedStylesStore().remember("still", other.name);
      const store = useGenerateFormStore();
      store.form.model = clipModel.name;
      store.form.family = clipModel.family;
      const wrapper = mount(CreateHeader, { props: { form: store.form } });
      await outputSegments(wrapper)[0]!.trigger("click");
      expect(store.form.model).toBe(other.name);
    });

    it("opens the 3-D door for a 3-D style and nothing else", () => {
      // The door's existence is the section rule's answer, so a clip style on
      // the machine can never be mistaken for a 3-D one.
      readyLocal();
      installLocal([stillModel, clipModel]);
      const wrapper = mount(CreateHeader, { props: { form: form() } });
      expect(outputSegments(wrapper).map((b) => b.text())).toEqual(["Still picture", "Short clip"]);
    });

    it("leaves 3-D holding a still style — never a clip one — when nothing was parked", async () => {
      readyLocal();
      installLocal([clipModel, stillModel, meshModel]);
      const store = useGenerateFormStore();
      store.form.model = meshModel.name;
      store.form.family = meshModel.family;
      const wrapper = mount(CreateHeader, { props: { form: store.form } });

      await outputSegments(wrapper)[0]!.trigger("click");

      expect(store.form.model).toBe(stillModel.name);
      expect(store.form.family).toBe("flux");
    });

    /**
     * The header used to partition `unionInstalled` — every style any machine
     * has — while the inspector partitioned the picker's own target rows. A
     * 3-D door that opens onto a style the pinned machine cannot run is a dead
     * end, so both read the one inventory.
     */
    it("offers only what the machine Create is aimed at can make", () => {
      readyLocal();
      installLocal([stillModel, meshModel]);
      pinRemote([stillModel]);
      const wrapper = mount(CreateHeader, { props: { form: form() } });
      expect(outputSegments(wrapper).map((b) => b.text())).toEqual(["Still picture", "Short clip"]);
    });

    /**
     * The parked still style is draft state, not component state: leaving New
     * image and coming back unmounts the header, and a `ref` took the parked
     * style with it — the return to Still picture then reached for the first
     * row on the machine instead of the style the person had been using.
     */
    it("keeps the parked still style across a visit to another workspace", async () => {
      readyLocal();
      const otherStill = { ...stillModel, name: "sdxl-base:fp16", family: "sdxl" } as ModelEntry;
      installLocal([otherStill, stillModel, meshModel]);
      const store = useGenerateFormStore();
      store.form.model = stillModel.name;
      store.form.family = stillModel.family;
      const wrapper = mount(CreateHeader, { props: { form: store.form } });

      await outputSegments(wrapper)[2]!.trigger("click");
      expect(store.form.model).toBe(meshModel.name);
      wrapper.unmount();

      const returned = mount(CreateHeader, { props: { form: store.form } });
      await outputSegments(returned)[0]!.trigger("click");
      expect(store.form.model).toBe(stillModel.name);
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

  /*
   * Simple | Scenes — how the video gets made — is `ClipModeStrip`'s, the row
   * beneath. Beside the kind control it pushed the whole right-hand cluster
   * left whenever Short clip was chosen, so the control a person had just clicked
   * jumped away from the pointer. The toolbar holds the same children in
   * every kind, so nothing on it ever moves.
   */
  describe("the row never changes shape with the kind", () => {
    function children(wrapper: ReturnType<typeof mount>) {
      const header = wrapper.get("[data-test='create-header']").element;
      return Array.from(header.children).map((el) => el.getAttribute("data-test") ?? el.className);
    }

    it("holds the same children for a picture, a video and a 3-D object", () => {
      readyLocal();
      installLocal([stillModel, clipModel, meshModel]);
      const still = children(mount(CreateHeader, { props: { form: form() } }));
      const clip = children(mount(CreateHeader, { props: { form: clipForm() } }));
      const mesh = form();
      mesh.family = meshModel.family;
      const threeD = children(mount(CreateHeader, { props: { form: mesh } }));
      expect(clip).toEqual(still);
      expect(threeD).toEqual(still);
    });

    it("carries no Simple | Scenes control of its own", () => {
      readyLocal();
      installLocal([stillModel, clipModel]);
      const wrapper = mount(CreateHeader, { props: { form: clipForm() } });
      expect(wrapper.find("[data-test='clip-mode']").exists()).toBe(false);
      expect(wrapper.text()).not.toContain("Scenes");
    });

    it("names Short clip the kind a one-shot on a clip style already is", () => {
      // The form holds a clip style with the output still one shot: that is
      // the Simple sub-mode, and calling it Still picture is the mislabelling
      // the section rule exists to end.
      readyLocal();
      installLocal([stillModel, clipModel]);
      const wrapper = mount(CreateHeader, { props: { form: clipForm() } });
      const on = outputSegments(wrapper).find((b) => b.attributes("aria-checked") === "true");
      expect(on?.text()).toBe("Short clip");
    });
  });

  it("carries Where it runs as the toolbar's last chip, after the doors", () => {
    // The routing pick used to sit at the foot of the inspector's Settings
    // list, where nobody found it. It is chrome now: always on screen, last
    // on the row, so the machine a print goes to is one glance away.
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    const chip = wrapper.get("[data-test='host-chip']");
    expect(chip.text()).toContain("This device");
    const header = wrapper.get("[data-test='create-header']").element;
    const order = Array.from(header.querySelectorAll("[data-test]")).map((el) =>
      el.getAttribute("data-test"),
    );
    expect(order.indexOf("host-chip")).toBeGreaterThan(order.indexOf("open-recent"));
    expect(order.at(-1)).toBe("host-chip");
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

    it("uses the 3-D placeholder for a 3-D style", async () => {
      readyLocal();
      const f = form();
      f.family = "hunyuan3d";
      const wrapper = mount(CreateHeader, { props: { form: f }, attachTo: document.body });
      await wrapper.get("[data-test='print-title']").trigger("click");
      expect(wrapper.get("[data-test='print-title-input']").attributes("placeholder")).toBe(
        "Untitled 3-D object",
      );
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
