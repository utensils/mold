import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import CreateHeader from "./CreateHeader.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";

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

function addRemote(id = "hal9000-7680", label = "hal9000") {
  useHostsStore().extras.push({
    id,
    label,
    url: `http://${label}:7680`,
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: null,
  });
}

describe("CreateHeader", () => {
  it("no longer renders the retired Single | Sequence switch", () => {
    // Output is a setting in the inspector now, not a place — the header
    // must not push a route to change modes.
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    expect(wrapper.find("[data-test='composer-mode']").exists()).toBe(false);
  });

  it("renders the live summary of shape, dimensions, and steps", () => {
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    expect(wrapper.get(".ms-header__title").text()).toBe("Untitled print");
    expect(wrapper.get(".ms-header__summary").text()).toBe("1:1 · 1024×1024 · 4 steps");
  });

  it("titles a sequence draft and summarizes clips + fps instead of steps", () => {
    readyLocal();
    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.ensureClips(97);
    const sequenceForm = form();
    sequenceForm.family = "ltx2";
    sequenceForm.width = 1216;
    sequenceForm.height = 704;
    sequenceForm.fps = 24;
    const wrapper = mount(CreateHeader, { props: { form: sequenceForm } });
    expect(wrapper.get(".ms-header__title").text()).toBe("Untitled sequence");
    expect(wrapper.get(".ms-header__summary").text()).toBe("16:9 · 1216×704 · 2 clips · 24 fps");
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
      expect(input.attributes("placeholder")).toBe("Untitled print");
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
      await input.setValue("bad\u0007title");
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

    it("uses the sequence placeholder for a sequence draft", async () => {
      readyLocal();
      useSequenceDraftStore().output = "sequence";
      const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
      await wrapper.get("[data-test='print-title']").trigger("click");
      expect(wrapper.get("[data-test='print-title-input']").attributes("placeholder")).toBe(
        "Untitled sequence",
      );
    });
  });

  it("does not open a routing menu with a single host", async () => {
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("toggles the routing menu open and closed from the chip", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(true);
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("lists Auto, Most capable, and every host; picking one persists and closes", async () => {
    readyLocal();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: null } as never;
    const update = vi.spyOn(prefs, "update").mockResolvedValue(undefined as never);
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-option-auto']").exists()).toBe(true);
    expect(wrapper.find("[data-test='host-option-capable']").exists()).toBe(true);
    await wrapper.get("[data-test='host-option-hal9000-7680']").trigger("click");
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ generateTargetHost: "hal9000-7680" });
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("maps Auto back to null in the persisted setting", async () => {
    readyLocal();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: "hal9000-7680" } as never;
    const update = vi.spyOn(prefs, "update").mockResolvedValue(undefined as never);
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    await wrapper.get("[data-test='host-option-auto']").trigger("click");
    expect(update).toHaveBeenCalledWith({ generateTargetHost: null });
  });

  it("shows a stale persisted pick as Auto when the host is gone", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: "ghost-7680" } as never;
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.get("[data-test='host-option-auto']").attributes("aria-checked")).toBe("true");
  });

  it("names the sticky pick on the chip", () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    expect(wrapper.get("[data-test='host-chip']").text()).toContain("hal9000");
  });

  it("closes the menu on Escape", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });
});
