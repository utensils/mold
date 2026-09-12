import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { reactive } from "vue";
import SettingsPage from "./SettingsPage.vue";
import settingsPageSource from "./SettingsPage.vue?raw";
import { matchSystem, theme } from "../lib/theme";
import { resetNotifications, useNotifications } from "../lib/toasts";
import { HOSTS_STORAGE_KEY, originHost } from "../lib/hostRegistry";
import { AUTO_TAG_SETTING_WEB } from "@studio/lib/fileUnder";
import { autoTagTitle, reloadAutoTagTitle } from "../lib/fileUnder";
import {
  ENGINE_KEY_SCHEMAS,
  PER_STYLE_FIELDS,
  sectionsForSurface,
} from "@studio/lib/settingsSchema";
import type { ConfigRow } from "@studio/api/config";
import type { ServerStatus } from "../types";

const statusRef = vi.hoisted(() => ({ value: null as ServerStatus | null }));
const subscribeToDeviceSnapshots = vi.hoisted(() => vi.fn());
const routeState = vi.hoisted(() => ({ query: {} as Record<string, string> }));
const pushMock = vi.hoisted(() => vi.fn());
const replaceMock = vi.hoisted(() => vi.fn());

vi.mock("../composables/useStatusPoll", () => ({
  useStatusPoll: () => ({ status: statusRef }),
}));
vi.mock("../lib/deviceEvents", () => ({ subscribeToDeviceSnapshots }));
vi.mock("vue-router", () => ({
  useRoute: () => reactive(routeState),
  useRouter: () => ({ push: pushMock, replace: replaceMock }),
  RouterLink: {
    props: { to: { type: [String, Object], required: true } },
    template: '<a :href=\'typeof to === "string" ? to : ""\'><slot /></a>',
  },
}));

/*
 * On web the shell runs as PANES: one section is on screen at a time and the
 * nav is a navigation, so a test that reads a section opens it first — the
 * way a person does. (There is no scroll-spy to stub; a pane creates none.)
 */
async function openSection(wrapper: VueWrapper, id: string) {
  await wrapper.get(`[data-test="settings-nav-${id}"]`).trigger("click");
  await flushPromises();
}

const originalFetch = globalThis.fetch;

function deviceWire(enabled = true, adminState = "enabled", ordinal = 0) {
  const suffix = String.fromCharCode("a".charCodeAt(0) + ordinal);
  return {
    id: `cuda:${suffix.repeat(32)}`,
    backend: "cuda",
    ordinal,
    device_kind: "full_gpu",
    nvml_uuid: `GPU-${suffix}`,
    physical_uuid: `GPU-${suffix}`,
    mig_uuid: null,
    mig_parent_uuid: null,
    mig_profile: null,
    name: `NVIDIA RTX 3090 #${ordinal}`,
    pci_bus_id: null,
    compute_capability: "8.6",
    memory: {
      total_bytes: 24_000_000_000,
      used_bytes: 4_000_000_000,
      mold_used_bytes: null,
      other_used_bytes: null,
    },
    telemetry: {
      utilization_percent: 10,
      temperature_c: null,
      power_w: null,
    },
    desired_enabled: enabled,
    admin_state: adminState,
    health: "healthy",
    activity: "idle",
    schedulable: enabled,
    unschedulable_reason: enabled ? null : "device_disabled",
    loaded_models: [],
    active_work_id: null,
    planned_work_ids: [],
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

/*
 * hal9000's shape: every engine key this build curates, thirteen styles of
 * per-style overrides, and one key from a server newer than this client. It
 * is the machine the redesign is for — 150-odd rows, 104 of them
 * `models.<style>.<field>`.
 */
const FIXTURE_STYLES = [
  "flux-dev:q4",
  "flux-schnell:q8",
  "flux2-klein:bf16",
  "hunyuan3d-2.1:fp16",
  "ltx-2:fp8",
  "ltx-video:fp16",
  "qwen-image:q4",
  "sd1.5",
  "sd3.5-large:q8",
  "sdxl",
  "wan21-t2v-1.3b:turbo",
  "wuerstchen-v2",
  "z-image:fp16",
];

function hal9000ConfigRows(): ConfigRow[] {
  const rows: ConfigRow[] = ENGINE_KEY_SCHEMAS.map((schema) => ({
    key: schema.key,
    value:
      schema.editor === "toggle"
        ? false
        : schema.editor === "number" || schema.editor === "slider"
          ? 1
          : "",
    source: "db" as const,
  }));
  for (const style of FIXTURE_STYLES) {
    for (const field of PER_STYLE_FIELDS) {
      rows.push({
        key: `models.${style}.${field}`,
        value: field.startsWith("default_") || field === "lora_scale" ? 8 : "",
        source: "db",
      });
    }
  }
  // A key this client has never heard of — the ONLY thing Advanced is for.
  rows.push({ key: "future.option", value: "on", source: "db" });
  return rows;
}

function configRespondingFetch(rows: ConfigRow[]) {
  return vi.fn(async (input) => {
    const url = String(input);
    const body = url.includes("/api/config/profiles")
      ? { profiles: ["default"], active: "default" }
      : url.endsWith("/api/config")
        ? { profile: "default", entries: rows }
        : url.endsWith("/api/models")
          ? []
          : url.endsWith("/api/catalog/credentials")
            ? {
                hf: { configured: false, source: null, masked: null },
                civitai: { configured: false, source: null, masked: null },
              }
            : { entries: [] };
    return { ok: true, json: async () => body } as Response;
  }) as unknown as typeof fetch;
}

describe("SettingsPage", () => {
  beforeEach(() => {
    statusRef.value = null;
    routeState.query = {};
    pushMock.mockClear();
    replaceMock.mockClear();
    subscribeToDeviceSnapshots.mockClear();
    resetNotifications();
    localStorage.clear();
    reloadAutoTagTitle();
    globalThis.fetch = vi.fn(
      async (input) =>
        ({
          ok: true,
          json: async () => {
            if (String(input).endsWith("/profiles"))
              return { profiles: ["default"], active: "default" };
            if (String(input).endsWith("/api/models")) return [];
            if (String(input).endsWith("/api/catalog/credentials")) {
              return {
                hf: { configured: false, source: null, masked: null },
                civitai: { configured: false, source: null, masked: null },
              };
            }
            if (String(input).endsWith("/api/pairing/clients")) {
              return {
                auth_required: true,
                pairing_available: true,
                clients: [],
              };
            }
            return { entries: [] };
          },
        }) as Response,
    ) as typeof fetch;
  });
  afterEach(() => {
    vi.useRealTimers();
    globalThis.fetch = originalFetch;
    theme.value = "safelight-dark";
    matchSystem.value = false;
    vi.restoreAllMocks();
  });

  it("keeps its padded content inside narrow web viewports", () => {
    const wrapper = mount(SettingsPage);
    // The page frame is the shared workspace column; the page's own rule adds
    // nothing that could overflow it.
    // `get` throws when the frame is missing, which is the assertion.
    wrapper.get("div.workspace-page.settings-page");

    const pageRule = settingsPageSource.match(/\.settings-page\s*\{([^}]*)\}/s);
    expect(pageRule).not.toBeNull();
    const declarations = pageRule?.[1] ?? "";
    expect(declarations).toMatch(/width:\s*100%/);
    expect(declarations).toMatch(/box-sizing:\s*border-box/);
  });

  it("persists the theme through the shared lib/theme refs, keeping the tone", async () => {
    const wrapper = mount(SettingsPage);
    // Section bodies arrive on the shell's own mounted hook, one tick later.
    await flushPromises();
    // A card names a THEME and nothing else — no option carries a tone.
    const cards = wrapper.findAll('[data-test="theme-select"] [role="radio"]');
    expect(cards.map((card) => card.attributes("data-test"))).toEqual([
      "theme-mocha",
      "theme-safelight",
      "theme-blueprint",
      "theme-graphite",
      "theme-nebula",
    ]);

    await wrapper.get('[data-test="theme-graphite"]').trigger("click");
    await flushPromises();
    expect(theme.value).toBe("graphite-dark");
    expect(matchSystem.value).toBe(false);
  });

  it("persists the tone through the shared lib/theme refs, keeping the theme", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();
    const tone = wrapper.get('[data-test="theme-tone"]');
    const system = tone.findAll("button").find((b) => b.text() === "System");
    await system?.trigger("click");
    await flushPromises();
    expect(matchSystem.value).toBe(true);
    expect(theme.value).toBe("safelight-dark");

    const light = tone.findAll("button").find((b) => b.text() === "Light");
    await light?.trigger("click");
    await flushPromises();
    expect(matchSystem.value).toBe(false);
    expect(theme.value).toBe("safelight-light");
  });

  it("lists every web section in the jump nav", () => {
    const wrapper = mount(SettingsPage);
    for (const section of sectionsForSurface("web")) {
      const row = wrapper.get(`[data-test="settings-nav-${section.id}"]`);
      expect(row.text()).toBe(section.label);
    }
    // Saving pictures & clips is a desktop-only section.
    expect(wrapper.find('[data-test="settings-nav-media"]').exists()).toBe(
      false,
    );
  });

  it("narrows the jump nav and the page to the sections that match a search", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();

    await wrapper.get('[data-test="settings-search"]').setValue("civitai");

    expect(wrapper.find('[data-test="settings-nav-accounts"]').exists()).toBe(
      true,
    );
    expect(wrapper.find('[data-test="settings-nav-cloud"]').exists()).toBe(
      false,
    );
    expect(wrapper.find('[data-test="section-accounts"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="section-updates"]').exists()).toBe(false);

    await wrapper.get('[data-test="settings-search"]').setValue("nothing here");
    expect(wrapper.find('[data-test="no-search-results"]').exists()).toBe(true);
  });

  it("lands on the section named by ?section=, and only that one", async () => {
    routeState.query = { section: "library" };
    const wrapper = mount(SettingsPage);
    await flushPromises();

    expect(
      wrapper
        .get('[data-test="settings-nav-library"]')
        .attributes("aria-current"),
    ).toBe("true");
    const shown = wrapper
      .findAll('[data-test^="section-"]')
      .map((el) => el.attributes("data-test"));
    expect(shown).toEqual(["section-library"]);
  });

  it("rewrites ?section= when a pane is picked, replacing rather than pushing", async () => {
    routeState.query = {};
    const wrapper = mount(SettingsPage);
    await flushPromises();
    replaceMock.mockClear();

    await openSection(wrapper, "cloud");
    expect(replaceMock).toHaveBeenLastCalledWith({
      query: { section: "cloud" },
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(
      wrapper
        .findAll('[data-test^="section-"]')
        .map((el) => el.attributes("data-test")),
    ).toEqual(["section-cloud"]);
  });

  it("folds the retired about deep link into Updates & about", async () => {
    routeState.query = { section: "about" };
    const wrapper = mount(SettingsPage);
    await flushPromises();

    expect(
      wrapper
        .get('[data-test="settings-nav-updates"]')
        .attributes("aria-current"),
    ).toBe("true");
  });

  it("keeps the browser-local auto-tag preference beside the engine's own", async () => {
    localStorage.clear();
    reloadAutoTagTitle();
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "library");
    await flushPromises();

    const toggle = wrapper.get('[data-test="config-auto-tag-title"]');
    expect(wrapper.text()).toContain("Tag new prints with their title");
    // Stored in this browser rather than on the machine, so there is nothing
    // to save or reset.
    expect(toggle.attributes("aria-checked")).toBe("true");

    await toggle.trigger("click");
    await flushPromises();
    expect(autoTagTitle.value).toBe(false);
    expect(localStorage.getItem(AUTO_TAG_SETTING_WEB)).toBe("false");
  });

  it("renders hal9000's 150 rows as a page you can read", async () => {
    const rows = hal9000ConfigRows();
    expect(rows).toHaveLength(ENGINE_KEY_SCHEMAS.length + 13 * 8 + 1);
    globalThis.fetch = configRespondingFetch(rows);

    const wrapper = mount(SettingsPage);
    await flushPromises();

    // Only one pane is on screen — the page never stacks its sections.
    expect(wrapper.findAll('[data-test^="section-"]')).toHaveLength(1);

    // (a) "Server-provided configuration key." can only ever mean a key newer
    // than this client — here, the one synthetic unknown, and only in Advanced.
    for (const id of ["generation", "cloud", "performance", "updates"]) {
      await openSection(wrapper, id);
      expect(wrapper.html(), id).not.toContain(
        "Server-provided configuration key.",
      );
    }
    await openSection(wrapper, "advanced");
    expect(
      wrapper.html().split("Server-provided configuration key.").length - 1,
    ).toBe(1);
    expect(wrapper.get('[data-test="section-advanced"]').text()).toContain(
      "future.option",
    );

    // (b) 104 per-style rows are 13 collapsed disclosures, not 104 rows.
    await openSection(wrapper, "styleDefaults");
    expect(wrapper.findAll('[data-test="per-style-name"]')).toHaveLength(13);
    expect(wrapper.get('[data-test="section-styleDefaults"]').text()).toContain(
      "flux-dev:q4",
    );
    expect(
      wrapper.findAll('[data-test="per-style-row-default_steps"]'),
    ).toHaveLength(0);

    // (c) the duplicate GPU card is gone; Machines owns the one device list.
    await openSection(wrapper, "hosts");
    expect(
      wrapper.find('[data-test="settings-device-controls"]').exists(),
    ).toBe(false);
    expect(wrapper.get('[data-test="section-hosts"]').text()).toContain(
      "Open Machines",
    );
  });

  it("expands one style's overrides to its eight fields", async () => {
    globalThis.fetch = configRespondingFetch(hal9000ConfigRows());
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "styleDefaults");
    await flushPromises();

    const disclosure = wrapper.findAll(
      '[data-test="section-styleDefaults"] details',
    )[0];
    expect(disclosure).toBeDefined();
    (disclosure!.element as HTMLDetailsElement).open = true;
    await disclosure!.trigger("toggle");
    await flushPromises();

    for (const field of PER_STYLE_FIELDS) {
      expect(
        wrapper.findAll(`[data-test="per-style-row-${field}"]`).length,
      ).toBe(1);
    }
  });

  it("does not leave the old profile editable when the switched profile cannot load", async () => {
    const rowsFetch = configRespondingFetch([
      { key: "default_steps", value: 20, source: "db" },
    ]);
    globalThis.fetch = vi.fn(async (input, init) =>
      String(input).includes("/api/config/profiles")
        ? ({
            ok: true,
            json: async () => ({
              profiles: ["default", "quality"],
              active: "default",
            }),
          } as Response)
        : rowsFetch(input, init),
    ) as typeof fetch;
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "generation");
    expect(
      wrapper
        .find('[data-test="section-generation"] input[type="number"]')
        .exists(),
    ).toBe(true);

    const original = globalThis.fetch;
    globalThis.fetch = vi.fn(async (input, init) =>
      String(input).endsWith("/api/config") && !init?.method
        ? ({ ok: false, status: 503, json: async () => ({}) } as Response)
        : original(input, init),
    ) as typeof fetch;
    await openSection(wrapper, "profiles");
    await wrapper.get('[data-test="profile-select"]').setValue("quality");
    await flushPromises();

    // The 503 profile's rows are withdrawn, not the previous profile's shown
    // under the new name.
    expect(wrapper.find('[role="alert"]').exists()).toBe(true);
    await openSection(wrapper, "generation");
    expect(
      wrapper
        .find('[data-test="section-generation"] input[type="number"]')
        .exists(),
    ).toBe(false);
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("retains a profile name when creation fails", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "profiles");
    const original = globalThis.fetch;
    globalThis.fetch = vi.fn(async (input, init) =>
      String(input).endsWith("/api/config/profile") && init?.method
        ? ({ ok: false, status: 503, json: async () => ({}) } as Response)
        : original(input, init),
    ) as typeof fetch;
    await wrapper
      .get('[data-test="profile-name"]')
      .setValue("unfinished profile");
    await wrapper.get('[data-test="profile-create"]').trigger("click");
    await flushPromises();
    expect(
      (wrapper.get('[data-test="profile-name"]').element as HTMLInputElement)
        .value,
    ).toBe("unfinished profile");
  });

  it("withdraws the device list when a refresh fails, rather than showing a stale one", async () => {
    let failDevices = false;
    globalThis.fetch = vi.fn(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/devices")) {
        if (failDevices) throw new Error("machine offline");
        return {
          ok: true,
          json: async () => ({ devices: [deviceWire()], plan_version: 1 }),
        } as Response;
      }
      return {
        ok: true,
        json: async () => {
          if (url.endsWith("/profiles"))
            return { profiles: ["default"], active: "default" };
          if (url.endsWith("/api/models")) return [];
          if (url.endsWith("/api/catalog/credentials"))
            return {
              hf: { configured: false, source: null, masked: null },
              civitai: { configured: false, source: null, masked: null },
            };
          if (url.endsWith("/api/capabilities"))
            return {
              devices: { lifecycle: true },
              dispatch: { v2_authoritative: true },
            };
          return { entries: [] };
        },
      } as Response;
    }) as typeof fetch;
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "hosts");
    expect(wrapper.findAll('[data-test="device-card"]')).toHaveLength(1);

    failDevices = true;
    const [, , refresh] = subscribeToDeviceSnapshots.mock.calls[0]!;
    refresh();
    await flushPromises();
    expect(wrapper.findAll('[data-test="device-card"]')).toHaveLength(0);
    wrapper.unmount();
  });

  it("saves a curated engine key through the shared config client", async () => {
    const rows: ConfigRow[] = [
      { key: "default_steps", value: 20, source: "db" },
      { key: "models_dir", value: "/models", source: "file" },
    ];
    const calls: { url: string; init: RequestInit | undefined }[] = [];
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);
      calls.push({ url, init: init ?? undefined });
      const body = url.includes("/api/config/profiles")
        ? { profiles: ["default"], active: "default" }
        : url.endsWith("/api/config")
          ? { profile: "default", entries: rows }
          : url.endsWith("/api/models")
            ? []
            : url.endsWith("/api/catalog/credentials")
              ? {
                  hf: { configured: false, source: null, masked: null },
                  civitai: { configured: false, source: null, masked: null },
                }
              : { entries: [] };
      return { ok: true, json: async () => body } as Response;
    }) as typeof fetch;

    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "generation");
    await flushPromises();

    const steps = wrapper.get(
      '[data-test="section-generation"] input[type="number"]',
    );
    await steps.setValue(28);
    await steps.trigger("change");
    await flushPromises();

    const put = calls.find(
      ({ url, init }) =>
        url.endsWith("/api/config/default_steps") && init?.method === "PUT",
    );
    expect(put?.init?.body).toBe(JSON.stringify({ value: 28 }));
  });

  it("offers a machine picker for licences only when more than one is known", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "licenses");
    await flushPromises();
    expect(wrapper.find('[data-test="licence-machine"]').exists()).toBe(false);
    wrapper.unmount();

    localStorage.setItem(
      HOSTS_STORAGE_KEY,
      JSON.stringify([
        { id: "plato", name: "plato", url: "http://plato.example:7680" },
      ]),
    );
    const withPeer = mount(SettingsPage);
    await flushPromises();
    await openSection(withPeer, "licenses");
    const picker = withPeer.get('[data-test="licence-machine"]');
    expect(picker.findAll("option").map((option) => option.text())).toContain(
      "plato",
    );
    // Acceptance is stored on the machine you pick, so the panel is told
    // which one that is.
    await picker.setValue("plato");
    await flushPromises();
    expect(withPeer.get('[data-test="section-licenses"]').text()).toContain(
      "plato",
    );
  });

  it("controls the origin server GPU from the Machines section", async () => {
    let enabled = true;
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/api/devices") && !init?.method) {
        return {
          ok: true,
          json: async () => ({
            devices: [deviceWire(enabled, enabled ? "enabled" : "disabled")],
            plan_version: enabled ? 1 : 2,
          }),
        } as Response;
      }
      if (url.endsWith("/api/capabilities")) {
        return {
          ok: true,
          json: async () => ({
            devices: {
              available: true,
              lifecycle: true,
              restart_enable: false,
            },
            dispatch: { active_mode: "v2", v2_authoritative: true },
          }),
        } as Response;
      }
      if (url.includes("/api/devices/cuda%3A") && init?.method === "PATCH") {
        enabled = false;
        return {
          ok: true,
          json: async () => deviceWire(false, "disabled"),
        } as Response;
      }
      return {
        ok: true,
        json: async () =>
          url.endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : url.endsWith("/api/models")
              ? []
              : url.endsWith("/api/catalog/credentials")
                ? {
                    hf: { configured: false, source: null, masked: null },
                    civitai: { configured: false, source: null, masked: null },
                  }
                : { entries: [] },
      } as Response;
    });

    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "hosts");
    await flushPromises();
    const machines = wrapper.get('[data-test="section-hosts"]');
    expect(machines.text()).toContain("NVIDIA RTX 3090");
    expect(
      machines
        .get("[data-test='device-panel']")
        .attributes("data-device-count"),
    ).toBe("1");
    // The scheduler lanes belong to the Machines workspace, not here.
    expect(machines.find('[data-test="cpu-utility-lane"]').exists()).toBe(
      false,
    );

    await wrapper.get("[data-test='device-toggle-0']").trigger("click");
    await flushPromises();

    const patch = fetchMock.mock.calls.find(
      ([input, init]) =>
        String(input).includes("/api/devices/cuda%3A") &&
        init?.method === "PATCH",
    );
    expect(patch?.[1]).toEqual(
      expect.objectContaining({
        body: JSON.stringify({ enabled: false }),
      }),
    );
    expect(wrapper.get("[data-test='device-toggle-0']").text()).toBe("Enable");
  });

  it("renders the Settings title and About rows", async () => {
    statusRef.value = {
      version: "9.9.9",
      models_loaded: [],
      busy: false,
      uptime_secs: 1,
    };
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "updates");
    await flushPromises();

    expect(wrapper.get("h1").text()).toBe("Settings");
    expect(wrapper.get('[data-test="about-version"]').text()).toBe("9.9.9");
    expect(wrapper.text()).toContain("this machine + the machines you add");
    expect(wrapper.text()).toContain("Core contributors");
    expect(wrapper.text()).toContain("James Brink");
    expect(wrapper.text()).toContain("Jeffrey Dilley");
    expect(wrapper.text()).not.toMatch(/equal (project )?owners/i);
  });

  it("falls back to an em dash when the server version is unknown", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "updates");
    await flushPromises();
    expect(wrapper.get('[data-test="about-version"]').text()).toBe("—");
  });

  it("refreshes device truth on SSE connect and semantic invalidations", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();
    expect(subscribeToDeviceSnapshots).toHaveBeenCalledTimes(1);
    const [target, _signal, refresh] =
      subscribeToDeviceSnapshots.mock.calls[0]!;
    expect(target).toEqual({
      baseUrl: originHost().url,
      apiKey: originHost().apiKey ?? null,
    });
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const before = fetchMock.mock.calls.filter(([input]) =>
      String(input).endsWith("/api/devices"),
    ).length;
    expect(before).toBe(1);

    refresh();
    await flushPromises();

    expect(
      fetchMock.mock.calls.filter(([input]) =>
        String(input).endsWith("/api/devices"),
      ),
    ).toHaveLength(before + 1);
    wrapper.unmount();
  });

  it("ignores an older same-origin device response after an event refetch", async () => {
    const older = deferred<Response>();
    const newer = deferred<Response>();
    let deviceCall = 0;
    globalThis.fetch = vi.fn(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/devices"))
        return deviceCall++ === 0 ? older.promise : newer.promise;
      return {
        ok: true,
        json: async () => {
          if (url.endsWith("/profiles"))
            return { profiles: ["default"], active: "default" };
          if (url.endsWith("/api/catalog/credentials"))
            return {
              hf: { configured: false, source: null, masked: null },
              civitai: { configured: false, source: null, masked: null },
            };
          if (url.endsWith("/api/capabilities"))
            return {
              devices: { available: true, lifecycle: true },
              dispatch: { active_mode: "v2", v2_authoritative: true },
            };
          return { entries: [], plan: null };
        },
      } as Response;
    }) as typeof fetch;
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "hosts");
    await vi.waitFor(() =>
      expect(subscribeToDeviceSnapshots).toHaveBeenCalled(),
    );
    const refresh = subscribeToDeviceSnapshots.mock.calls[0]?.[2] as () => void;
    refresh();
    newer.resolve({
      ok: true,
      json: async () => ({
        devices: [deviceWire(false, "disabled")],
        plan_version: 2,
      }),
    } as Response);
    await vi.waitFor(() => expect(wrapper.text()).toContain("disabled"));
    older.resolve({
      ok: true,
      json: async () => ({
        devices: [deviceWire(true, "enabled")],
        plan_version: 1,
      }),
    } as Response);
    await flushPromises();

    expect(wrapper.text()).toContain("disabled");
  });

  it("keeps credential status unknown after a failed read and retries inline", async () => {
    const fallback = globalThis.fetch;
    let fail = true;
    globalThis.fetch = vi.fn(async (input, init) => {
      if (String(input).endsWith("/api/catalog/credentials") && fail) {
        return {
          ok: false,
          status: 503,
          text: async () => "temporarily unavailable",
        } as Response;
      }
      return fallback(input, init);
    }) as typeof fetch;
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "accounts");
    await flushPromises();
    expect(wrapper.get('[data-test="credentials-error"]').text()).toContain(
      "Could not load",
    );
    expect(wrapper.find("input[name=hf_token]").exists()).toBe(false);
    fail = false;
    await wrapper.get('[data-test="retry-credentials"]').trigger("click");
    await flushPromises();
    expect(wrapper.find('[data-test="credentials-error"]').exists()).toBe(
      false,
    );
    expect(wrapper.find("input[name=hf_token]").exists()).toBe(true);
    wrapper.unmount();
  });

  it("serializes token changes and preserves other provider drafts", async () => {
    const fallback = globalThis.fetch;
    let finish!: (response: Response) => void;
    let writes = 0;
    globalThis.fetch = vi.fn(async (input, init) => {
      if (
        String(input).endsWith("/api/catalog/credentials/hf") &&
        init?.method === "PUT"
      ) {
        writes++;
        return new Promise<Response>((resolve) => {
          finish = resolve;
        });
      }
      return fallback(input, init);
    }) as typeof fetch;
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "accounts");
    await flushPromises();
    await wrapper.get("input[name=hf_token]").setValue("hf_fixture");
    await wrapper.get("input[name=civitai_token]").setValue("cv_keep_draft");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    expect(writes).toBe(1);
    expect(
      wrapper.get('[data-test="credential-fields"]').attributes("disabled"),
    ).toBeDefined();
    finish({
      ok: true,
      json: async () => ({
        hf: { configured: true, source: "server", masked: "hf_••••ture" },
        civitai: { configured: false, source: null, masked: null },
      }),
    } as Response);
    await flushPromises();
    expect(
      wrapper.get('[data-test="credential-fields"]').attributes("disabled"),
    ).toBeUndefined();
    expect(
      (wrapper.get("input[name=civitai_token]").element as HTMLInputElement)
        .value,
    ).toBe("cv_keep_draft");
    wrapper.unmount();
  });

  it("stores the Hugging Face token on the server and shows a masked state", async () => {
    localStorage.clear();
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      if (
        String(input).endsWith("/api/catalog/credentials/hf") &&
        init?.method === "PUT"
      ) {
        return {
          ok: true,
          json: async () => ({
            hf: {
              configured: true,
              source: "server",
              masked: "hf_••••1234",
            },
            civitai: { configured: false, source: null, masked: null },
          }),
        } as Response;
      }
      return {
        ok: true,
        json: async () =>
          String(input).endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : {
                hf: { configured: false, source: null, masked: null },
                civitai: { configured: false, source: null, masked: null },
              },
      } as Response;
    });
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "accounts");
    await flushPromises();
    await wrapper.get("input[name=hf_token]").setValue("hf_secretvalue1234");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await flushPromises();

    const saveCall = fetchMock.mock.calls.find((c) =>
      String(c[0]).endsWith("/api/catalog/credentials/hf"),
    );
    expect(saveCall?.[1]).toEqual(
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ token: "hf_secretvalue1234" }),
      }),
    );
    expect(localStorage.getItem("mold.web.accounts.v1")).toBeNull();
    expect(wrapper.get('[data-test="hf-mask"]').text()).toBe("hf_••••1234");
    expect(useNotifications().toasts.some((t) => /server/.test(t.text))).toBe(
      true,
    );
  });

  it("reflects an already-saved server token on load and can clear it", async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      const configured = {
        hf: { configured: false, source: null, masked: null },
        civitai: {
          configured: true,
          source: "server",
          masked: "cv_••••7890",
        },
      };
      const cleared = {
        hf: { configured: false, source: null, masked: null },
        civitai: { configured: false, source: null, masked: null },
      };
      return {
        ok: true,
        json: async () =>
          String(input).endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : init?.method === "DELETE"
              ? cleared
              : configured,
      } as Response;
    });
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "accounts");
    await flushPromises();
    expect(wrapper.get('[data-test="civitai-mask"]').text()).toBe(
      "cv_••••7890",
    );
    await wrapper.get('[data-test="clear-civitai"]').trigger("click");
    await flushPromises();
    expect(wrapper.find('[data-test="civitai-mask"]').exists()).toBe(false);
    expect(
      fetchMock.mock.calls.some(
        ([input, init]) =>
          String(input).endsWith("/api/catalog/credentials/civitai") &&
          init?.method === "DELETE",
      ),
    ).toBe(true);
  });

  it("lets a server-saved token override an environment fallback and clear back to it", async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      const environment = {
        hf: {
          configured: true,
          source: "environment",
          masked: "hf_••••base",
        },
        civitai: { configured: false, source: null, masked: null },
      };
      const override = {
        hf: {
          configured: true,
          source: "server",
          masked: "hf_••••ride",
        },
        civitai: { configured: false, source: null, masked: null },
      };
      return {
        ok: true,
        json: async () => {
          if (String(input).endsWith("/profiles"))
            return { profiles: ["default"], active: "default" };
          if (init?.method === "PUT") return override;
          return environment;
        },
      } as Response;
    });

    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "accounts");
    await flushPromises();
    expect(wrapper.get('[data-test="hf-source"]').text()).toBe("Environment");
    await wrapper.get('[data-test="replace-hf"]').trigger("click");
    await wrapper.get("input[name=hf_token]").setValue("hf_user_override");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await flushPromises();
    expect(wrapper.get('[data-test="hf-mask"]').text()).toBe("hf_••••ride");
    expect(wrapper.get('[data-test="hf-source"]').text()).toBe(
      "Saved override",
    );

    await wrapper.get('[data-test="clear-hf"]').trigger("click");
    await flushPromises();
    expect(wrapper.get('[data-test="hf-mask"]').text()).toBe("hf_••••base");
    expect(wrapper.get('[data-test="hf-source"]').text()).toBe("Environment");
  });

  it("keeps the token in the field when the server rejects the save", async () => {
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "accounts");
    await flushPromises();
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "disk full",
    });

    await wrapper.get("input[name=hf_token]").setValue("hf_secretvalue1234");
    await wrapper.get('[data-test="save-hf"]').trigger("click");
    await flushPromises();

    const input = wrapper.get("input[name=hf_token]")
      .element as HTMLInputElement;
    expect(input.value).toBe("hf_secretvalue1234");
    expect(wrapper.find('[data-test="hf-mask"]').exists()).toBe(false);

    const { toasts } = useNotifications();
    expect(toasts.some((t) => t.kind === "success")).toBe(false);
    const error = toasts.find((t) => t.kind === "error");
    expect(error?.text).toContain("disk full");
  });

  it("does not claim a server token was removed when the clear fails", async () => {
    const fetchMock = globalThis.fetch as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input, init) => {
      if (init?.method === "DELETE") {
        return {
          ok: false,
          status: 500,
          text: async () => "read-only filesystem",
        } as Response;
      }
      return {
        ok: true,
        json: async () =>
          String(input).endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : {
                hf: { configured: false, source: null, masked: null },
                civitai: {
                  configured: true,
                  source: "server",
                  masked: "cv_••••7890",
                },
              },
      } as Response;
    });
    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "accounts");
    await flushPromises();

    await wrapper.get('[data-test="clear-civitai"]').trigger("click");
    await flushPromises();

    expect(wrapper.get('[data-test="civitai-mask"]').text()).toBe(
      "cv_••••7890",
    );
    const error = useNotifications().toasts.find((t) => t.kind === "error");
    expect(error?.text).toContain("read-only filesystem");
  });

  it("does not render a fictional default-scheduler control", () => {
    const wrapper = mount(SettingsPage);
    expect(wrapper.find("select[name=default_scheduler]").exists()).toBe(false);
  });

  it("does not render an NSFW visibility control", () => {
    const wrapper = mount(SettingsPage);
    expect(wrapper.find("input[name=catalog_show_nsfw]").exists()).toBe(false);
    expect(wrapper.text()).not.toContain("Show NSFW");
  });

  it("keeps concurrent device mutations busy through out-of-order completion", async () => {
    const first = deviceWire(true, "enabled", 0);
    const second = deviceWire(true, "enabled", 1);
    const firstPatch = deferred<Response>();
    const secondPatch = deferred<Response>();
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);
      if (init?.method === "PATCH") {
        return url.includes(encodeURIComponent(first.id))
          ? firstPatch.promise
          : secondPatch.promise;
      }
      const body = url.endsWith("/api/devices")
        ? { devices: [first, second], plan_version: 1 }
        : url.endsWith("/api/capabilities")
          ? {
              devices: { available: true, lifecycle: true },
              dispatch: { active_mode: "v2", v2_authoritative: true },
            }
          : url.endsWith("/profiles")
            ? { profiles: ["default"], active: "default" }
            : url.endsWith("/api/catalog/credentials")
              ? {
                  hf: { configured: false, source: null, masked: null },
                  civitai: { configured: false, source: null, masked: null },
                }
              : { entries: [], plan: null };
      return { ok: true, json: async () => body } as Response;
    }) as typeof fetch;

    const wrapper = mount(SettingsPage);
    await flushPromises();
    await openSection(wrapper, "hosts");
    await vi.waitFor(() =>
      expect(wrapper.findAll('[data-test="device-card"]')).toHaveLength(2),
    );

    await wrapper.get('[data-test="device-toggle-0"]').trigger("click");
    await wrapper.get('[data-test="device-toggle-1"]').trigger("click");
    expect(
      wrapper.get('[data-test="device-toggle-0"]').attributes("disabled"),
    ).toBeDefined();
    expect(
      wrapper.get('[data-test="device-toggle-1"]').attributes("disabled"),
    ).toBeDefined();

    secondPatch.resolve({
      ok: true,
      json: async () => ({ ...second, desired_enabled: false }),
    } as Response);
    await vi.waitFor(() =>
      expect(
        wrapper.get('[data-test="device-toggle-1"]').attributes("disabled"),
      ).toBeUndefined(),
    );
    expect(
      wrapper.get('[data-test="device-toggle-0"]').attributes("disabled"),
    ).toBeDefined();

    firstPatch.resolve({
      ok: true,
      json: async () => ({ ...first, desired_enabled: false }),
    } as Response);
    await vi.waitFor(() =>
      expect(
        wrapper.get('[data-test="device-toggle-0"]').attributes("disabled"),
      ).toBeUndefined(),
    );
  });
});
