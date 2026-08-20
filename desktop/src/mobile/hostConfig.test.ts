import { beforeEach, describe, expect, it, vi } from "vitest";

const { apiJsonTo } = vi.hoisted(() => ({ apiJsonTo: vi.fn() }));
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJsonTo,
}));

import {
  TRASH_RETENTION_CONFIG_KEY,
  fetchHostConfigKey,
  hostConfigEditable,
  hostConfigLocked,
  retentionDaysFromConfigValue,
  setHostConfigKey,
} from "./hostConfig";

const target = { baseUrl: "http://plato.tailnet.ts.net:7680", apiKey: "keychain-secret" };

beforeEach(() => {
  apiJsonTo.mockReset();
});

describe("mobile host config client", () => {
  it("reads one key from the exact authenticated host", async () => {
    apiJsonTo.mockResolvedValue({ key: TRASH_RETENTION_CONFIG_KEY, value: 30, source: "db" });
    const entry = await fetchHostConfigKey(target, TRASH_RETENTION_CONFIG_KEY);
    expect(entry.value).toBe(30);
    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/config/gallery.trash_retention_days", {
      signal: null,
    });
  });

  it("writes through PUT with a JSON `value` body and never puts the key in the URL", async () => {
    apiJsonTo.mockResolvedValue({ key: TRASH_RETENTION_CONFIG_KEY, value: 7, source: "db" });
    await setHostConfigKey(target, TRASH_RETENTION_CONFIG_KEY, 7);
    const [calledTarget, path, init] = apiJsonTo.mock.calls[0] as [
      typeof target,
      string,
      RequestInit,
    ];
    expect(calledTarget).toBe(target);
    expect(path).toBe("/api/config/gallery.trash_retention_days");
    expect(path).not.toContain("keychain-secret");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(String(init.body))).toEqual({ value: 7 });
  });

  it("parses retention values and recognizes env-pinned keys", () => {
    expect(retentionDaysFromConfigValue(30)).toBe(30);
    expect(retentionDaysFromConfigValue("7")).toBe(7);
    expect(retentionDaysFromConfigValue(null)).toBe(0);
    expect(retentionDaysFromConfigValue("")).toBe(0);
    expect(retentionDaysFromConfigValue("soon")).toBeNull();
    expect(retentionDaysFromConfigValue(-3)).toBeNull();
    expect(hostConfigLocked({ key: "k", value: 1, source: "env", env_var: "X" })).toBe(true);
    expect(hostConfigLocked({ key: "k", value: 1, source: "db" })).toBe(false);
    expect(hostConfigLocked(null)).toBe(false);
  });

  it("treats unknown config authority as read-only, never editable", () => {
    // A failed probe leaves no entry: the control must stay disabled because
    // an env-pinned key would otherwise become editable on a transient error.
    expect(hostConfigEditable(null)).toBe(false);
    expect(hostConfigEditable({ key: "k", value: 1, source: "env", env_var: "X" })).toBe(false);
    expect(hostConfigEditable({ key: "k", value: 1, source: "db" })).toBe(true);
    expect(hostConfigEditable({ key: "k", value: 1, source: "file" })).toBe(true);
    expect(hostConfigEditable({ key: "k", value: 1, source: "default" })).toBe(true);
  });
});
