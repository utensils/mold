import { afterEach, describe, expect, it, vi } from "vitest";
import type { ApiTarget } from "./client";
import {
  canResetConfig,
  isRowLocked,
  listConfig,
  listProfiles,
  provenance,
  resetConfig,
  setConfig,
  switchProfile,
} from "./config";

const target: ApiTarget = { baseUrl: "http://plato:7680", apiKey: "secret" };

interface Captured {
  url: string;
  method: string;
  headers: Headers;
  body: unknown;
}

function stub(respond: () => Response): () => Captured {
  let captured: Captured | null = null;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      captured = {
        url: String(input),
        method: init?.method ?? "GET",
        headers: new Headers(init?.headers),
        body:
          typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
      };
      return respond();
    }),
  );
  return () => {
    if (!captured) throw new Error("fetch was not called");
    return captured;
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("listConfig", () => {
  const rows = [{ key: "expand.enabled", value: true, source: "db" }];

  /* Three shapes have shipped on this endpoint. A client that understood only
   * the newest would show an empty Settings page against an older machine. */
  it("reads a bare array, `{ profile, entries }`, and `{ rows }` alike", async () => {
    for (const body of [
      rows,
      { profile: "default", entries: rows },
      { rows },
    ]) {
      stub(() => Response.json(body));
      await expect(listConfig(target)).resolves.toEqual(rows);
    }
  });

  it("reads a body carrying none of them as no rows, never as a failure", async () => {
    stub(() => Response.json({ profile: "default" }));
    await expect(listConfig(target)).resolves.toEqual([]);
  });

  it("asks the target it was handed, with its own key", async () => {
    const captured = stub(() => Response.json(rows));
    await listConfig(target);
    expect(captured().url).toBe("http://plato:7680/api/config");
    expect(captured().headers.get("x-api-key")).toBe("secret");
  });

  it("raises the host's own error rather than an empty list", async () => {
    stub(() => Response.json({ error: "no metadata DB" }, { status: 503 }));
    await expect(listConfig(target)).rejects.toThrow("no metadata DB");
  });
});

describe("setConfig", () => {
  it("PUTs the value as JSON under the escaped key", async () => {
    const captured = stub(() => new Response(null, { status: 204 }));
    await setConfig(target, "models.sd1.5.lora", "anime");
    const call = captured();
    expect(call.url).toBe("http://plato:7680/api/config/models.sd1.5.lora");
    expect(call.method).toBe("PUT");
    expect(call.headers.get("content-type")).toBe("application/json");
    expect(call.body).toEqual({ value: "anime" });
  });

  it("sends a cleared value as null, not as an empty string", async () => {
    const captured = stub(() => new Response(null, { status: 204 }));
    await setConfig(target, "default_negative_prompt", null);
    expect(captured().body).toEqual({ value: null });
  });
});

describe("resetConfig", () => {
  it("DELETEs the key", async () => {
    const captured = stub(() => new Response(null, { status: 204 }));
    await resetConfig(target, "expand.top_p");
    expect(captured().url).toBe("http://plato:7680/api/config/expand.top_p");
    expect(captured().method).toBe("DELETE");
  });

  it("surfaces the host's refusal of a file-backed key", async () => {
    stub(() =>
      Response.json(
        { error: "'models_dir' is stored in config.toml" },
        { status: 422 },
      ),
    );
    await expect(resetConfig(target, "models_dir")).rejects.toThrow(
      "config.toml",
    );
  });
});

describe("profiles", () => {
  it("defaults a partial listing rather than failing", async () => {
    stub(() => Response.json({}));
    await expect(listProfiles(target)).resolves.toEqual({
      profiles: [],
      active: "default",
    });
  });

  it("reads the listing a current host sends", async () => {
    stub(() =>
      Response.json({ profiles: ["default", "film"], active: "film" }),
    );
    await expect(listProfiles(target)).resolves.toEqual({
      profiles: ["default", "film"],
      active: "film",
    });
  });

  it("switches by name", async () => {
    const captured = stub(() => new Response(null, { status: 204 }));
    await switchProfile(target, "film");
    const call = captured();
    expect(call.url).toBe("http://plato:7680/api/config/profile");
    expect(call.method).toBe("PUT");
    expect(call.body).toEqual({ name: "film" });
  });
});

describe("provenance", () => {
  it("tags every source the endpoint reports", () => {
    expect(provenance("db")).toEqual({ glyph: "⌂", label: "db" });
    expect(provenance("file")).toEqual({ glyph: "⛁", label: "file" });
    expect(provenance("env")).toEqual({ glyph: "⚿", label: "env" });
    expect(provenance("default")).toEqual({ glyph: "·", label: "default" });
  });

  it("locks a row the environment answers for", () => {
    expect(isRowLocked({ key: "k", value: 1, source: "env" })).toBe(true);
    expect(isRowLocked({ key: "k", value: 1, source: "db" })).toBe(false);
  });
});

describe("canResetConfig", () => {
  /* DELETE resets only DB-surface keys; this mirrors
   * `mold_core::config_keys::effective_surface`, and a row it answers `true`
   * for wrongly shows a ↺ the host will refuse. */
  it("resets the DB-surface prefixes", () => {
    for (const key of [
      "tui.theme",
      "expand.top_p",
      "generate.auto_tag_title",
      "gallery.trash_retention_days",
      "scheduler.warm_wait_max_ms",
      "queue.held_retention_days",
      "model_prefs.anything",
    ]) {
      expect(canResetConfig(key), key).toBe(true);
    }
  });

  it("resets the flat generation defaults that moved to the DB", () => {
    for (const key of [
      "default_width",
      "default_height",
      "default_steps",
      "embed_metadata",
      "default_negative_prompt",
      "t5_variant",
      "qwen3_variant",
    ]) {
      expect(canResetConfig(key), key).toBe(true);
    }
  });

  it("refuses every key that lives in config.toml", () => {
    for (const key of [
      "models_dir",
      "output_dir",
      "server_port",
      "default_model",
      "umt5_variant",
      "logging.level",
      "logging.dir",
      "runpod.api_key",
      "lambda.endpoint",
    ]) {
      expect(canResetConfig(key), key).toBe(false);
    }
  });

  it("splits a per-style key on its last dot, the way the engine does", () => {
    expect(canResetConfig("models.sd1.5.lora_scale")).toBe(true);
    expect(canResetConfig("models.flux-dev:q4.default_steps")).toBe(true);
    // Component paths stay in config.toml even for a style whose prefs do not.
    expect(canResetConfig("models.flux-dev:q4.transformer")).toBe(false);
  });
});
