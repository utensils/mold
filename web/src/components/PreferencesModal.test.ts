import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PreferencesModal from "./PreferencesModal.vue";

const originalFetch = globalThis.fetch;

function mountModal() {
  return mount(PreferencesModal, {
    props: { open: true },
    attachTo: document.body,
  });
}

describe("PreferencesModal", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true }) as typeof fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
    document.body.innerHTML = "";
  });

  it("renders the Preferences header", () => {
    const w = mountModal();
    expect(document.body.textContent).toContain("Preferences");
    w.unmount();
  });

  it("POSTs huggingface.token to /api/settings/set on HF token change", async () => {
    const w = mountModal();
    const input = document.body.querySelector(
      "input[name=hf_token]",
    ) as HTMLInputElement;
    expect(input).not.toBeNull();
    input.value = "hf_secret";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await w.vm.$nextTick();
    const fm = globalThis.fetch as ReturnType<typeof vi.fn>;
    expect(fm).toHaveBeenCalledWith(
      "/api/settings/set",
      expect.objectContaining({ method: "POST" }),
    );
    const body = JSON.parse(fm.mock.calls[0][1].body as string);
    expect(body).toEqual({ key: "huggingface.token", value: "hf_secret" });
    w.unmount();
  });

  it("POSTs civitai.token on Civitai token change", async () => {
    const w = mountModal();
    const input = document.body.querySelector(
      "input[name=civitai_token]",
    ) as HTMLInputElement;
    input.value = "cv_secret";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await w.vm.$nextTick();
    const fm = globalThis.fetch as ReturnType<typeof vi.fn>;
    const body = JSON.parse(fm.mock.calls[0][1].body as string);
    expect(body).toEqual({ key: "civitai.token", value: "cv_secret" });
    w.unmount();
  });

  it("does not render an NSFW visibility control", () => {
    const w = mountModal();
    expect(
      document.body.querySelector("input[name=catalog_show_nsfw]"),
    ).toBeNull();
    expect(document.body.textContent).not.toContain("Show NSFW models");
    w.unmount();
  });

  it("POSTs generate.scheduler when default scheduler changes", async () => {
    const w = mountModal();
    const sel = document.body.querySelector(
      "select[name=default_scheduler]",
    ) as HTMLSelectElement;
    expect(sel).not.toBeNull();
    sel.value = "ddim";
    sel.dispatchEvent(new Event("input", { bubbles: true }));
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    await w.vm.$nextTick();
    const fm = globalThis.fetch as ReturnType<typeof vi.fn>;
    const body = JSON.parse(fm.mock.calls[0][1].body as string);
    expect(body).toEqual({ key: "generate.scheduler", value: "ddim" });
    w.unmount();
  });

  it("does not render token inputs when closed", () => {
    const w = mount(PreferencesModal, {
      props: { open: false },
      attachTo: document.body,
    });
    expect(document.body.querySelector("input[name=hf_token]")).toBeNull();
    w.unmount();
  });
});
