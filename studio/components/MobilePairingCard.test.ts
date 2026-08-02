import { afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import MobilePairingCard from "./MobilePairingCard.vue";

const qrPayloads: string[] = [];
vi.mock("qrcode", () => ({
  default: {
    toDataURL: vi.fn(async (value: string) => {
      qrPayloads.push(value);
      return "data:image/png;base64,pairing";
    }),
  },
}));

afterEach(() => {
  qrPayloads.length = 0;
  vi.unstubAllGlobals();
});

describe("MobilePairingCard", () => {
  it("authenticates session creation but keeps the durable key out of the QR", async () => {
    const fetch = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            token: "one-time-token",
            expires_at: Math.floor(Date.now() / 1000) + 120,
            auth_required: true,
            instance_id: "server-id",
            hostname: "studio-mac",
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetch);
    const wrapper = mount(MobilePairingCard, {
      props: {
        target: { baseUrl: "http://127.0.0.1:7680", apiKey: "durable-secret" },
        suggestedBaseUrl: "http://127.0.0.1:7680",
        hostLabel: "This Mac",
      },
    });

    await wrapper.get("button").trigger("click");
    await vi.waitFor(() => expect(qrPayloads).toHaveLength(1));

    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:7680/api/pairing/sessions",
      expect.objectContaining({
        method: "POST",
        headers: expect.any(Headers),
      }),
    );
    const request = fetch.mock.calls[0] as unknown as [string, RequestInit];
    const headers = request[1].headers as Headers;
    expect(headers.get("x-api-key")).toBe("durable-secret");
    expect(qrPayloads[0]).not.toContain("durable-secret");
    const qrUrl = new URL(qrPayloads[0]!);
    expect(qrUrl.protocol).toBe("mold:");
    expect(qrUrl.host).toBe("pair");
    expect(Object.fromEntries(qrUrl.searchParams)).toMatchObject({
      base_url: "http://studio-mac.local:7680",
      token: "one-time-token",
      instance_id: "server-id",
    });
    expect(wrapper.get("img").attributes("src")).toBe(
      "data:image/png;base64,pairing",
    );
    expect(wrapper.text()).toContain("Mobile pairing");
    expect(wrapper.text()).not.toContain("iPhone");
    expect(wrapper.get("img").attributes("alt")).toBe(
      "Mold mobile pairing QR code",
    );
  });
});
