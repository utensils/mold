import { afterEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import PairingAccessPanel from "./PairingAccessPanel.vue";

vi.mock("qrcode", () => ({
  default: { toDataURL: vi.fn(async () => "data:image/png;base64,pairing") },
}));

afterEach(() => vi.unstubAllGlobals());

describe("PairingAccessPanel", () => {
  it("lists grants and requires a second click before revoking one", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            auth_required: true,
            pairing_available: true,
            clients: [
              {
                id: "pair_phone",
                name: "Mold on iPhone",
                client_kind: "iphone",
                created_at_ms: 100,
                last_used_at_ms: 200,
              },
            ],
          }),
          { headers: { "content-type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetch);
    const wrapper = mount(PairingAccessPanel, {
      props: {
        target: { baseUrl: "http://studio:7680", apiKey: "operator-key" },
        suggestedBaseUrl: "http://studio:7680",
      },
    });
    await flushPromises();

    expect(wrapper.text()).toContain("Mold on iPhone");
    const revoke = wrapper.get("[data-test='revoke-paired-pair_phone']");
    await revoke.trigger("click");
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(revoke.text()).toContain("Revoke Mold on iPhone?");
    await revoke.trigger("click");
    await flushPromises();

    expect(fetch).toHaveBeenLastCalledWith(
      "http://studio:7680/api/pairing/clients/pair_phone",
      expect.objectContaining({ method: "DELETE" }),
    );
    expect(wrapper.text()).not.toContain("Mold on iPhone");
  });

  it("explains when the host does not require access credentials", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              auth_required: false,
              pairing_available: true,
              clients: [],
            }),
            { headers: { "content-type": "application/json" } },
          ),
      ),
    );
    const wrapper = mount(PairingAccessPanel, {
      props: {
        target: { baseUrl: "http://studio:7680", apiKey: null },
        suggestedBaseUrl: "http://studio:7680",
      },
    });
    await flushPromises();
    expect(wrapper.text()).toContain("Access control is off");
  });

  it("discards a response from the previous target", async () => {
    let resolveA!: (response: Response) => void;
    let resolveB!: (response: Response) => void;
    const fetch = vi.fn((url: string | URL | Request) => {
      const pending = new Promise<Response>((resolve) => {
        if (String(url).startsWith("http://server-a")) resolveA = resolve;
        else resolveB = resolve;
      });
      return pending;
    });
    vi.stubGlobal("fetch", fetch);
    const wrapper = mount(PairingAccessPanel, {
      props: {
        target: { baseUrl: "http://server-a", apiKey: "operator-key" },
        suggestedBaseUrl: "http://server-a",
      },
    });
    await wrapper.setProps({
      target: { baseUrl: "http://server-b", apiKey: "operator-key" },
      suggestedBaseUrl: "http://server-b",
    });
    resolveB(
      Response.json({
        auth_required: true,
        pairing_available: true,
        clients: [
          {
            id: "pair_b",
            name: "Server B iPhone",
            client_kind: "iphone",
            created_at_ms: 100,
            last_used_at_ms: null,
          },
        ],
      }),
    );
    await flushPromises();
    resolveA(
      Response.json({
        auth_required: true,
        pairing_available: true,
        clients: [
          {
            id: "pair_a",
            name: "Server A iPhone",
            client_kind: "iphone",
            created_at_ms: 100,
            last_used_at_ms: null,
          },
        ],
      }),
    );
    await flushPromises();

    expect(wrapper.text()).toContain("Server B iPhone");
    expect(wrapper.text()).not.toContain("Server A iPhone");
  });

  it("does not restore a revoked client from an older refresh", async () => {
    const client = {
      id: "pair_phone",
      name: "Mold on iPhone",
      client_kind: "iphone",
      created_at_ms: 100,
      last_used_at_ms: null,
    };
    let resolveRefresh!: (response: Response) => void;
    let getCount = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn((_: string | URL | Request, init?: RequestInit) => {
        if (init?.method === "DELETE") {
          return Promise.resolve(new Response(null, { status: 204 }));
        }
        getCount += 1;
        if (getCount === 1) {
          return Promise.resolve(
            Response.json({
              auth_required: true,
              pairing_available: true,
              clients: [client],
            }),
          );
        }
        return new Promise<Response>((resolve) => {
          resolveRefresh = resolve;
        });
      }),
    );
    const wrapper = mount(PairingAccessPanel, {
      props: {
        target: { baseUrl: "http://studio:7680", apiKey: "operator-key" },
        suggestedBaseUrl: "http://studio:7680",
      },
    });
    await flushPromises();
    await wrapper.get("[data-test='paired-access-refresh']").trigger("click");
    const revoke = wrapper.get("[data-test='revoke-paired-pair_phone']");
    await revoke.trigger("click");
    await revoke.trigger("click");
    await flushPromises();
    resolveRefresh(
      Response.json({
        auth_required: true,
        pairing_available: true,
        clients: [client],
      }),
    );
    await flushPromises();

    expect(wrapper.text()).not.toContain("Mold on iPhone");
  });
});
