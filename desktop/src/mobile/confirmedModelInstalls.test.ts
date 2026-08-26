import { describe, expect, it } from "vitest";
import type { ModelEntry } from "../lib/api/types";
import type { HostRoute } from "../stores/hosts";
import type { MobileHost } from "./hosts";
import {
  accountForConfirmedInventory,
  confirmedModelHostIds,
  confirmModelInstall,
  CONFIRMED_MODEL_INSTALL_TTL_MS,
  retireConfirmedModelAuthority,
} from "./confirmedModelInstalls";

const route: HostRoute = {
  hostId: "render",
  label: "Render",
  kind: "remote",
  target: { baseUrl: "http://render:7680", apiKey: "secret" },
  instanceId: "instance-a",
};
const host = {
  id: "render",
  name: "Render",
  baseUrl: route.target.baseUrl,
  apiKey: route.target.apiKey,
  instanceId: route.instanceId,
} as MobileHost;
const matches = (candidate: HostRoute, current: MobileHost) =>
  candidate.target.baseUrl === current.baseUrl && candidate.instanceId === current.instanceId;

describe("confirmed model installs", () => {
  it("survives an empty stale snapshot until inventory accounts for the model", () => {
    const confirmedAt = 10_000;
    const confirmed = confirmModelInstall({}, route, "ltx2:q8", confirmedAt);
    const stale = accountForConfirmedInventory(confirmed, host.id, []);
    expect(confirmedModelHostIds(stale, "ltx2:q8", [host], matches, confirmedAt)).toEqual([
      host.id,
    ]);

    expect(
      confirmedModelHostIds(
        stale,
        "ltx2:q8",
        [host],
        matches,
        confirmedAt + CONFIRMED_MODEL_INSTALL_TTL_MS,
      ),
    ).toEqual([]);

    const accounted = accountForConfirmedInventory(stale, host.id, [
      { name: "ltx2:q8", downloaded: true } as ModelEntry,
    ]);
    expect(confirmedModelHostIds(accounted, "ltx2:q8", [host], matches, confirmedAt)).toEqual([]);
  });

  it("does not transfer a claim to a replacement instance or retired authority", () => {
    const confirmed = confirmModelInstall({}, route, "ltx2:q8");
    expect(
      confirmedModelHostIds(confirmed, "ltx2:q8", [{ ...host, instanceId: "instance-b" }], matches),
    ).toEqual([]);
    expect(
      confirmedModelHostIds(
        retireConfirmedModelAuthority(confirmed, host.id),
        "ltx2:q8",
        [host],
        matches,
      ),
    ).toEqual([]);
  });

  it("does not resurrect an expired model when another model is confirmed", () => {
    const confirmedAt = 20_000;
    const first = confirmModelInstall({}, route, "model-a", confirmedAt);
    const second = confirmModelInstall(
      first,
      route,
      "model-b",
      confirmedAt + CONFIRMED_MODEL_INSTALL_TTL_MS,
    );

    expect(
      confirmedModelHostIds(
        second,
        "model-a",
        [host],
        matches,
        confirmedAt + CONFIRMED_MODEL_INSTALL_TTL_MS,
      ),
    ).toEqual([]);
    expect(
      confirmedModelHostIds(
        second,
        "model-b",
        [host],
        matches,
        confirmedAt + CONFIRMED_MODEL_INSTALL_TTL_MS,
      ),
    ).toEqual([host.id]);
  });
});
