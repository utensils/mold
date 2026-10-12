import { beforeEach, describe, expect, it } from "vitest";
import {
  directOrigin,
  connectionModeFor,
  rememberConnectionMode,
  CONNECTION_MODE_HEADER,
} from "./connectionMode";
import { apiHeaders } from "./client";
beforeEach(() => localStorage.clear());
describe("explicit direct connection selection", () => {
  it.each([
    "http://studio.local:7680",
    "http://100.64.0.7:7680",
    "http://203.0.113.4:7680",
    "https://render.example",
  ])("accepts the direct origin %s", (url) =>
    expect(directOrigin(url)).toBe(url),
  );
  it.each([
    "https://host.mold-link.urandom.io",
    "https://MOLD-LINK.URANDOM.IO.",
    "https://user:secret@host.test",
    "https://host.test/api/status",
    "https://host.test?key=x",
    "wss://host.test",
  ])("refuses %s", (url) => expect(() => directOrigin(url)).toThrow());
  it("does not reject a lookalike domain or infer regular mode from missing pairing", () => {
    expect(directOrigin("https://mold-link.urandom.io.example")).toBe(
      "https://mold-link.urandom.io.example",
    );
    expect(connectionModeFor("http://host.test")).toBe("secure");
  });
  it("persists only an explicit selection and lets a frozen target override it", () => {
    rememberConnectionMode("http://host.test", "regular");
    expect(connectionModeFor("http://host.test")).toBe("regular");
    expect(
      apiHeaders({
        baseUrl: "http://host.test",
        apiKey: null,
        connectionMode: "secure",
      }).get(CONNECTION_MODE_HEADER),
    ).toBe("secure");
    rememberConnectionMode("http://host.test", "secure");
    expect(connectionModeFor("http://host.test")).toBe("secure");
  });
});

import originFixtures from "./fixtures/direct-origins-v1.json";
describe("shared direct origin contract", () => {
  it.each(originFixtures.accepted)(
    "canonicalizes $input",
    ({ input, origin }) => expect(directOrigin(input)).toBe(origin),
  );
  it.each(originFixtures.rejected)("refuses %s", (input) =>
    expect(() => directOrigin(input)).toThrow(),
  );
});
