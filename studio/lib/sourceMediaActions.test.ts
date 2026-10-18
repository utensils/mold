import { expect, it } from "vitest";
import {
  sourceAddress,
  sourceCommand,
  sourceFileName,
} from "./sourceMediaActions";
import type { SourceMediaSelection } from "./sourceMediaBrowser";
const selection: SourceMediaSelection = {
  owner: {
    key: "owner",
    title: "Print",
    filename: "it's a print.png",
    hostLabel: "Workstation",
    target: { baseUrl: "https://host", apiKey: "private-key" },
  },
  member: {
    member_id: "abcdef0123456789",
    role: "source_image",
    display_name: "input.png",
    size_bytes: 12,
  },
};
it("uses exact owning host, complete member ID, and safe shell quoting without secrets", () => {
  expect(sourceAddress(selection)).toBe(
    "source:it's a print.png#abcdef0123456789",
  );
  const command = sourceCommand(selection);
  expect(command).toContain("--host 'https://host'");
  expect(command).toContain("'it'\\''s a print.png'");
  expect(command).not.toContain("private-key");
});
it("never copies credentials from an URL or constructs a traversal download name", () => {
  expect(() =>
    sourceCommand({
      ...selection,
      owner: {
        ...selection.owner,
        target: {
          baseUrl: "https://user:password@host/path?secret=key",
          apiKey: null,
        },
      },
    }),
  ).toThrow("machine URL");
  expect(
    sourceFileName({
      ...selection,
      member: { ...selection.member, display_name: "../../bad.png" },
    }),
  ).toBe("bad.png");
});
