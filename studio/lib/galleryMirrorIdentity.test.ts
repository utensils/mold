import { expect, it } from "vitest";
import { mirrorMetadataMatches } from "./galleryMirrorIdentity";
it("keeps exact u64 and accepts only documented archive omissions", () => {
  const source =
    '{"seed":18446744073709551615,"version":"1.2.3 (abcdef0 2026-10-18)","job_id":"j","generation_time_ms":12}';
  expect(
    mirrorMetadataMatches(
      source,
      '{"version":"1.2.3","seed":18446744073709551615,"has_alpha":false}',
    ),
  ).toBe(true);
  expect(
    mirrorMetadataMatches(
      source,
      '{"version":"1.2.3","seed":18446744073709551614}',
    ),
  ).toBe(false);
  expect(
    mirrorMetadataMatches('{"has_alpha":true}', '{"has_alpha":false}'),
  ).toBe(false);
  expect(mirrorMetadataMatches('{"future":1}', '{"future":2}')).toBe(false);
});
it("requires exact embedded corroboration for omitted scheduler and transparency", () => {
  expect(
    mirrorMetadataMatches('{"seed":1}', '{"seed":1,"scheduler":"euler"}'),
  ).toBe(false);
  expect(
    mirrorMetadataMatches(
      '{"seed":1}',
      '{"seed":1,"scheduler":"euler"}',
      '{"seed":1,"scheduler":"euler"}',
    ),
  ).toBe(true);
  expect(
    mirrorMetadataMatches(
      '{"seed":1}',
      '{"seed":1,"scheduler":"euler"}',
      '{"seed":1,"scheduler":"other"}',
    ),
  ).toBe(false);
});

it("compares equivalent small JSON float spellings without rounding integer seeds", () => {
  expect(
    mirrorMetadataMatches(
      '{"guidance":7,"seed":18446744073709551615}',
      '{"guidance":7.0,"seed":18446744073709551615}',
    ),
  ).toBe(true);
});
