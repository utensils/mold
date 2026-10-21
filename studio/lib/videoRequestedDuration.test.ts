import { expect, it } from "vitest";
import {
  parseRequestedVideoDuration,
  formatRequestedVideoDuration,
} from "./videoRequestedDuration";

it("keeps requested milliseconds exact rather than snapping to a model frame grid", () => {
  expect(parseRequestedVideoDuration("30")).toBe("30000");
  expect(parseRequestedVideoDuration("4.458")).toBe("4458");
  expect(parseRequestedVideoDuration("0.001")).toBe("1");
  expect(formatRequestedVideoDuration("4458")).toBe("4.458");
});
it("accepts long durations without a product time ceiling", () => {
  expect(parseRequestedVideoDuration("86400")).toBe("86400000");
  expect(parseRequestedVideoDuration("9007199254740.991")).toBe(
    "9007199254740991",
  );
});
it("refuses invalid, ambiguous and unrepresentable values without inventing zero", () => {
  for (const value of [
    "",
    " ",
    "0",
    "-1",
    "NaN",
    "Infinity",
    "1e3",
    "1.0001",
    "18446744073709551.616",
    "1,5",
  ])
    expect(parseRequestedVideoDuration(value)).toBeNull();
});

it("round-trips the full server millisecond range without JavaScript rounding", () => {
  expect(parseRequestedVideoDuration("18446744073709551.615")).toBe(
    "18446744073709551615",
  );
  expect(formatRequestedVideoDuration("18446744073709551615")).toBe(
    "18446744073709551.615",
  );
  expect(parseRequestedVideoDuration("9007199254740.992")).toBe(
    "9007199254740992",
  );
});
