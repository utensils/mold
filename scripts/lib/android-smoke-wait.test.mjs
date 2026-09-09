import { describe, expect, test } from "bun:test";
import {
  DEFAULT_SMOKE_TIMEOUT_MS,
  actUntil,
  resolveDeadlineMs,
  until,
} from "./android-smoke-wait.mjs";

describe("resolveDeadlineMs", () => {
  test("an absent or blank value falls back", () => {
    for (const raw of [undefined, null, "", "   "])
      expect(resolveDeadlineMs(raw)).toBe(DEFAULT_SMOKE_TIMEOUT_MS);
  });

  test("an empty variable never becomes a zero deadline", () => {
    // `Number(process.env.X ?? 30_000)` answers 0 here, which runs every wait
    // zero times and fails the whole suite in milliseconds.
    expect(resolveDeadlineMs("")).toBeGreaterThan(0);
  });

  test("a usable override is taken", () => {
    expect(resolveDeadlineMs("45000")).toBe(45_000);
    expect(resolveDeadlineMs(45_000)).toBe(45_000);
  });

  test("a present but unusable value is refused by name", () => {
    for (const raw of ["abc", "0", "-5", "NaN"])
      expect(() => resolveDeadlineMs(raw)).toThrow(
        /MOLD_ANDROID_SMOKE_TIMEOUT_MS/,
      );
  });
});

/** A clock the test moves by hand, so no test waits on real time. */
function fakeClock() {
  let t = 0;
  return {
    now: () => t,
    sleep: async (ms) => {
      t += ms;
    },
  };
}

describe("until", () => {
  test("returns the first truthy value without waiting out the deadline", async () => {
    const clock = fakeClock();
    let reads = 0;
    const value = await until(
      () => (++reads >= 3 ? "ready" : false),
      "something",
      { ...clock, timeoutMs: 30_000 },
    );
    expect(value).toBe("ready");
    expect(reads).toBe(3);
  });

  test("names the label, the elapsed time and the attempt count", async () => {
    const clock = fakeClock();
    await expect(
      until(() => false, "select hosts", { ...clock, timeoutMs: 1_000 }),
    ).rejects.toThrow(/Timed out: select hosts \(1s, 5 attempts\)/);
  });

  test("keeps the last read error as the cause", async () => {
    const clock = fakeClock();
    const failure = new Error("CDP timeout: Runtime.evaluate");
    const error = await until(
      () => {
        throw failure;
      },
      "reading",
      { ...clock, timeoutMs: 400 },
    ).catch((e) => e);
    expect(error.cause).toBe(failure);
  });
});

describe("actUntil", () => {
  test("dispatches once when the action lands", async () => {
    const clock = fakeClock();
    let taps = 0;
    let landed = false;
    await actUntil(
      () => {
        taps += 1;
        landed = true;
      },
      () => landed,
      "select hosts",
      { ...clock, timeoutMs: 30_000 },
    );
    expect(taps).toBe(1);
  });

  test("re-dispatches a dropped action until it takes effect", async () => {
    const clock = fakeClock();
    // The emulator swallows the first two taps, exactly as the CI evidence
    // shows: the app stays on the previous tab while the wait polls a screen
    // no tap ever reached.
    let taps = 0;
    let landed = false;
    await actUntil(
      () => {
        taps += 1;
        if (taps > 2) landed = true;
      },
      () => landed,
      "select hosts",
      { ...clock, timeoutMs: 30_000, redispatchEvery: 5 },
    );
    expect(taps).toBe(3);
    expect(landed).toBe(true);
  });

  test("a retry that throws does not abandon the wait", async () => {
    const clock = fakeClock();
    let taps = 0;
    let landed = false;
    await actUntil(
      () => {
        taps += 1;
        if (taps === 2) throw new Error("dispatch refused");
        if (taps > 2) landed = true;
      },
      () => landed,
      "select hosts",
      { ...clock, timeoutMs: 30_000, redispatchEvery: 5 },
    );
    expect(landed).toBe(true);
  });

  /*
   * The property that makes it safe to retry an action that is NOT idempotent
   * (the hardware Back key): a re-dispatch only ever follows a poll that has
   * just read the condition as false, and nothing is dispatched once it holds.
   * A blind timer would press Back again after the panel had already closed,
   * popping the navigation underneath it.
   */
  test("never dispatches again once the condition holds", async () => {
    const clock = fakeClock();
    const order = [];
    let landed = false;
    await actUntil(
      () => {
        order.push("act");
        // The action lands on the second dispatch.
        if (order.filter((o) => o === "act").length >= 2) landed = true;
      },
      () => {
        order.push("read");
        return landed;
      },
      "native Back dismisses settings",
      { ...clock, timeoutMs: 30_000, redispatchEvery: 5 },
    );
    // Every dispatch after the first is immediately preceded by a false read,
    // and none follows the read that returned true.
    expect(order[0]).toBe("act");
    expect(order.at(-1)).toBe("read");
    for (let i = 1; i < order.length; i++)
      if (order[i] === "act") expect(order[i - 1]).toBe("read");
    expect(order.filter((o) => o === "act")).toHaveLength(2);
  });

  test("a slower cadence dispatches less often over the same wait", async () => {
    const counts = [];
    for (const redispatchEvery of [5, 15]) {
      const clock = fakeClock();
      let taps = 0;
      await actUntil(
        () => {
          taps += 1;
        },
        () => false,
        "back",
        { ...clock, timeoutMs: 30_000, redispatchEvery },
      ).catch(() => {});
      counts.push(taps);
    }
    expect(counts[1]).toBeLessThan(counts[0]);
  });

  test("still fails when the action never takes effect", async () => {
    const clock = fakeClock();
    let taps = 0;
    await expect(
      actUntil(
        () => {
          taps += 1;
        },
        () => false,
        "select hosts",
        { ...clock, timeoutMs: 1_000, redispatchEvery: 5 },
      ),
    ).rejects.toThrow(/Timed out: select hosts/);
    // One opening dispatch, and one more each time the poll count comes round.
    expect(taps).toBe(2);
  });
});
