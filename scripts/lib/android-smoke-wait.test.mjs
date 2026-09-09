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
   * (the hardware Back key): nothing is dispatched once the condition holds.
   *
   * Counting the ACTS is what pins it. An earlier version recorded only the
   * ORDER of reads and acts and survived moving the retry ahead of the read —
   * the precise mutation it existed to catch. That order is no longer the
   * protection anyway: the re-read guard below is, and it is what the next
   * test pins.
   */
  test("never dispatches again once the condition holds", async () => {
    const clock = fakeClock();
    let acts = 0;
    let reads = 0;
    await actUntil(
      () => {
        acts += 1;
      },
      () => ++reads >= 5,
      "native Back dismisses settings",
      { ...clock, timeoutMs: 30_000, redispatchEvery: 5 },
    );
    // The opening dispatch and nothing else: the fifth read succeeds, and no
    // retry may be scheduled off a read that came back true.
    expect(acts).toBe(1);
  });

  /*
   * ...and it re-READS immediately before re-dispatching. The poll that
   * scheduled a retry is already one interval old, and an Android Back that
   * arrives after the panel has closed finds nothing to consume and exits the
   * app — passing the assertion, failing three steps later for no reason.
   */
  test("does not dispatch when the condition settled since the last poll", async () => {
    const clock = fakeClock();
    let acts = 0;
    let settled = false;
    await actUntil(
      () => {
        acts += 1;
      },
      () => {
        const answer = settled;
        // Settles in the gap AFTER the poll that will schedule the retry.
        if (acts === 1 && !settled) settled = true;
        return answer;
      },
      "native Back dismisses settings",
      { ...clock, timeoutMs: 30_000, redispatchEvery: 1 },
    );
    expect(acts).toBe(1);
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

  /*
   * The opening dispatch is as retryable as the rest. A CI emulator answered
   * `Input.dispatchTouchEvent` past the 10s CDP timeout and the run died with
   * a bare "CDP timeout" naming no step, because that first dispatch happened
   * outside the loop.
   */
  test("recovers when the very first dispatch throws", async () => {
    const clock = fakeClock();
    let attemptsToAct = 0;
    let landed = false;
    await actUntil(
      () => {
        attemptsToAct += 1;
        if (attemptsToAct === 1)
          throw new Error("CDP timeout: Input.dispatchTouchEvent");
        landed = true;
      },
      () => landed,
      "select hosts",
      { ...clock, timeoutMs: 30_000, redispatchEvery: 5 },
    );
    expect(attemptsToAct).toBe(2);
    expect(landed).toBe(true);
  });

  test("names the failed dispatch in the timeout message", async () => {
    const clock = fakeClock();
    await expect(
      actUntil(
        () => {
          throw new Error("CDP timeout: Input.dispatchTouchEvent");
        },
        () => false,
        "select hosts",
        { ...clock, timeoutMs: 1_000, redispatchEvery: 5 },
      ),
    ).rejects.toThrow(
      /last dispatch failed: CDP timeout: Input\.dispatchTouchEvent/,
    );
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
