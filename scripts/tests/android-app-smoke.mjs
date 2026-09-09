// Run with Bun against an installed DEBUG APK on an emulator. No server or
// generation is needed. CDP controls the real Tauri WebView, not a browser copy.
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { runLegacyMediaSmoke } from "./android-legacy-media-smoke.mjs";

const adb = process.env.ADB ?? "adb";
const serial = process.env.ANDROID_SERIAL ?? "emulator-5554";
const output =
  process.env.MOLD_ANDROID_EVIDENCE ?? "/tmp/mold-android-app-smoke";
const run = (...args) =>
  execFileSync(adb, ["-s", serial, ...args], { encoding: "utf8" }).trim();
const shell = (...args) => run("shell", ...args);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
/**
 * How long a single wait may take.
 *
 * This polls every 200ms and returns the instant its condition holds, so a
 * generous deadline costs a healthy run NOTHING — it only lengthens the time
 * to fail. On a CI runner the emulator renders in software (`lavapipe`) and
 * `adb` intermittently exits 1, which this loop swallows into `lastError` and
 * retries; 30s was not enough window for adb to recover, and the late steps of
 * a long interaction sequence — the last tab of five, the Back that follows
 * it — were the ones that ran out. Override for a slower machine still.
 */
const UNTIL_TIMEOUT_MS = Number(
  process.env.MOLD_ANDROID_SMOKE_TIMEOUT_MS ?? 90_000,
);

async function until(read, label) {
  const started = Date.now();
  const deadline = started + UNTIL_TIMEOUT_MS;
  let lastError;
  let attempts = 0;
  while (Date.now() < deadline) {
    attempts += 1;
    try {
      const value = await read();
      if (value) return value;
    } catch (error) {
      lastError = error;
    }
    await sleep(200);
  }
  // Say what was waited on and for how long: "Timed out: select hosts" alone
  // could not distinguish a slow emulator from a condition that never holds.
  throw new Error(
    `Timed out: ${label} (${Math.round((Date.now() - started) / 1000)}s, ` +
      `${attempts} attempts)`,
    { cause: lastError },
  );
}

assert(
  shell("getprop", "ro.kernel.qemu") === "1",
  "This test requires an emulator",
);
mkdirSync(output, { recursive: true });
const api = shell("getprop", "ro.build.version.sdk");
if (api === "28") {
  shell("am", "force-stop", "com.utensils.mold");
  shell(
    "pm",
    "revoke",
    "com.utensils.mold",
    "android.permission.WRITE_EXTERNAL_STORAGE",
  );
  shell(
    "pm",
    "revoke",
    "com.utensils.mold",
    "android.permission.READ_EXTERNAL_STORAGE",
  );
}
const originalScale = shell("settings", "get", "system", "font_scale");
let socket;
let originalTab;
let evaluate;
let port;
const navigationEvidence = [];
async function recordNavigation(label) {
  if (!evaluate) return;
  navigationEvidence.push({
    label,
    at: new Date().toISOString(),
    state: await evaluate(`({
      settingsOpen: !!document.querySelector('.is-settings-open'),
      tab: document.querySelector('.mobile-tab[aria-current=page]')?.dataset.test,
      transient: history.state?.['mold.mobile.transient'] ?? null,
      historyLength: history.length,
      activeTag: document.activeElement?.tagName,
      activeControl: document.activeElement?.getAttribute('data-test'),
      userActivated: navigator.userActivation?.hasBeenActive,
      viewportHeight: visualViewport?.height,
      windowHeight: innerHeight,
      visibility: document.visibilityState
    })`),
  });
  writeFileSync(
    output + "/navigation.json",
    JSON.stringify(navigationEvidence, null, 2),
  );
}
try {
  // The runner unlocks immediately after sys.boot_completed; API35 Quickstep
  // can time out before it owns a focused window and display its ANR over Mold.
  // Recover only that recorded launcher boot failure, before starting the app.
  // Never dismiss app errors or retry app assertions after the test begins.
  const bootAnr = shell("dumpsys", "activity", "lastanr");
  const firstAnrTask = bootAnr.match(/\* Task\{[^\n]+/)?.[0] ?? "";
  if (
    firstAnrTask.includes("type=home") &&
    firstAnrTask.includes("I=com.android.launcher3/") &&
    /^\s*Reason: Input dispatching timed out \(Application does not have a focused window\)\.\s*$/m.test(
      bootAnr,
    ) &&
    /^  ResumedActivity: ActivityRecord\{[^\n]* com\.android\.launcher3\//m.test(
      bootAnr,
    ) &&
    !bootAnr.includes("com.utensils.mold")
  ) {
    writeFileSync(output + "/boot-launcher-anr.txt", bootAnr);
    // Pull the PNG as a file: full-resolution screenshots can exceed the
    // child-process stdout buffer before the app smoke test even starts.
    shell("screencap", "-p", "/sdcard/mold-boot-launcher-anr.png");
    run(
      "pull",
      "/sdcard/mold-boot-launcher-anr.png",
      output + "/boot-launcher-anr.png",
    );
    shell("rm", "-f", "/sdcard/mold-boot-launcher-anr.png");
    shell("am", "force-stop", "com.android.launcher3");
  }
  shell("am", "start", "-n", "com.utensils.mold/.MainActivity");
  const pid = await until(
    () => shell("pidof", "com.utensils.mold"),
    "app process",
  );
  await until(
    () =>
      shell("cat", "/proc/net/unix").includes("webview_devtools_remote_" + pid),
    "debug WebView",
  );
  port = run(
    "forward",
    "tcp:0",
    "localabstract:webview_devtools_remote_" + pid,
  );
  const page = await until(async () => {
    const pages = await (
      await fetch("http://127.0.0.1:" + port + "/json/list", {
        signal: AbortSignal.timeout(5000),
      })
    ).json();
    return pages.find(
      (entry) =>
        entry.url === "http://tauri.localhost/" && entry.webSocketDebuggerUrl,
    );
  }, "Mold page");
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error("CDP handshake timeout")),
      10_000,
    );
    socket.addEventListener(
      "open",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
    socket.addEventListener(
      "error",
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
      { once: true },
    );
  });
  let id = 0;
  const pending = new Map();
  socket.addEventListener("message", ({ data }) => {
    const result = JSON.parse(data);
    const callback = pending.get(result.id);
    if (callback) {
      pending.delete(result.id);
      callback(result);
    }
  });
  const command = (method, params) =>
    new Promise((resolve, reject) => {
      const requestId = ++id;
      const timer = setTimeout(() => {
        pending.delete(requestId);
        reject(new Error("CDP timeout: " + method));
      }, 10_000);
      pending.set(requestId, (message) => {
        clearTimeout(timer);
        if (message.error || message.result?.exceptionDetails)
          reject(new Error(JSON.stringify(message)));
        else resolve(message.result);
      });
      socket.send(
        JSON.stringify({
          id: requestId,
          method,
          params,
        }),
      );
    });
  evaluate = async (expression) =>
    (
      await command("Runtime.evaluate", {
        expression,
        returnByValue: true,
        awaitPromise: true,
      })
    ).result.value;
  if (api === "28") {
    await runLegacyMediaSmoke({ evaluate, run, shell, until, output });
  } else {
    const selector = (name) => '[data-test="' + name + '"]';
    const click = async (name) => {
      const point = await evaluate(
        "(() => { const e=document.querySelector(" +
          JSON.stringify(selector(name)) +
          "); if (!e || !e.checkVisibility()) return null; e.scrollIntoView({block: 'nearest'}); const r=e.getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2}; })()",
      );
      assert(point, "Visible control missing: " + name);
      // Trusted input matters: Chromium may skip pushState entries made without
      // user activation when Android Back traverses the WebView history.
      await command("Input.dispatchTouchEvent", {
        type: "touchStart",
        touchPoints: [point],
      });
      await command("Input.dispatchTouchEvent", {
        type: "touchEnd",
        touchPoints: [],
      });
    };
    await until(
      () =>
        evaluate('!!document.querySelector(".mobile-tab[aria-current=page]")'),
      "navigation ready",
    );
    originalTab = await evaluate(
      'document.querySelector(".mobile-tab[aria-current=page]").dataset.test',
    );
    for (const tab of ["generate", "queue", "gallery", "catalog", "hosts"]) {
      await click("mobile-tab-" + tab);
      await until(
        () =>
          evaluate(
            "document.querySelector(" +
              JSON.stringify(selector("mobile-tab-" + tab)) +
              ')?.getAttribute("aria-current") === "page"',
          ),
        "select " + tab,
      );
    }
    await click("mobile-open-settings");
    await until(
      () => evaluate('!!document.querySelector(".is-settings-open")'),
      "settings opens",
    );
    await recordNavigation("before native Back");
    shell("input", "keyevent", "4");
    await until(
      () =>
        evaluate('!!document.querySelector(".mobile-tab[aria-current=page]")'),
      "native Back dismisses settings",
    );
    assert(
      await evaluate(
        'document.querySelector(".mobile-tab[aria-current=page]").dataset.test === "mobile-tab-hosts"',
      ),
      "Back changed the underlying destination",
    );
    const metrics = () =>
      evaluate(
        '({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth, font: parseFloat(getComputedStyle(document.querySelector(".mobile-wordmark")).fontSize) })',
      );
    shell("settings", "put", "system", "font_scale", "1");
    await sleep(500);
    const normal = await metrics();
    shell("settings", "put", "system", "font_scale", "2");
    const large = await until(async () => {
      const m = await metrics();
      return m.font >= normal.font * 1.9 && m;
    }, "live system font scale");
    assert(
      large.scrollWidth <= large.width + 1,
      "Large text causes horizontal overflow",
    );
    assert(
      shell("pidof", "com.utensils.mold") === pid,
      "Text scale recreated the app process",
    );
    run("shell", "screencap", "-p", "/sdcard/mold-app-smoke.png");
    run("pull", "/sdcard/mold-app-smoke.png", output + "/large-text.png");
    writeFileSync(
      output + "/result.json",
      JSON.stringify(
        {
          api: shell("getprop", "ro.build.version.sdk"),
          abi: shell("getprop", "ro.product.cpu.abi"),
          normal,
          large,
          navigation: "five destinations",
          back: "settings dismissed to Machines",
        },
        null,
        2,
      ),
    );
    console.log(
      "Android app navigation, native Back and live text-scale smoke passed",
    );
  }
} catch (error) {
  // Capture the failing state before restoration or emulator teardown. Only
  // control metadata is recorded; no prompt, credentials, or history payloads.
  try {
    writeFileSync(output + "/failure.txt", String(error));
  } catch (captureError) {
    console.error("Could not record Android failure:", captureError);
  }
  try {
    await recordNavigation("failure");
  } catch (captureError) {
    console.error("Could not capture navigation state:", captureError);
  }
  try {
    run("shell", "screencap", "-p", "/sdcard/mold-app-smoke-failure.png");
    run("pull", "/sdcard/mold-app-smoke-failure.png", output + "/failure.png");
    shell("rm", "-f", "/sdcard/mold-app-smoke-failure.png");
  } catch (captureError) {
    console.error("Could not capture Android failure evidence:", captureError);
  }
  for (const [name, args] of [
    ["last-anr.txt", ["dumpsys", "activity", "lastanr"]],
    [
      "system-crash.txt",
      ["logcat", "-d", "-b", "system", "-b", "crash", "-t", "500"],
    ],
  ]) {
    try {
      writeFileSync(output + "/" + name, shell(...args));
    } catch (captureError) {
      console.error("Could not capture " + name + ":", captureError);
    }
  }
  throw error;
} finally {
  shell(
    "settings",
    originalScale === "null" ? "delete" : "put",
    "system",
    "font_scale",
    ...(originalScale === "null" ? [] : [originalScale]),
  );
  if (evaluate && originalTab && socket?.readyState === WebSocket.OPEN) {
    await evaluate(
      "document.querySelector(" +
        JSON.stringify('[data-test="' + originalTab + '"]') +
        ")?.click()",
    ).catch(() => {});
  }
  socket?.close();
  if (port) run("forward", "--remove", "tcp:" + port);
  if (api === "28") {
    shell(
      "pm",
      "revoke",
      "com.utensils.mold",
      "android.permission.WRITE_EXTERNAL_STORAGE",
    );
    shell(
      "pm",
      "revoke",
      "com.utensils.mold",
      "android.permission.READ_EXTERNAL_STORAGE",
    );
  }
}
