// Run with Bun against the isolated com.utensils.mold.redesignuat DEBUG APK.
// Set MOLD_ANDROID_TEST_PACKAGE to that ID; camera permission is reset. No server or
// generation is needed. CDP controls the real Tauri WebView, not a browser copy.
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import {
  resolveDeadlineMs,
  until as untilWith,
} from "./lib/android-smoke-wait.mjs";

const adb = process.env.ADB ?? "adb";
const serial = process.env.ANDROID_SERIAL ?? "emulator-5554";
const output =
  process.env.MOLD_ANDROID_EVIDENCE ?? "/tmp/mold-android-identity-picker";
// A `uiautomator dump` of a full screen and an `exec-out screencap` both
// outgrow execFileSync's 1 MB default; ENOBUFS there kills the run outright.
const ADB_MAX_BUFFER = 32 * 1024 * 1024;
const run = (...args) =>
  execFileSync(adb, ["-s", serial, ...args], {
    encoding: "utf8",
    maxBuffer: ADB_MAX_BUFFER,
  }).trim();
const shell = (...args) => run("shell", ...args);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
/** The deadline every wait in this script shares; see the shared module. */
const timeoutMs = resolveDeadlineMs(process.env.MOLD_ANDROID_SMOKE_TIMEOUT_MS);
const until = (read, label) => untilWith(read, label, { timeoutMs });

assert(
  shell("getprop", "ro.kernel.qemu") === "1",
  "This test requires an emulator",
);
mkdirSync(output, { recursive: true });
const testPackage = process.env.MOLD_ANDROID_TEST_PACKAGE;
assert(
  testPackage === "com.utensils.mold.redesignuat",
  "Use the isolated redesign UAT package",
);
shell("pm", "revoke", testPackage, "android.permission.CAMERA");
shell(
  "pm",
  "clear-permission-flags",
  testPackage,
  "android.permission.CAMERA",
  "user-set",
  "user-fixed",
);
let socket;
let evaluate;
let port;
try {
  shell(
    "am",
    "start",
    "-n",
    "com.utensils.mold.redesignuat/com.utensils.mold.MainActivity",
  );
  const pid = await until(
    () => shell("pidof", "com.utensils.mold.redesignuat"),
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
  await until(
    () =>
      evaluate(
        "typeof window.__TAURI_INTERNALS__ === 'object' && typeof window.__TAURI_INTERNALS__.invoke === 'function'",
      ),
    "bridge",
  );

  const results = [];
  const before = await evaluate(
    "localStorage.getItem('mold.mobile.generate-draft.v1')",
  );
  const permissionButton = async (id) => {
    const bounds = await until(() => {
      shell("uiautomator", "dump", "/sdcard/mold-picker-uat.xml");
      const xml = shell("cat", "/sdcard/mold-picker-uat.xml");
      const node = xml.match(
        new RegExp('<node[^>]*resource-id="[^"]*' + id + '"[^>]*>'),
      );
      return node?.[0]
        .match(/bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"/)
        ?.slice(1)
        .map(Number);
    }, id);
    shell(
      "input",
      "tap",
      String(Math.floor((bounds[0] + bounds[2]) / 2)),
      String(Math.floor((bounds[1] + bounds[3]) / 2)),
    );
  };
  for (const source of [
    "camera-deny",
    "library-after-denial",
    "camera",
    "camera-granted",
    "library",
  ]) {
    await evaluate(
      `window.__pickerUat = null; window.__TAURI_INTERNALS__.invoke('pick_identity_photo', {source: '${source.startsWith("camera") ? "camera" : "library"}'}).then(value => {window.__pickerUat = {ok:true,value};}, error => {window.__pickerUat = {ok:false,error:String(error)};}); true`,
    );
    if (source === "camera-deny") {
      await permissionButton("permission_deny_button");
      const denied = await until(
        () => evaluate("window.__pickerUat"),
        "camera denial callback",
      );
      assert(
        !denied.ok && denied.error.includes("Camera access is required"),
        JSON.stringify(denied),
      );
      results.push({ source, result: denied });
      writeFileSync(
        output + "/picker-cancellation.json",
        JSON.stringify({ results }, null, 2),
      );
      continue;
    }
    if (source === "camera")
      await permissionButton("permission_allow_foreground_only_button");
    await sleep(1200);
    const activity = shell("dumpsys", "activity", "activities")
      .split("\n")
      .filter(
        (line) =>
          line.includes("mResumedActivity") ||
          line.includes("topResumedActivity"),
      )
      .join("\n");
    const early = await evaluate("window.__pickerUat");
    assert(!early, source + " did not open: " + JSON.stringify(early));
    writeFileSync(
      output + "/" + source + ".png",
      execFileSync(adb, ["-s", serial, "exec-out", "screencap", "-p"], {
        maxBuffer: ADB_MAX_BUFFER,
      }),
    );
    // Deliberately NOT wrapped in `actUntil` like the CI smoke test's Back is.
    // The hazard is the same — a dropped press waits out the whole deadline —
    // but this script is manual-only (no workflow runs it; it needs a device
    // and the isolated redesign UAT package), so the change could not be
    // verified before shipping. It gains the shared deadline and the richer
    // timeout message either way.
    shell("input", "keyevent", "KEYCODE_BACK");
    const result = await until(
      () => evaluate("window.__pickerUat"),
      source + " cancellation callback",
    );
    assert(
      result.ok && result.value.cancelled === true,
      source + " cancellation: " + JSON.stringify(result),
    );
    assert(!result.value.dataB64, "Cancellation returned media bytes");
    results.push({ source, activity, result });
    writeFileSync(
      output + "/picker-cancellation.json",
      JSON.stringify({ results }, null, 2),
    );
  }
  const after = await evaluate(
    "localStorage.getItem('mold.mobile.generate-draft.v1')",
  );
  assert(before === after, "Picker cancellation changed the composer draft");
  const residue = shell("run-as", testPackage, "ls", "cache/identity");
  assert(residue === "", "Camera cancellation left temporary files");
  writeFileSync(
    output + "/picker-cancellation.json",
    JSON.stringify(
      { results, draftUnchanged: true, cameraCacheEmpty: true },
      null,
      2,
    ),
  );
  console.log(JSON.stringify(results));
} finally {
  socket?.close();
  if (port) run("forward", "--remove", "tcp:" + port);
  shell("rm", "-f", "/sdcard/mold-picker-uat.xml");
}
