// Run with Bun against an installed DEBUG APK on an emulator. No server or
// generation is needed. CDP controls the real Tauri WebView, not a browser copy.
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";

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
async function until(read, label) {
  const deadline = Date.now() + 30_000;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const value = await read();
      if (value) return value;
    } catch (error) {
      lastError = error;
    }
    await sleep(200);
  }
  throw new Error("Timed out: " + label, { cause: lastError });
}

assert(
  shell("getprop", "ro.kernel.qemu") === "1",
  "This test requires an emulator",
);
mkdirSync(output, { recursive: true });
const originalScale = shell("settings", "get", "system", "font_scale");
let socket;
let originalTab;
let evaluate;
let port;
try {
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
}
