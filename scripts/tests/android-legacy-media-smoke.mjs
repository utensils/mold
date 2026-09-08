// API28 native bridge acceptance. The bundled legacy WebView cannot render
// the modern frontend; this exercises the installed app's real permission
// callback and public Downloads path with local fixture bytes, not a render.
import { readFileSync, writeFileSync } from "node:fs";
export async function runLegacyMediaSmoke({
  evaluate,
  run,
  shell,
  until,
  output,
}) {
  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
  };
  await until(
    () =>
      evaluate(
        "typeof window.__TAURI_INTERNALS__ === 'object' && typeof window.__TAURI_INTERNALS__.invoke === 'function'",
      ),
    "native bridge ready",
  );
  const token = Date.now().toString();
  const filename = "mold-permission-" + token + ".stl";
  const destination = "/sdcard/Download/Mold/" + filename;
  const bytes = "solid permission\nendsolid permission\n";
  let requests = 0;
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(request) {
      requests++;
      return new Response(bytes, { headers: { "content-type": "model/stl" } });
    },
  });
  const args = {
    url: "http://10.0.2.2:" + server.port + "/export",
    apiKey: null,
    request: { format: "stl" },
    filename,
    reuseKey: "permission-" + token,
  };
  const invoke = async () =>
    evaluate(
      "window.uatOutcome=null;window.__TAURI_INTERNALS__.invoke('save_export_to_mold_folder'," +
        JSON.stringify(args) +
        ").then(function(value){window.uatOutcome={ok:true,value:value}},function(error){window.uatOutcome={ok:false,error:String(error)}});true",
    );
  async function dialogButton(id) {
    const xml = await until(() => {
      shell("uiautomator", "dump", "/sdcard/mold-permission-uat.xml");
      const value = shell("cat", "/sdcard/mold-permission-uat.xml");
      return value.includes(id) ? value : null;
    }, "permission dialog " + id);
    writeFileSync(output + "/" + id.split("/").pop() + ".xml", xml);
    shell("screencap", "-p", "/sdcard/mold-permission-uat.png");
    run(
      "pull",
      "/sdcard/mold-permission-uat.png",
      output + "/" + id.split("/").pop() + ".png",
    );
    const node = xml.match(
      new RegExp('<node[^>]*resource-id="' + id + '"[^>]*>'),
    )?.[0];
    assert(node, "Permission button missing");
    const bounds = node.match(/bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"/);
    assert(bounds, "Permission bounds missing");
    shell(
      "input",
      "tap",
      String(Math.round((Number(bounds[1]) + Number(bounds[3])) / 2)),
      String(Math.round((Number(bounds[2]) + Number(bounds[4])) / 2)),
    );
  }
  try {
    await invoke();
    await dialogButton(
      "com.android.packageinstaller:id/permission_deny_button",
    );
    const denied = await until(
      () => evaluate("window.uatOutcome"),
      "denied callback",
    );
    assert(
      !denied.ok && denied.error.includes("Storage access"),
      "Denial was not returned: " + JSON.stringify(denied),
    );
    assert(requests === 0, "Denied save downloaded bytes");
    await invoke();

    await dialogButton(
      "com.android.packageinstaller:id/permission_allow_button",
    );
    const granted = await until(
      () => evaluate("window.uatOutcome"),
      "granted callback",
    );
    assert(granted.ok, "Grant failed: " + JSON.stringify(granted));
    run("pull", destination, output + "/" + filename);
    assert(
      readFileSync(output + "/" + filename).equals(Buffer.from(bytes)),
      "Saved bytes mismatch",
    );
    assert(requests === 1, "Unexpected download count: " + requests);
    const result = {
      api: 28,
      denied,
      granted,
      requests,
      filename,
      bytes: bytes.length,
    };
    writeFileSync(output + "/result.json", JSON.stringify(result, null, 2));
    console.log(result);
  } finally {
    server.stop();
    shell("rm", "-f", destination);
    shell(
      "rm",
      "-f",
      "/sdcard/mold-permission-uat.xml",
      "/sdcard/mold-permission-uat.png",
    );
  }
}
