import WebSocket from "ws";
import http from "node:http";
import { pipeline } from "node:stream/promises";
import { RelayDuplex } from "./wire.mjs";
import { parameter } from "./aws-store.mjs";
export async function createGuest({
  deadlineMs = 840000,
  token: providedToken,
  url = process.env.WS_ENDPOINT,
  allowInsecureLoopback = false,
} = {}) {
  const endpoint = new URL(url);
  if (
    endpoint.protocol !== "wss:" &&
    !(
      allowInsecureLoopback &&
      endpoint.protocol === "ws:" &&
      ["127.0.0.1", "localhost", "[::1]"].includes(endpoint.hostname)
    )
  )
    throw new Error("Relay requires verified WSS");
  const token =
    providedToken ?? (await parameter(process.env.BRIDGE_TOKEN_PARAMETER));
  const ws = new WebSocket(url, {
    headers: {
      authorization: `Bearer ${token}`,
      "x-mold-relay-role": "frontend",
    },
    maxPayload: 24576,
    perMessageDeflate: false,
    followRedirects: false,
    handshakeTimeout: 15000,
  });
  let ready, accepted, pipe, heartbeat, deadline;
  const connected = new Promise((resolve, reject) => {
    const handshake = setTimeout(() => {
      reject(new Error("Relay host unavailable"));
      ws.terminate();
    }, 15000);
    const cleanup = () => clearTimeout(handshake);
    ws.on("open", () => ws.send(JSON.stringify({ a: "hello", v: 2 })));
    ws.on("message", (bytes) => {
      let frame;
      try {
        frame = JSON.parse(bytes.toString());
      } catch {
        ws.terminate();
        return;
      }
      if (
        frame.a === "ready" &&
        frame.v === 2 &&
        frame.role === "frontend" &&
        typeof frame.sid === "string" &&
        typeof frame.rid === "string"
      ) {
        if (ready) {
          ws.terminate();
          return;
        }
        ready = frame;
        pipe = new RelayDuplex({
          ...ready,
          send: (value) => {
            if (ws.readyState === WebSocket.OPEN)
              ws.send(JSON.stringify(value));
            else pipe.destroy(new Error("Relay closed"));
          },
        });
        pipe.on("error", () => ws.close());
        heartbeat = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN)
            ws.send(JSON.stringify({ a: "heartbeat", v: 2 }));
        }, 30000);
        heartbeat.unref();
        deadline = setTimeout(() => {
          pipe.destroy(new Error("Relay stream deadline"));
          ws.close();
        }, deadlineMs);
        deadline.unref();
      } else if (frame.a === "accept") {
        if (!ready || frame.sid !== ready.sid || frame.rid !== ready.rid) {
          ws.terminate();
          return;
        }
        accepted = true;
        cleanup();
        resolve({
          socket: pipe,
          sid: ready.sid,
          rid: ready.rid,
          close: () => {
            pipe.destroy();
            ws.close();
          },
        });
      } else if (frame.a === "heartbeat") {
        if (frame.v !== 2) ws.terminate();
      } else if (pipe && accepted) pipe.receive(frame);
      else if (frame.a === "cancel") {
        cleanup();
        reject(new Error("Relay host unavailable"));
        ws.close();
      }
    });
    ws.on("error", () => {
      cleanup();
      reject(new Error("Relay connection failed"));
      pipe?.destroy(new Error("Relay connection failed"));
    });
    ws.on("close", () => {
      cleanup();
      clearInterval(heartbeat);
      clearTimeout(deadline);
      reject(new Error("Relay connection closed"));
      pipe?.destroy();
    });
  });
  return connected;
}
const forbidden = new Set([
  "connection",
  "proxy-connection",
  "transfer-encoding",
  "keep-alive",
  "trailer",
  "upgrade",
  "te",
  "forwarded",
]);
export function cleanHeaders(headers, { response = false } = {}) {
  return Object.fromEntries(
    Object.entries(headers).filter(
      ([name]) =>
        !forbidden.has(name.toLowerCase()) &&
        !name.toLowerCase().startsWith("x-forwarded-") &&
        !name.toLowerCase().startsWith("x-mold-viewer-") &&
        (!response || name.toLowerCase() !== "x-mold-relay-object"),
    ),
  );
}
export async function openRequest(
  {
    method = "GET",
    path,
    headers = {},
    body = Buffer.alloc(0),
    size = body?.length ?? 0,
    onForwardAttempt = () => {},
  },
  { guestFactory = createGuest } = {},
) {
  if (
    !path?.startsWith("/") ||
    path.startsWith("//") ||
    /[\r\n]/.test(path) ||
    path.split("?")[0].replace(/\/+$/, "").startsWith("/metrics")
  )
    throw new Error("Forbidden relay path");
  if (!/^[A-Z]+$/.test(method)) throw new Error("Invalid request method");
  const guest = await guestFactory();
  const prepared = {
    ...cleanHeaders(headers),
    "content-length": String(size),
    connection: "close",
  };
  delete prepared["x-amz-content-sha256"];
  delete prepared["x-mold-viewer-authorization"];
  const agent = new http.Agent({ keepAlive: false });
  agent.createConnection = () => guest.socket;
  const promise = new Promise((resolve, reject) => {
    const request = http.request(
      { hostname: "mold-host.invalid", method, path, headers: prepared, agent },
      (response) => resolve({ response, sid: guest.sid, close: guest.close }),
    );
    request.on("error", reject);
    request.on("finish", () => {
      if (!guest.socket.writableEnded) guest.socket.end();
    });
    onForwardAttempt();
    if (Buffer.isBuffer(body)) {
      request.end(body);
    } else pipeline(body, request).catch(reject);
  });
  try {
    return await promise;
  } catch (error) {
    guest.close();
    throw error;
  }
}
