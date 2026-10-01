- Added optional AWS Lambda remote access with an outbound authenticated host
  connector and a normal HTTPS address for CLI, browser, desktop and mobile clients.
  Large uploads and media use private staged transfers; progress streams reconnect
  and saved generation results recover through the durable queue.
- Authenticated servers now allow the browser shell and its assets to load before
  API authentication; API and media permission checks remain enforced.
- Direct relay streams now complete the WebSocket close handshake after both TCP
  directions end, preserving final responses when heartbeat traffic is pending.
