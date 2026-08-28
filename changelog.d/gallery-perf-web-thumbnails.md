- **Web Library thumbnails persist across reloads.** On a secure origin the
  web app keeps authenticated hosts' tiles in the browser's Cache API (keyed
  by host, filename, content version, and rendition; bounded to 4 000 tiles),
  so a reload paints the grid without one request per tile; keyless hosts
  already rode the browser's HTTP cache. Every surface now asks a current
  server for the display's rendition (`?size=512&fmt=jpeg` on retina), grid
  tiles in the overscan band load at a lower priority than the ones on screen
  and are promoted the moment they scroll into view, and a 2 000-print guard
  pins the number of tiles the web grid mounts.
