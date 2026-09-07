- **Turntables can export without a backdrop.** The 3-D export sheet now has a
  **Background · Transparent** checkbox, remembered for the next export, so a
  turntable GIF, APNG or WebP can be dropped onto a slide or a README instead
  of carrying mold's slate square with it. APNG and WebP keep the object's
  antialiased outline; a GIF has one transparent colour, so its outline is a
  hard cut. Also on `mold library export --transparent`, the `export_mesh` MCP
  tool, and `POST /api/gallery/export/:filename`.
