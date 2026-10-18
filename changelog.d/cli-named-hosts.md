- **Named hosts and one global `--host`.** Every remote command now takes a
  global `--host` (before or after the verb), and it, `MOLD_HOST`, and a new
  default host all accept a saved host name, a URL, or a Tailscale IP the
  paired host advertised, so it finds a host paired under its relay origin
  (and says so on stderr); LAN addresses are used as typed. Pairing saves the host under its own name; `mold host list`,
  `mold host use NAME` (or `--clear`) and `mold host rm NAME` manage them.
  Precedence is `--host`, then `MOLD_HOST`, then the default host, then
  `http://localhost:7680`. Commands that only run on this machine now refuse a
  typed `--host` instead of ignoring it.
- **`mold upscale` and `mold video-upscale` honor `MOLD_API_KEY` and paired
  hosts.** They used to build an unauthenticated client of their own.
- **`mold --help` describes `upscale` and `completions` correctly.** Their help
  text had been swapped.
