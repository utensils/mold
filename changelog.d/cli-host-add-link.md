- **`mold host invite` and `mold host add`: pair the CLI in one step.** On the
  host, `mold host invite` prints the same one-use pairing link the apps show
  (`--qr` draws it for a phone, `--output` keeps the owner-only file); on the
  client, `mold host add` (then paste the link) pairs and saves the host under
  its name.
  `mold host add URL` with the host's operator key exchanges it for the
  client's own revocable grant without storing it, saves an app key as a
  Regular connection, and saves a host with authentication off keyless.
  `mold pair` and `mold pairing create` keep working as hidden aliases, and
  `mold pair --invite-file` also reads a file holding a link.
- **`mold pairing create` and `mold pair` print like their new commands.**
  `mold pairing create --output` now writes its messages to stderr, leaving
  stdout empty, and `mold pair` prints the `Added <name>` line and checks the
  host once after pairing. Pairing hints in errors now name `mold host invite`
  and `mold host add`.
- **A saved host's own credential always wins over `MOLD_API_KEY`.** A
  client now sends `MOLD_API_KEY` only to a host with no saved credential; a
  saved Regular key (or a host saved keyless) and a Secure pairing always take
  precedence, so a key exported for one machine is never sent to another.
