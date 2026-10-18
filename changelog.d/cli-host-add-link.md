- **`mold host invite` and `mold host add`: pair the CLI in one step.** On the
  host, `mold host invite` prints the same one-use pairing link the apps show
  (`--qr` draws it for a phone, `--output` keeps the owner-only file); on the
  client, `mold host add '<link>'` pairs and saves the host under its name.
  `mold host add URL` with the host's operator key exchanges it for the
  client's own revocable grant without storing it, saves an app key as a
  Regular connection, and saves a host with authentication off keyless.
  `mold pair` and `mold pairing create` keep working as hidden aliases, and
  `mold pair --invite-file` also reads a file holding a link.
