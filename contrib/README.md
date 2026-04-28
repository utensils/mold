# contrib/

Optional integration files for running `mold serve` as a managed service.

## systemd

Two variants are provided. Pick one.

### System-wide (`mold-server.service`)

For boxes where `mold` is installed at `/usr/local/bin/mold` and runs as
a dedicated service user — the typical NixOS / production deploy.

```bash
sudo cp contrib/mold-server.service /etc/systemd/system/
sudo cp contrib/mold-server.env.example /etc/mold/mold.env
sudo chmod 600 /etc/mold/mold.env
sudo $EDITOR /etc/mold/mold.env       # fill in tokens
sudo systemctl daemon-reload
sudo systemctl enable --now mold-server
journalctl -u mold-server -f
```

### User mode (`mold-server.user.service`)

For developer / single-user GPU boxes where `mold` lives inside a user
checkout (e.g. `~/github/mold/target/release/mold`). Inherits no
ambient shell environment, so tokens must come from the env file.

```bash
mkdir -p ~/.config/systemd/user ~/.config/mold
cp contrib/mold-server.user.service ~/.config/systemd/user/mold-server.service
cp contrib/mold-server.env.example ~/.config/mold/server.env
chmod 600 ~/.config/mold/server.env
$EDITOR ~/.config/mold/server.env     # fill in tokens

# Survive ssh logout — without this the service stops when the user
# logs out. Required exactly once per box.
sudo loginctl enable-linger "$USER"

systemctl --user daemon-reload
systemctl --user enable --now mold-server
journalctl --user -u mold-server -f
```

To swap the binary in place after a rebuild:

```bash
cargo build --profile dev-fast -p mold-ai --features cuda,preview,discord,expand,tui,webp,mp4,metrics
systemctl --user restart mold-server
```

### Environment file

`mold-server.env.example` documents the recognised keys. Tokens live
here — never commit a populated copy. The leading `-` on
`EnvironmentFile=` in both unit files makes the file optional, so the
service still starts on a fresh box without one.
