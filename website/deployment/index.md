# Deployment

mold supports multiple deployment modes; from a single GPU machine to cloud GPU
providers.

## Remote Rendering

The simplest setup: run the server on a GPU host, generate from anywhere.

```bash
# On your GPU server
mold serve

# From your laptop
MOLD_HOST=http://gpu-server:7680 mold run "a cat"
```

## Deployment Options

| Method                                    | Best For                          |
| ----------------------------------------- | --------------------------------- |
| [Docker & RunPod](/deployment/docker)     | Cloud GPUs, RunPod pods           |
| [mold runpod CLI](/deployment/runpod-cli) | Integrated pod and volume control |
| [mold lambda CLI](/deployment/lambda-cli) | Private Lambda Cloud web UI       |
| [NixOS](/deployment/nixos)                | NixOS systems, declarative setup  |
| Systemd service                           | Any Linux with NVIDIA GPU         |

## Systemd Service

Two sample units ship in `contrib/`. Pick one.

System-wide (`contrib/mold-server.service`), for a dedicated service user:

```bash
sudo cp contrib/mold-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mold-server
```

User mode (`contrib/mold-server.user.service`), for a single-user GPU box:

```bash
cp contrib/mold-server.user.service ~/.config/systemd/user/mold-server.service
sudo loginctl enable-linger "$USER"   # survive logout; once per box
systemctl --user daemon-reload
systemctl --user enable --now mold-server
```

Key settings:

```ini
[Service]
ExecStart=/usr/local/bin/mold serve --port 7680 --bind 0.0.0.0
Environment=MOLD_LOG=info
# NixOS with a non-flake `/usr/local/bin/mold`: uncomment for CUDA driver access
# Environment=LD_LIBRARY_PATH=/run/opengl-driver/lib
```
