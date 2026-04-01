# Deployment

mold supports multiple deployment modes — from a single GPU machine to cloud GPU
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

| Method                                | Best For                         |
| ------------------------------------- | -------------------------------- |
| [Docker & RunPod](/deployment/docker) | Cloud GPUs, RunPod pods          |
| [NixOS](/deployment/nixos)            | NixOS systems, declarative setup |
| Systemd service                       | Any Linux with NVIDIA GPU        |

## Systemd Service

A sample systemd unit is at `contrib/mold-server.service`:

```bash
cp contrib/mold-server.service ~/.config/systemd/user/
systemctl --user enable --now mold-server
```

Key settings:

```ini
[Service]
ExecStart=/usr/local/bin/mold serve --port 7680 --bind 0.0.0.0
Environment=MOLD_LOG=info
# NixOS: uncomment for CUDA driver access
# Environment=LD_LIBRARY_PATH=/run/opengl-driver/lib
```
