#!/usr/bin/env python3
"""Exercise the real GPU-free CLI, including a proven HTTP round trip."""
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

binary = str(Path(sys.argv[1]).resolve())
seen = []


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        seen.append(self.path)
        if self.path != "/api/models":
            self.send_error(404)
            return
        body = b"[]"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


with tempfile.TemporaryDirectory() as home:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MOLD_")}
    env.update(HOME=home, XDG_CONFIG_HOME=home, XDG_DATA_HOME=home,
               LD_LIBRARY_PATH="", CUDA_VISIBLE_DEVICES="-1", NO_COLOR="1",
               MOLD_HOST="http://127.0.0.1:0")

    def run(*args):
        return subprocess.run([binary, *args], env=env, check=True,
                              capture_output=True, text=True, timeout=30).stdout

    print(run("--version").strip())
    assert "Usage:" in run("--help")
    for shell in ("bash", "zsh", "fish"):
        assert run("completions", shell).strip(), shell
    assert "without a GPU backend" in run("gpu", "list")
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env["MOLD_HOST"] = f"http://127.0.0.1:{server.server_port}"
            assert json.loads(run("list", "--json")) == []
            assert seen == ["/api/models"], seen
        finally:
            server.shutdown()
            thread.join()
print("GPU-free CLI and remote HTTP smoke passed")
