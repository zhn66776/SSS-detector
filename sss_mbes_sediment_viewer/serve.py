#!/usr/bin/env python3
"""Local server for the viewer: static files, the derived cache under /data, and a
reverse proxy to the existing native ECW tile/crop service under /sss."""
import argparse, atexit, http.server, socket, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
SSS_PORT = 8090


class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'  # keep-alive: browsers otherwise see resets on bursts of tile requests

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def translate_path(self, path):
        # Derived cache is deliberately kept outside web source; expose only /data.
        if path.startswith('/data/'):
            return str(ROOT / 'derived' / path.lstrip('/').split('?')[0])
        return super().translate_path(path)

    def do_GET(self):
        if self.path.startswith('/sss/'):
            return self.proxy()
        return super().do_GET()

    def proxy(self):
        url = f'http://127.0.0.1:{SSS_PORT}/' + self.path[5:]
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                self.relay(r.status, r.headers, r.read())
        except urllib.error.HTTPError as e:
            self.relay(e.code, e.headers, e.read())  # keep the backend's 4xx/5xx and JSON error body
        except Exception as e:
            self.send_error(503, f'Native ECW service unavailable on port {SSS_PORT}: {e}')

    def relay(self, status, headers, body):
        self.send_response(status)
        self.send_header('Content-Type', headers.get('Content-Type', 'application/octet-stream'))
        for name in ('X-Native-Resolution-M', 'X-Crop-Span-M', 'X-Crop-Pixels'):
            if headers.get(name):
                self.send_header(name, headers[name])
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):
        if '/sss/api/tile' not in self.path and '/data/mbes/' not in self.path:
            super().log_message(fmt, *args)


def ensure_sss_service():
    probe = socket.socket()
    running = probe.connect_ex(('127.0.0.1', SSS_PORT)) == 0
    probe.close()
    if running:
        return
    worker = subprocess.Popen([sys.executable, str(ROOT.parent / 'sss_contact_inspector' / 'serve.py'), '--port', str(SSS_PORT)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(worker.terminate)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8091)
    p.add_argument('--no-sss', action='store_true', help='do not start the ECW service; SSS base shows a notice')
    a = p.parse_args()
    try:
        server = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    except OSError as e:
        print(f'port {a.port} is already in use ({e.strerror}). If the viewer is already running, just open '
              f'http://127.0.0.1:{a.port}; otherwise stop the other process (lsof -nP -iTCP:{a.port}) or pass --port.', file=sys.stderr)
        sys.exit(1)
    if not a.no_sss:
        ensure_sss_service()
    print(f'viewer: http://127.0.0.1:{a.port}  (SSS proxy -> 127.0.0.1:{SSS_PORT})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
