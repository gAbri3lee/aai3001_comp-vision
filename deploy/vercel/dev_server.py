"""Run the Vercel functions + static page locally, without the Vercel CLI.

    python deploy/vercel/dev_server.py [port]      # then open http://127.0.0.1:3000
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / 'api'))
import detect, health  # noqa: E402


class Dev(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(here / 'public'), **kw)

    def _route(self, method):
        path = self.path.split('?')[0]
        target = {'/api/detect': detect.handler, '/api/health': health.handler}.get(path)
        if target is None or not hasattr(target, f'do_{method}'):
            return False
        getattr(target, f'do_{method}')(self)
        return True

    def do_GET(self):
        if not self._route('GET'):
            super().do_GET()

    def do_POST(self):
        if not self._route('POST'):
            self.send_error(404)

    def _send(self, *a):  # reuse the detect handler's JSON helper
        return detect.handler._send(self, *a)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f'http://127.0.0.1:{port}')
    ThreadingHTTPServer(('127.0.0.1', port), Dev).serve_forever()
