"""A static file server that sends the two headers a page needs for shared memory (threads)."""
import http.server, sys

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def log_message(self, *args):
        pass

Handler.extensions_map[".wasm"] = "application/wasm"
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
