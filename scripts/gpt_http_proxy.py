#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = os.environ.get("GPT_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("GPT_PROXY_PORT", "8000"))
UPSTREAM_URL = os.environ.get(
    "GPT_PROXY_UPSTREAM_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
UPSTREAM_API_KEY = (
    os.environ.get("DASHSCOPE_API_KEY")
    or os.environ.get("GPT_PROXY_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
TIMEOUT = int(os.environ.get("GPT_PROXY_TIMEOUT", "60"))


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "GPTHttpProxy/1.0"

    def log_message(self, fmt, *args):
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json(
                200,
                {
                    "ok": True,
                    "endpoint": "/chat",
                    "upstream": UPSTREAM_URL,
                    "has_api_key": bool(UPSTREAM_API_KEY),
                },
            )
            return
        self._send_json(404, {"error": "use POST /chat"})

    def do_POST(self):
        if self.path not in ("/chat", "/v1/chat/completions"):
            self._send_json(404, {"error": "use POST /chat"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return

        if content_length <= 0:
            self._send_json(400, {"error": "empty request body"})
            return

        body = self.rfile.read(content_length)
        auth = self.headers.get("Authorization", "")
        if UPSTREAM_API_KEY:
            auth = "Bearer " + UPSTREAM_API_KEY

        if not auth:
            self._send_json(
                401,
                {
                    "error": "missing Authorization header; set DASHSCOPE_API_KEY/GPT_PROXY_API_KEY on PC"
                },
            )
            return

        request = urllib.request.Request(
            UPSTREAM_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": auth,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                response_body = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            response_body = exc.read()
            status = exc.code
            content_type = exc.headers.get("Content-Type", "application/json")
        except Exception as exc:
            self._send_json(502, {"error": "upstream request failed", "detail": str(exc)})
            return

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def main():
    server = ThreadingHTTPServer((HOST, PORT), ProxyHandler)
    print("GPT HTTP proxy listening on http://%s:%d/chat" % (HOST, PORT))
    print("Forwarding to %s" % UPSTREAM_URL)
    if UPSTREAM_API_KEY:
        print("Using API key from PC environment")
    else:
        print("Warning: DASHSCOPE_API_KEY/GPT_PROXY_API_KEY is not set")
    server.serve_forever()


if __name__ == "__main__":
    main()
