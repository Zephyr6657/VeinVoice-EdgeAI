#!/usr/bin/env python3
import base64
import json
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import websocket
except ImportError:
    websocket = None


HOST = os.environ.get("ASR_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASR_PROXY_PORT", "8000"))
API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ASR_PROXY_API_KEY", "")
MODEL = os.environ.get("ASR_PROXY_MODEL", "qwen3-asr-flash-realtime")
WS_BASE_URL = os.environ.get(
    "ASR_PROXY_WS_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
)
LANGUAGE = os.environ.get("ASR_PROXY_LANGUAGE", "zh")
TIMEOUT = int(os.environ.get("ASR_PROXY_TIMEOUT", "120"))
CHUNK_SIZE = int(os.environ.get("ASR_PROXY_CHUNK_SIZE", "3200"))
MANUAL_MODE = os.environ.get("ASR_PROXY_MANUAL", "1") != "0"


def build_ws_url():
    parts = urllib.parse.urlsplit(WS_BASE_URL)
    query = urllib.parse.parse_qs(parts.query)
    query["model"] = [MODEL]
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urllib.parse.urlencode(query, doseq=True),
            parts.fragment,
        )
    )


def event_id(prefix):
    return "%s_%d" % (prefix, int(time.time() * 1000))


def send_json(ws, payload):
    ws.send(json.dumps(payload, ensure_ascii=False))


def make_session_update(sample_rate):
    session = {
        "modalities": ["text"],
        "input_audio_format": "pcm",
        "sample_rate": sample_rate,
        "turn_detection": None
        if MANUAL_MODE
        else {
            "type": "server_vad",
            "threshold": 0.0,
            "silence_duration_ms": 400,
        },
    }

    if LANGUAGE:
        session["input_audio_transcription"] = {"language": LANGUAGE}

    return {
        "event_id": event_id("session_update"),
        "type": "session.update",
        "session": session,
    }


def wait_session_ready(ws):
    deadline = time.time() + 15
    last_event = None

    while time.time() < deadline:
        message = ws.recv()
        if not message:
            continue
        data = json.loads(message)
        last_event = data.get("type")
        if last_event in ("session.updated", "session.created"):
            return data
        if last_event == "error":
            raise RuntimeError(json.dumps(data, ensure_ascii=False))

    raise TimeoutError("wait session.updated timeout, last_event=%s" % last_event)


def send_pcm_audio(ws, pcm):
    offset = 0
    total = len(pcm)

    while offset < total:
        chunk = pcm[offset : offset + CHUNK_SIZE]
        offset += len(chunk)
        send_json(
            ws,
            {
                "event_id": event_id("audio"),
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            },
        )
        time.sleep(0.1)

    if MANUAL_MODE:
        send_json(
            ws,
            {
                "event_id": event_id("commit"),
                "type": "input_audio_buffer.commit",
            },
        )

    send_json(
        ws,
        {
            "event_id": event_id("finish"),
            "type": "session.finish",
        },
    )


def recv_transcript(ws):
    deadline = time.time() + TIMEOUT
    final_text = ""
    preview_text = ""

    while time.time() < deadline:
        message = ws.recv()
        if not message:
            continue

        data = json.loads(message)
        event_type = data.get("type")

        if event_type == "conversation.item.input_audio_transcription.text":
            preview_text = (data.get("text") or "") + (data.get("stash") or "")
            if preview_text:
                print("ASR preview:", preview_text)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            final_text = data.get("transcript") or ""
            print("ASR final:", final_text)
        elif event_type == "session.finished":
            return final_text or preview_text
        elif event_type == "error":
            raise RuntimeError(json.dumps(data, ensure_ascii=False))

    raise TimeoutError("wait ASR result timeout")


def qwen_asr_transcribe(pcm, sample_rate=16000):
    if websocket is None:
        raise RuntimeError("missing dependency: pip install websocket-client")
    if not API_KEY:
        raise RuntimeError("set DASHSCOPE_API_KEY first")

    ws_url = build_ws_url()
    headers = [
        "Authorization: Bearer " + API_KEY,
        "OpenAI-Beta: realtime=v1",
    ]

    ws = websocket.create_connection(ws_url, header=headers, timeout=TIMEOUT)
    try:
        print("ASR websocket connected:", ws_url)
        send_json(ws, make_session_update(sample_rate))
        wait_session_ready(ws)
        send_pcm_audio(ws, pcm)
        return recv_transcript(ws)
    finally:
        ws.close()


class ASRHandler(BaseHTTPRequestHandler):
    server_version = "QwenASRHttpProxy/1.0"

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

    def _send_text(self, status, text):
        body = (text or "").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json(
                200,
                {
                    "ok": True,
                    "endpoint": "/asr",
                    "ws_url": build_ws_url(),
                    "model": MODEL,
                    "manual_mode": MANUAL_MODE,
                    "language": LANGUAGE,
                    "websocket_client": websocket is not None,
                    "has_api_key": bool(API_KEY),
                },
            )
            return
        self._send_json(404, {"error": "use POST /asr"})

    def do_POST(self):
        if self.path != "/asr":
            self._send_json(404, {"error": "use POST /asr"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return

        if content_length <= 0:
            self._send_json(400, {"error": "empty PCM body"})
            return

        sample_rate = int(self.headers.get("X-Audio-Sample-Rate", "16000"))
        channels = int(self.headers.get("X-Audio-Channels", "1"))
        if channels != 1:
            self._send_json(400, {"error": "Qwen-ASR proxy expects mono PCM"})
            return

        pcm = self.rfile.read(content_length)
        print("received pcm: %d bytes, sample_rate=%d" % (len(pcm), sample_rate))

        try:
            text = qwen_asr_transcribe(pcm, sample_rate=sample_rate)
        except Exception as exc:
            self._send_json(502, {"error": "Qwen-ASR request failed", "detail": str(exc)})
            return

        self._send_text(200, text)


def main():
    server = ThreadingHTTPServer((HOST, PORT), ASRHandler)
    print("Qwen ASR HTTP proxy listening on http://%s:%d/asr" % (HOST, PORT))
    print("Forwarding to %s" % build_ws_url())
    print("Manual mode: %s, language=%s" % (MANUAL_MODE, LANGUAGE or "auto"))
    if websocket is None:
        print("Missing dependency: run `pip install websocket-client`")
    if not API_KEY:
        print("Warning: DASHSCOPE_API_KEY is not set")
    server.serve_forever()


if __name__ == "__main__":
    main()
