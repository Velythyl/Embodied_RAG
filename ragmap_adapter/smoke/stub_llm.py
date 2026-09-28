"""A stand-in OpenAI chat-completions server for the CI smoke test.

Answers every request in upstream's AREA_NAME / AREA_SUMMARY format, so the
container's whole path (graph export, upstream forest build, LLM round trip,
area export) runs without a model. Stdlib only.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

REPLY = "AREA_NAME: <<smoke_test_zone>>\nAREA_SUMMARY: <<A smoke-test area.>>"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        payload = json.dumps({
            "id": "stub", "object": "chat.completion", "created": 0, "model": body.get("model", "stub"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
